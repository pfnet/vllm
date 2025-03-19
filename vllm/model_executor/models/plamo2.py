# SPDX-License-Identifier: Apache-2.0
"""Inference-only PLaMo2 model."""
import enum
import math
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn, selective_state_update)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader, default_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.interfaces import (HasInnerState, IsHybrid,
                                                   SupportsPP, SupportsV0Only)
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LinearType(str, enum.Enum):
    Normal = "normal"
    Fp8 = "fp8"
    Fp8Retain = "fp8-retain"


# Only used for type hinting and PlamoPreTrainedModel.config_class.
class PlamoConfig(PretrainedConfig):  # type: ignore
    model_type: str = "plamo"

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = False,
        # Attention
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        hidden_size_per_head: int = 128,
        max_position_embeddings: int = 2048,
        attention_window_size: int = 2048,
        # Mamba
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_num_heads: int = 64,
        mamba_step: int = 2,
        mamba_chunk_size: int = 256,
        # MLP
        intermediate_size: int = 13312,
        # Tokenizer
        vocab_size: int = 32000,
        tokenizer_class: str = "PlamoTokenizer",
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # MoE
        n_expert: Optional[int] = None,
        k_expert: Optional[int] = None,
        expert_dropout: float = 0.0,
        capacity_factor: float = 1.0,
        group_size: int = 1024,
        sparse_step: Optional[int] = None,
        sparse_intermediate_size: Optional[int] = None,
        shared_intermediate_size: Optional[int] = None,
        # FP8
        linear_type: LinearType = LinearType.Normal,
        fp8_accum_dtype: Optional[str] = None,
        # Evaluation
        eval_attention_n_bit: Optional[int] = None,
        eval_mlp_n_bit: Optional[int] = None,
        eval_offload_moe: bool = False,
        use_cache: bool = True,
        use_predefined_initial_state: bool = False,
        **kwargs: Any,
    ) -> None:
        # max_position_embeddings is often used to determine the max length
        # during inference, but samba should have extrapolation abilities
        self.max_position_embeddings = max(10 * 1024 * 1024,
                                           max_position_embeddings)
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_head = hidden_size_per_head
        self.num_key_value_heads = num_key_value_heads
        self.attention_window_size = attention_window_size

        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_num_heads = mamba_num_heads
        self.mamba_step = mamba_step
        self.mamba_chunk_size = mamba_chunk_size

        self.intermediate_size = intermediate_size

        self.vocab_size = vocab_size

        self.n_expert = n_expert
        self.k_expert = k_expert
        self.sparse_intermediate_size = sparse_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.expert_dropout = expert_dropout
        self.capacity_factor = capacity_factor
        self.group_size = group_size
        self.sparse_step = sparse_step

        self.linear_type = linear_type
        self.fp8_accum_dtype = fp8_accum_dtype

        self.eval_attention_n_bit = eval_attention_n_bit
        self.eval_mlp_n_bit = eval_mlp_n_bit
        self.eval_offload_moe = eval_offload_moe
        self.use_cache = use_cache

        self.use_predefined_initial_state = use_predefined_initial_state

        super().__init__(
            tokenizer_class=tokenizer_class,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class PlamoPreTrainedModel(PreTrainedModel):  # type: ignore
    config_class = PlamoConfig
    _no_split_modules: List[str]
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PlamoDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module: torch.nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


def get_initial_dt_bias(num_heads: int) -> torch.Tensor:
    dt_min = 0.001
    dt_max = 0.1
    dt = torch.exp(
        torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) +
        math.log(dt_min))
    dt = torch.clamp(dt, 1e-4)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    return inv_dt


def is_mamba(config: PlamoConfig, i: int) -> bool:
    assert config.mamba_step > 1

    if config.num_hidden_layers <= (config.mamba_step // 2):
        # use attention in last layer
        return i != config.num_hidden_layers - 1
    return (i % config.mamba_step) != (config.mamba_step // 2)


def _swiglu(h: torch.Tensor) -> torch.Tensor:
    h0, h1 = h.chunk(2, dim=-1)
    return torch.nn.functional.silu(h0) * h1


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class Plamo2MambaMixer(nn.Module):
    # TODO(Shinichi): Rebase on Mamba2 implementation.

    def __init__(self, config: PlamoConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = (config.mamba_num_heads *
                                  config.hidden_size_per_head)
        tp_size = get_tensor_model_parallel_world_size()
        self.intermediate_size_per_tp_worker = self.intermediate_size // tp_size
        self.hidden_size_per_head = config.hidden_size_per_head
        self.num_heads = config.mamba_num_heads
        self.time_step_rank = max(64, self.hidden_size // 16)
        self.use_conv_bias = False
        self.use_bias = False
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.intermediate_size,
            bias=self.use_conv_bias,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=self.use_bias,
            prefix=f"{prefix}.in_proj",
        )
        # selective projection used to make dt, B and C input dependent
        self.x_proj = RowParallelLinear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
            prefix=f"{prefix}.x_proj",
        )
        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(self.time_step_rank,
                                            self.num_heads,
                                            bias=True,
                                            skip_bias_add=True)

        self.A = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_tp_worker,
                self.ssm_state_size,
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(self.intermediate_size_per_tp_worker))

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True,
        )
        # The activation function is fixed to SiLU.
        self.activation = "silu"

        self.dt_layernorm = RMSNorm(self.time_step_rank,
                                    eps=config.rms_norm_eps)
        self.b_layernorm = RMSNorm(self.ssm_state_size,
                                   eps=config.rms_norm_eps)
        self.c_layernorm = RMSNorm(self.ssm_state_size,
                                   eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                mamba_cache_params: MambaCacheParams):

        attn_metadata: AttentionMetadata = get_forward_context().attn_metadata

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)[0]
        gate, hidden_states = projected_states.transpose(0, 1).chunk(2, dim=-2)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if attn_metadata.query_start_loc is not None \
            and attn_metadata.context_lens_tensor is not None:
            # |---------- N-1 iteration --------|
            # |---------------- N iteration ---------------------|
            # |- tokenA -|......................|-- newTokens ---|
            # |---------- context_len ----------|
            # |-------------------- seq_len ---------------------|
            #                                   |-- query_len ---|
            hidden_states = causal_conv1d_fn(
                hidden_states,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=mamba_cache_params.conv_state,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                cache_indices=mamba_cache_params.state_indices_tensor,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            hidden_states = causal_conv1d_update(
                hidden_states.transpose(0, 1),
                mamba_cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_cache_params.state_indices_tensor)
            hidden_states = hidden_states.transpose(0, 1)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(-2, -1))[0]

        # Splitting the ssm_parameters as in modeling_plamo.py.
        B, C, time_step = torch.split(
            ssm_parameters,
            [self.ssm_state_size, self.ssm_state_size, self.time_step_rank],
            dim=-1,
        )
        time_step = self.dt_layernorm(time_step.contiguous())
        B = self.b_layernorm(B.contiguous())
        C = self.c_layernorm(C.contiguous())

        discrete_time_step = self.dt_proj(time_step)[0].transpose(-2, -1)
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = (self.dt_proj.bias.float() if hasattr(
            self.dt_proj, "bias") else None)

        # Broadcasting as in modeling_plamo.py.
        discrete_time_step = discrete_time_step.transpose(
            0, 1)[..., None].expand(-1, -1, self.hidden_size_per_head)
        discrete_time_step = discrete_time_step.reshape(
            -1, self.intermediate_size_per_tp_worker).transpose(0, 1)
        time_proj_bias = time_proj_bias[...,
                                        None].expand(-1,
                                                     self.hidden_size_per_head)
        time_proj_bias = time_proj_bias.reshape(-1)

        if attn_metadata.query_start_loc is not None \
            and attn_metadata.context_lens_tensor is not None:
            scan_outputs = selective_scan_fn(
                hidden_states,
                mamba_cache_params.ssm_state,
                discrete_time_step,
                self.A,
                B.transpose(-2, -1),
                C.transpose(-2, -1),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=mamba_cache_params.state_indices_tensor,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            scan_outputs = selective_state_update(
                mamba_cache_params.ssm_state,
                hidden_states.transpose(0, 1),
                discrete_time_step.transpose(0, 1),
                self.A,
                B,
                C,
                self.D,
                gate.transpose(0, 1),
                time_proj_bias,
                dt_softplus=True,
                state_batch_indices=mamba_cache_params.state_indices_tensor)
            scan_outputs = scan_outputs.transpose(0, 1)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(-2,
                                                                     -1))[0]
        return contextualized_states


class Plamo2MoE(nn.Module):

    def __init__(self,
                 config: PlamoConfig,
                 num_experts: Optional[int] = None,
                 top_k: Optional[int] = None,
                 params_dtype: Optional[torch.dtype] = None,
                 tp_size: Optional[int] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        assert num_experts is None or num_experts <= 1, "MoE not supported"
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size, [self.intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(self.intermediate_size,
                                           self.hidden_size,
                                           bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.gate_up_proj(hidden_states)[0]
        h = _swiglu(h)
        output, _ = self.down_proj(h)
        return output  # type: ignore


class DenseMLP(Plamo2MoE):

    def __init__(self,
                 config: PlamoConfig,
                 params_dtype: Optional[torch.dtype] = None,
                 tp_size: Optional[int] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__(config,
                         num_experts=1,
                         top_k=1,
                         params_dtype=params_dtype,
                         tp_size=tp_size,
                         quant_config=quant_config)


class Plamo2MambaDecoderLayer(nn.Module):

    def __init__(self,
                 config: PlamoConfig,
                 layer_idx: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 max_model_len: int | None = None,
                 **kwargs) -> None:
        super().__init__()
        self.config = config
        self.mamba = Plamo2MambaMixer(config)

        ffn_layer_class = DenseMLP
        self.mlp = ffn_layer_class(config, quant_config=quant_config)
        self.pre_mixer_norm = RMSNorm(config.hidden_size,
                                      eps=config.rms_norm_eps)
        self.post_mixer_norm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_mlp_norm = RMSNorm(config.hidden_size,
                                    eps=config.rms_norm_eps)
        self.post_mlp_norm = RMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_mixer_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_mixer_norm(
                hidden_states, residual)

        hidden_states = self.mamba(hidden_states, mamba_cache_params)
        hidden_states = self.post_mixer_norm(hidden_states)
        # Fully Connected
        hidden_states, residual = self.pre_mlp_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_norm(hidden_states)
        return hidden_states, residual


class Plamo2AttentionDecoderLayer(nn.Module):

    def __init__(self,
                 config: PlamoConfig,
                 layer_idx: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 max_model_len: int | None = None,
                 prefix: str = "",
                 **kwargs) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size_per_head
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)

        self.rope_theta = config.rope_theta if hasattr(config,
                                                       "rope_theta") else 10000
        self.rope_scaling = config.rope_scaling if hasattr(
            config, "rope_scaling") else None

        assert max_model_len is not None, "max_model_len must be provided"
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_model_len,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        self.q_norm = RMSNorm(config.hidden_size_per_head,
                              eps=config.rms_norm_eps)
        self.q_norm.weight = torch.nn.Parameter(
            torch.ones((self.num_heads, config.hidden_size_per_head)))
        set_weight_attrs(self.q_norm.weight,
                         {"weight_loader": sharded_weight_loader(0)})
        self.k_norm = RMSNorm(config.hidden_size_per_head,
                              eps=config.rms_norm_eps)
        self.k_norm.weight = torch.nn.Parameter(
            torch.ones((self.num_kv_heads, config.hidden_size_per_head)))
        set_weight_attrs(self.k_norm.weight,
                         {"weight_loader": sharded_weight_loader(0)})

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

        ffn_layer_class = DenseMLP
        self.mlp = ffn_layer_class(config, quant_config=quant_config)
        self.pre_mixer_norm = RMSNorm(config.hidden_size,
                                      eps=config.rms_norm_eps)
        self.post_mixer_norm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_mlp_norm = RMSNorm(config.hidden_size,
                                    eps=config.rms_norm_eps)
        self.post_mlp_norm = RMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_shape = q.shape
        q = q.reshape(q_shape[:-1] + self.q_norm.weight.shape)
        q = self.q_norm.forward_native(q).reshape(q_shape)
        k_shape = k.shape
        k = k.reshape(k_shape[:-1] + self.k_norm.weight.shape)
        k = self.k_norm.forward_native(k).reshape(k_shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_mixer_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_mixer_norm(
                hidden_states, residual)

        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = self.post_mixer_norm(hidden_states)
        # Fully Connected
        hidden_states, residual = self.pre_mlp_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_norm(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Plamo2AttentionDecoderLayer,
    "mamba": Plamo2MambaDecoderLayer
}


class Plamo2Model(PlamoPreTrainedModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config.model_config.hf_config)

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.embed_tokens",
        )

        extra_kwargs = {"is_lora_enabled": bool(vllm_config.lora_config)}

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = ALL_DECODER_LAYER_TYPES[
                config.layers_block_type[layer_idx]]
            return layer_class(
                config,
                layer_idx,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
                max_model_len=vllm_config.scheduler_config.max_model_len,
                **extra_kwargs)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.final_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        mamba_cache_index = 0
        for i, layer in enumerate(
                self.layers[self.start_layer:self.end_layer]):
            i = i + self.start_layer
            layer_mamba_cache_params = None
            if isinstance(layer, Plamo2MambaDecoderLayer):
                layer_mamba_cache_params = mamba_cache_params.at_layer_idx(
                    mamba_cache_index)
                mamba_cache_index += 1

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                mamba_cache_params=layer_mamba_cache_params)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class Plamo2ForCausalLM(PlamoPreTrainedModel, HasInnerState, SupportsPP,
                        IsHybrid, SupportsV0Only):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        scheduler_config = vllm_config.scheduler_config
        assert not vllm_config.cache_config.enable_prefix_caching, \
            "PLaMo2 currently does not support prefix caching"

        super().__init__(config)
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = scheduler_config

        # TODO(Shinichi): Remove this workaround.
        # vllm.model_executor.models.interfaces.IsHybrid requires
        # self.config.layers_block_type to be set.
        self.config.layers_block_type = [
            "mamba" if is_mamba(self.config, i) else "attention"
            for i in range(self.config.num_hidden_layers)
        ]
        # ModelConfig.get_head_size assumes head_dim is set or calculated as
        # hidden_size // num_attention_heads. However, this is not always
        # the case for PLaMo2, as indicated by the FIXME comment.
        self.config.head_dim = self.config.hidden_size_per_head

        self.model = Plamo2Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))
        self.vocab_size = self.config.vocab_size
        self.unpadded_vocab_size = self.config.vocab_size
        num_embeddings = ((self.vocab_size + 15) // 16) * 16
        self.lm_head = ParallelLMHead(
            num_embeddings,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            prefix=f"{prefix}.lm_head",
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.config.vocab_size)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs):
        if self.mamba_cache is None:
            num_mamba_layers = self.model_config.get_num_layers_by_block_type(
                self.vllm_config.parallel_config, LayerBlockType.mamba)
            self.mamba_cache = MambaCacheManager(
                self.vllm_config, self.lm_head.weight.dtype, num_mamba_layers,
                *self._get_mamba_cache_shape())

        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.model(input_ids, positions, mamba_cache_params,
                                   intermediate_tensors, inputs_embeds)
        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = (self.config.mamba_num_heads *
                       self.config.hidden_size_per_head)
        conv_state_shape = (
            hidden_size // world_size,
            self.config.mamba_d_conv - 1,
        )
        temporal_state_shape = (
            hidden_size // world_size,
            self.config.mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:

            # Both tie_word_embeddings=True and lm_head.weight in the safetensor
            # at the same time causes dict key access error.
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                assert "lm_head.weight" not in params_dict
                continue

            # Update the weight names to be compatible with the vllm version
            # of the model.
            # Do not change the order of the replacements.
            replacements = {
                # Skip PlamoDecoderLayers.
                ".layers.layers": ".layers",
                # Skip PlmoDecoderLayer.
                ".mixer": "",
                # Rename the final layernorm of the model.
                "model.norm.weight": "model.final_layernorm.weight",
                # Rename the RMSNorm layers' weights.
                ".q_weight": ".q_norm.weight",
                ".k_weight": ".k_norm.weight",

                # Rename each mamba layer's components.
                ".A_log": ".mamba.A",
                ".B_norm_weight": ".mamba.b_layernorm.weight",
                ".C_norm_weight": ".mamba.c_layernorm.weight",
                ".dt_norm_weight": ".mamba.dt_layernorm.weight",
                ".bcdt_proj.weight": ".mamba.x_proj.weight",
                ".conv1d.weight": ".mamba.conv1d.weight",
                ".in_proj.weight": ".mamba.in_proj.weight",
                ".out_proj.weight": ".mamba.out_proj.weight",
                ".D": ".mamba.D",
                ".dt_bias": ".mamba.dt_proj.bias",
                ".dt_proj.weight": ".mamba.dt_proj.weight",
            }
            # Apply replacements based on the defined mappings
            for old, new in replacements.items():
                if old in name:
                    name = name.replace(old, new)

            # Broadcast the loaded weight to match the model's parameter shape.
            if ".A" in name:
                loaded_weight = loaded_weight[:, None, None].expand(
                    -1, self.config.hidden_size_per_head,
                    self.config.mamba_d_state)
                loaded_weight = loaded_weight.reshape(
                    -1, self.config.mamba_d_state)
            elif ".D" in name:
                loaded_weight = loaded_weight[:, None].expand(
                    -1, self.config.hidden_size_per_head)
                loaded_weight = loaded_weight.reshape(-1)
            elif ".mamba.in_proj.weight" in name:
                # Rearranging mamba's in_proj weights to fit vLLM's expectation.
                loaded_weight = loaded_weight.transpose(0, 1).reshape(
                    self.config.hidden_size, self.config.mamba_num_heads, -1)
                gate_weight, hidden_states_weight = loaded_weight.chunk(2,
                                                                        dim=-1)
                gate_weight = gate_weight.reshape(self.config.hidden_size, -1)
                hidden_states_weight = hidden_states_weight.reshape(
                    self.config.hidden_size, -1)
                loaded_weight = torch.cat([gate_weight, hidden_states_weight],
                                          dim=-1).transpose(0, 1)
            # Offset parameter with vllm's RMSNorm haven't been supported yet.
            if ".pre_mixer_norm" in name:
                loaded_weight += 1.0
            elif ".post_mixer_norm" in name:
                loaded_weight += 1.0 / 5
            elif ".pre_mlp_norm" in name:
                loaded_weight += 1.0
            elif ".post_mlp_norm" in name:
                loaded_weight += 1.0 / (5**1.5)
            elif "model.final_layernorm.weight" in name:
                loaded_weight += 1.0
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
