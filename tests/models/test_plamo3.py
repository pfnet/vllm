# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.plamo3 import Plamo3RMSNorm


@pytest.mark.parametrize("offset", [1.0, 1.0 / 5, 1.0 / (5**1.5)])
@pytest.mark.parametrize("with_residual", [False, True])
def test_plamo3_rms_norm(default_vllm_config, offset, with_residual):
    loaded_weight = torch.linspace(-0.2, 0.2, 16)
    x = torch.randn(3, 16)
    residual = torch.randn_like(x) if with_residual else None

    reference = RMSNorm(16)
    reference.weight.data.copy_(loaded_weight + offset)

    norm = Plamo3RMSNorm(16, eps=1e-6, offset=offset)
    norm.weight_loader(norm.weight, loaded_weight)

    assert torch.equal(norm.weight, loaded_weight)
    assert "weight_with_offset" not in norm.state_dict()

    reference_residual = residual.clone() if residual is not None else None
    actual_residual = residual.clone() if residual is not None else None
    expected = reference.forward_native(x.clone(), reference_residual)
    actual = norm.forward_native(x.clone(), actual_residual)
    if with_residual:
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    else:
        assert torch.equal(actual, expected)
