# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any

# Tag tokenization in the PLaMo-3 vocabulary
# ============================================
# Each "<|plamo:begin_NAME:plamo|>" or "<|plamo:end_NAME:plamo|>" tag is
# **not** a single token; it encodes as at least three tokens:
#   prefix token  +  name token(s)  +  suffix token
#   e.g. BEGIN_THINK_TAG = <|plamo:begin_ (256) + think (21279) + :plamo|> (258)
#        END_THINK_TAG   = <|plamo:end_  (257) + think (21279) + :plamo|> (258)
# The streaming parsers therefore hold back the trailing portion of the buffer
# whenever its tail is a prefix of such a tag (see compute_safe_until).
#
# The only exception is EOT_TAG ("<|plamo:tag|>"), which *is* a single token
# and therefore never appears partially in the buffer.

BEGIN_TOOL_REQUESTS_TAG = "<|plamo:begin_tool_requests:plamo|>"
END_TOOL_REQUESTS_TAG = "<|plamo:end_tool_requests:plamo|>"
BEGIN_TOOL_REQUEST_TAG = "<|plamo:begin_tool_request:plamo|>"
END_TOOL_REQUEST_TAG = "<|plamo:end_tool_request:plamo|>"
BEGIN_TOOL_NAME_TAG = "<|plamo:begin_tool_name:plamo|>"
END_TOOL_NAME_TAG = "<|plamo:end_tool_name:plamo|>"
# BEGIN_TOOL_ARGS_TAG is not a simple delimiter; it is a composite
# constant that combines the arguments block start tag
# (<|plamo:begin_tool_arguments:plamo|>) with the llguidance JSON
# constraint control tokens (<|plamo:constrain|>json<|plamo:msg|>).
# These are intentionally grouped together because the model generates
# this exact token sequence.
BEGIN_TOOL_ARGS_TAG = (
    "<|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>"
)
END_TOOL_ARGS_TAG = "<|plamo:end_tool_arguments:plamo|>"
EOT_TAG = "<|plamo:tag|>"
BEGIN_THINK_TAG = "<|plamo:begin_think:plamo|>"
END_THINK_TAG = "<|plamo:end_think:plamo|>"

# All atomic PLaMo-3 special-token tags (each is a single "<|plamo:...|>" unit).
# BEGIN_TOOL_ARGS_TAG is a composite of three of these, so it is decomposed here
# into begin_tool_arguments + constrain + msg.
_ALL_SPECIAL_TAGS: list[str] = [
    BEGIN_TOOL_REQUESTS_TAG,
    END_TOOL_REQUESTS_TAG,
    BEGIN_TOOL_REQUEST_TAG,
    END_TOOL_REQUEST_TAG,
    BEGIN_TOOL_NAME_TAG,
    END_TOOL_NAME_TAG,
    "<|plamo:begin_tool_arguments:plamo|>",
    "<|plamo:constrain|>",
    "<|plamo:msg|>",
    END_TOOL_ARGS_TAG,
    EOT_TAG,
    BEGIN_THINK_TAG,
    END_THINK_TAG,
]

_SPECIAL_TOKEN_PREFIX = "<|plamo:"


def strip_trailing_partial_marker(text: str) -> str:
    """Strip a trailing *incomplete* PLaMo-3 special-token fragment from ``text``.

    When generation is cut by ``max_tokens`` mid-marker (e.g. only the single
    ``<|plamo:begin_`` anchor token is emitted before a length stop), the
    non-streaming parsers find no complete tag and pass the partial marker
    through as content. Complete tags are already removed by the parsers, so any
    residual ``<|plamo:...`` run at the tail is necessarily an incomplete marker
    — drop it so special-token markup never reaches user-visible content.

    User content never legitimately contains the ``<|plamo:`` special-token
    prefix, so this only fires on truncation artifacts. The streaming paths
    achieve the same hold-back via ``compute_safe_until``.
    """
    idx = text.rfind(_SPECIAL_TOKEN_PREFIX)
    if idx == -1:
        return text
    tail = text[idx:]
    for tag in _ALL_SPECIAL_TAGS:
        # A proper prefix of a tag is an incomplete marker; a complete tag
        # (tail == tag) is left alone (the parsers own complete-tag removal).
        if tag.startswith(tail) and tail != tag:
            return text[:idx]
    return text


def strip_at_eot(text: str) -> str:
    return text.split(EOT_TAG, maxsplit=1)[0]


def compute_safe_until(buf: str, floor: int, tags: list[tuple[str, str]]) -> int:
    """Compute the largest buffer position that can be safely flushed.

    Holds back any *strict* tail of ``buf`` that is a prefix of any tag so
    that the downstream ``find(TAG, emit_pos)`` can still locate the
    complete tag once more bytes arrive. Full-tag-at-end is not held —
    that case is the caller's responsibility (handled by its own
    ``find()`` before calling here).

    Two-step search per tag:

    1. **Anchor fast path.** The ``anchor`` is the longest single-token
       prefix of each tag (e.g. ``<|plamo:end_``). ``rfind(anchor)`` on a
       small window at the buffer tail locates candidate match positions
       in O(anchor_len). Most calls (no partial tag at end) exit here
       immediately because ``rfind`` returns -1.

    2. **Sub-anchor fallback.** Text-based stop strings (e.g.
       ``request.stop=["<|plamo:tag|>"]``) cause vllm's detokenizer to
       flush delta fragments shorter than a single token to evaluate
       stop-string matches greedily, so the buffer tail can be an
       arbitrary prefix of the anchor (e.g. just ``<|pla``). The anchor
       fast path returns -1 for these. Walk ``k`` from ``anchor_len - 1``
       down to ``1`` and hold back the longest matching prefix found,
       using ``buf.endswith(tag[:k])`` (C-implemented in CPython) to keep
       the per-comparison cost low.

    Without the fallback, sub-anchor fragments leaked into the emitted
    delta, advanced ``emit_pos`` past the real start of the tag, and the
    streaming reasoning parser would never detect END_THINK afterwards.

    Args:
        buf: the current stream buffer.
        floor: minimum value to return (already-emitted position).
        tags: list of (full_tag, token_anchor) pairs. ``anchor`` should
            be the longest single-token prefix of ``full_tag``.
    """
    buf_len = len(buf)
    max_hold = 0
    for tag, anchor in tags:
        # The two-step search assumes ``anchor`` is a prefix of ``tag``
        # (and therefore shorter than it). Enforce the prefix relation
        # so the sub-anchor fallback's ``tag[:k]`` slicing for
        # ``k < anchor_len`` always produces a real anchor-prefix —
        # otherwise its hold could exceed ``len(tag)`` for short tags.
        assert len(anchor) <= len(tag) and tag.startswith(anchor), (
            f"anchor {anchor!r} must be a prefix of tag {tag!r}"
        )
        anchor_len = len(anchor)
        check_len = min(len(tag) - 1, buf_len)
        # Step 1: anchor-based search for prefixes ≥ anchor_len.
        if check_len >= anchor_len:
            search_end = buf_len
            search_start = buf_len - check_len
            while True:
                p = buf.rfind(anchor, search_start, search_end)
                if p == -1:
                    break
                k = buf_len - p
                if buf[p:] == tag[:k]:
                    if k > max_hold:
                        max_hold = k
                    break
                search_end = p
        # Step 2: sub-anchor prefixes (length 1..anchor_len-1). Skip if
        # step 1 already produced a hold ≥ anchor_len since any shorter
        # match cannot improve on it.
        if max_hold < anchor_len:
            max_short = min(anchor_len - 1, buf_len)
            for k in range(max_short, 0, -1):
                if buf.endswith(tag[:k]):
                    if k > max_hold:
                        max_hold = k
                    break
    safe_until = buf_len - max_hold
    if safe_until < floor:
        safe_until = floor
    assert safe_until <= buf_len, f"floor={floor} exceeds buffer length={buf_len}"
    return safe_until


class ReasoningParserStreamPhase(Enum):
    BEFORE_REASONING = "before_reasoning"
    IN_REASONING = "in_reasoning"
    AFTER_REASONING = "after_reasoning"
    DONE = "done"


class Plamo3ReasoningParser(ReasoningParser):
    @property
    def reasoning_start_str(self) -> str:
        return BEGIN_THINK_TAG

    @property
    def reasoning_end_str(self) -> str:
        return END_THINK_TAG

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # When thinking is disabled, delegate to IdentityReasoningParser which
        # treats the entire model output as content (no reasoning separation).
        chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}

        self._identity_parser: IdentityReasoningParser | None
        if not chat_template_kwargs.get("enable_thinking", True):
            self._identity_parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._identity_parser = None

        self._begin_think_token_ids: list[int] = list(
            tokenizer.encode(BEGIN_THINK_TAG, add_special_tokens=False)
        )
        self._end_think_token_ids: list[int] = list(
            tokenizer.encode(END_THINK_TAG, add_special_tokens=False)
        )
        if not self._begin_think_token_ids or not self._end_think_token_ids:
            raise ValueError(
                "PLaMo3 reasoning parser failed to tokenize think tags: "
                f"{BEGIN_THINK_TAG!r} -> {self._begin_think_token_ids}, "
                f"{END_THINK_TAG!r} -> {self._end_think_token_ids}."
            )
        # Streaming state
        self._stream_phase: ReasoningParserStreamPhase = (
            ReasoningParserStreamPhase.BEFORE_REASONING
        )
        self._stream_emit_pos: int = 0
        self._identity_stream_terminated: bool = False
        # Some vLLM call paths pass only delta token IDs, which is too short to
        # detect the multi-token END_THINK sequence. Keep the full stream token
        # IDs updated in extract_reasoning_streaming and fall back to them
        # inside is_reasoning_end / extract_content_ids.
        self._stream_token_ids: list[int] = []

    @staticmethod
    def _tokens_match_at(
        input_ids: Sequence[int], token_ids: Sequence[int], offset: int
    ) -> bool:
        if offset < 0 or offset + len(token_ids) > len(input_ids):
            return False
        for i, token_id in enumerate(token_ids):
            if input_ids[offset + i] != token_id:
                return False
        return True

    def _find_end_think(self, input_ids: Sequence[int], start: int = 0) -> int:
        n = len(self._end_think_token_ids)
        for i in range(start, len(input_ids) - n + 1):
            if self._tokens_match_at(input_ids, self._end_think_token_ids, i):
                return i
        return -1

    def _rfind_begin_think(self, input_ids: Sequence[int]) -> int:
        n = len(self._begin_think_token_ids)
        for i in range(len(input_ids) - n, -1, -1):
            if self._tokens_match_at(input_ids, self._begin_think_token_ids, i):
                return i
        return -1

    def _rfind_end_think(self, input_ids: Sequence[int]) -> int:
        n = len(self._end_think_token_ids)
        for i in range(len(input_ids) - n, -1, -1):
            if self._tokens_match_at(input_ids, self._end_think_token_ids, i):
                return i
        return -1

    def _effective_input_ids(self, input_ids: Sequence[int]) -> list[int]:
        return self._stream_token_ids or list(input_ids)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end(input_ids)
        if not (ids := self._effective_input_ids(input_ids)):
            return False
        if (last_end := self._rfind_end_think(ids)) == -1:
            return False
        if (last_begin := self._rfind_begin_think(ids)) == -1:
            return True
        return last_end > last_begin

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end_streaming(
                input_ids, delta_ids
            )
        # Whether the multi-token END_THINK completes within this decode step.
        # Scan only the window that can end inside delta (delta plus the (n-1)
        # tokens before it), mirroring gptoss's windowed streaming check: an
        # END_THINK already confirmed further in the past is not re-detected.
        # delta_ids is Iterable (islice in vLLM v0.20.2+); materialise for len().
        delta_ids = list(delta_ids)
        n = len(self._end_think_token_ids)
        delta_start = len(input_ids) - len(delta_ids)
        window_start = max(0, delta_start - (n - 1))
        for i in range(window_start, len(input_ids) - n + 1):
            if self._tokens_match_at(input_ids, self._end_think_token_ids, i):
                return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._identity_parser is not None:
            return self._identity_parser.extract_content_ids(input_ids)

        ids = self._effective_input_ids(input_ids)
        end_start = self._find_end_think(ids)
        if end_start == -1:
            return []

        return ids[end_start + len(self._end_think_token_ids) :]

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """

        if self._identity_parser is not None:
            if self._identity_stream_terminated:
                return None
            if EOT_TAG in delta_text:
                self._identity_stream_terminated = True
                delta_text = strip_at_eot(delta_text)
            return self._identity_parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        # Keep accumulated token IDs available because some vLLM call paths
        # pass only delta token IDs to is_reasoning_end / extract_content_ids.
        self._stream_token_ids = list(current_token_ids)

        # FSM: incrementally emit reasoning (and then content) as DeltaMessage
        while True:
            if self._stream_phase == ReasoningParserStreamPhase.BEFORE_REASONING:
                # Proceed only when the begin-think tag appears exactly at the head.
                if not current_text:
                    # Nothing decoded yet; wait for real text before deciding
                    # whether this turn is reasoning.
                    break
                if current_text.startswith(BEGIN_THINK_TAG):
                    self._stream_emit_pos = len(BEGIN_THINK_TAG)
                    self._stream_phase = ReasoningParserStreamPhase.IN_REASONING
                    continue
                # If only a prefix fragment of BEGIN_THINK_TAG is at the head,
                # wait until the tag completes
                if current_text and BEGIN_THINK_TAG.startswith(current_text):
                    logger.debug(
                        "pre_reasoning: begin tag prefix detected,"
                        " waiting for completion"
                    )
                    break
                # Thinking is enabled (no identity parser): treat all output
                # as reasoning until END_THINK_TAG appears. This covers chat
                # templates that inject the begin-think tag into the prompt.
                self._stream_phase = ReasoningParserStreamPhase.IN_REASONING
                continue

            if self._stream_phase == ReasoningParserStreamPhase.IN_REASONING:
                search_offset = self._stream_emit_pos
                text_end = len(current_text)
                end_tag_start = current_text.find(END_THINK_TAG, search_offset)
                end_tag_start = text_end if end_tag_start == -1 else end_tag_start
                eot_start = current_text.find(EOT_TAG, search_offset)
                eot_start = text_end if eot_start == -1 else eot_start
                # When EOT appears before END_THINK, emit reasoning up to EOT and
                # transition to DONE. The content after EOT is ignored.
                if eot_start < end_tag_start:
                    self._stream_emit_pos = eot_start + len(EOT_TAG)
                    self._stream_phase = ReasoningParserStreamPhase.DONE
                    if reasoning_delta := current_text[search_offset:eot_start]:
                        return DeltaMessage(reasoning=reasoning_delta)
                    break
                # When both EOT and END_THINK are absent, emit reasoning up to the safe point
                # and hold back the tail in case it is a partial END_THINK tag.
                if end_tag_start == text_end:
                    safe_until = compute_safe_until(
                        current_text,
                        search_offset,
                        [(END_THINK_TAG, "<|plamo:end_")],
                    )
                    if safe_until > search_offset:
                        self._stream_emit_pos = safe_until
                        return DeltaMessage(
                            reasoning=current_text[search_offset:safe_until]
                        )
                    break
                # When END_THINK appears before EOT, emit reasoning up to the tag and content after the tag.
                if eot_start != text_end:
                    self._stream_emit_pos = eot_start + len(EOT_TAG)
                    self._stream_phase = ReasoningParserStreamPhase.DONE
                else:
                    self._stream_emit_pos = text_end
                    self._stream_phase = ReasoningParserStreamPhase.AFTER_REASONING
                end_tag_end = end_tag_start + len(END_THINK_TAG)
                reasoning_delta = current_text[search_offset:end_tag_start]
                content_delta = current_text[end_tag_end:eot_start]
                if reasoning_delta or content_delta:
                    return DeltaMessage(
                        reasoning=reasoning_delta or None,
                        content=content_delta or None,
                    )
                break

            if self._stream_phase == ReasoningParserStreamPhase.AFTER_REASONING:
                eot_start = current_text.find(EOT_TAG, self._stream_emit_pos)
                if eot_start != -1:
                    delta = current_text[self._stream_emit_pos : eot_start]
                    self._stream_emit_pos = eot_start + len(EOT_TAG)
                    self._stream_phase = ReasoningParserStreamPhase.DONE
                else:
                    delta = current_text[self._stream_emit_pos :]
                    self._stream_emit_pos = len(current_text)
                if delta:
                    return DeltaMessage(content=delta)
                break

            if self._stream_phase == ReasoningParserStreamPhase.DONE:
                break

        return None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self._identity_parser is not None:
            reasoning, content = self._identity_parser.extract_reasoning(
                model_output, request
            )
            return reasoning, strip_at_eot(content) if content is not None else None

        model_output = strip_at_eot(model_output)
        begin_tag_end = (
            len(BEGIN_THINK_TAG) if model_output.startswith(BEGIN_THINK_TAG) else 0
        )
        end_tag_start = model_output.find(END_THINK_TAG, begin_tag_end)
        if end_tag_start == -1:
            reasoning = strip_trailing_partial_marker(model_output[begin_tag_end:])
            return reasoning or None, None

        end_tag_end = end_tag_start + len(END_THINK_TAG)
        reasoning = model_output[begin_tag_end:end_tag_start]
        content = strip_trailing_partial_marker(model_output[end_tag_end:])
        return reasoning, content
