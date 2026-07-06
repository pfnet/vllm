# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike

# PLaMo-3 wraps reasoning with explicit tags that are *not* registered as
# added special tokens, so they tokenize into several normal tokens and remain
# visible even when ``skip_special_tokens=True``.  The tag strings below are the
# PLaMo-3 reasoning delimiters; the corresponding token IDs are cached in
# ``__init__`` from the provided tokenizer.
BEGIN_THINK_TAG = "<|plamo:begin_think:plamo|>"
END_THINK_TAG = "<|plamo:end_think:plamo|>"

_SPECIAL_TOKEN_PREFIX = "<|plamo:"


def strip_trailing_partial_marker(text: str) -> str:
    """Strip a trailing incomplete reasoning-tag fragment."""
    if (idx := text.rfind(_SPECIAL_TOKEN_PREFIX)) == -1:
        return text
    tail = text[idx:]
    for tag in (BEGIN_THINK_TAG, END_THINK_TAG):
        if tag.startswith(tail) and tail != tag:
            return text[:idx]
    return text


def compute_safe_until(
    buf: str,
    floor: int,
    tags: list[tuple[str, str]],
) -> int:
    """Largest buffer position that can be flushed without splitting a tag.

    Holds back any strict tail of ``buf`` that is a prefix of one of ``tags``.
    ``anchor`` must be the longest single-token prefix of the corresponding
    ``tag``.
    """
    buf_len = len(buf)
    max_hold = 0
    for tag, anchor in tags:
        assert len(anchor) <= len(tag) and tag.startswith(anchor), (
            f"anchor {anchor!r} must be a prefix of tag {tag!r}"
        )
        anchor_len = len(anchor)
        check_len = min(len(tag) - 1, buf_len)
        if check_len >= anchor_len:
            search_end = buf_len
            search_start = buf_len - check_len
            while True:
                if (p := buf.rfind(anchor, search_start, search_end)) == -1:
                    break
                k = buf_len - p
                if buf[p:] == tag[:k]:
                    if k > max_hold:
                        max_hold = k
                    break
                search_end = p
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


if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any


class ReasoningParserStreamPhase(Enum):
    BEFORE_REASONING = "before_reasoning"
    IN_REASONING = "in_reasoning"
    AFTER_REASONING = "after_reasoning"


class Plamo3ReasoningParser(ReasoningParser):
    """Reasoning parser for PLaMo-3 explicit thinking blocks.

    PLaMo-3 wraps chain-of-thought reasoning with:

        <|plamo:begin_think:plamo|>...<|plamo:end_think:plamo|>

    The tags are not added special tokens, so the streaming detokenizer can
    split them across chunks.  This parser reconstructs the full text from the
    token-id sequence and tracks a small state machine so reasoning and final
    content are emitted separately without leaking the special tags.
    """

    @property
    def start_token(self) -> str:
        return BEGIN_THINK_TAG

    @property
    def end_token(self) -> str:
        return END_THINK_TAG

    @property
    def reasoning_start_str(self) -> str:
        return self.start_token

    @property
    def reasoning_end_str(self) -> str:
        return self.end_token

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # PLaMo-3 think tags tokenize to fixed multi-token sequences; cache the
        # ids from the provided tokenizer so the parser adapts to the model.
        self._begin_think_token_ids: list[int] = list(
            tokenizer.encode(BEGIN_THINK_TAG, add_special_tokens=False)
        )
        self._end_think_token_ids: list[int] = list(
            tokenizer.encode(END_THINK_TAG, add_special_tokens=False)
        )

        # Streaming state: which phase we are in and how much of ``reconstructed``
        # has already been emitted to the client.
        self._stream_phase: ReasoningParserStreamPhase = (
            ReasoningParserStreamPhase.BEFORE_REASONING
        )
        self._stream_emit_pos: int = 0

    def _starts_with_begin_think(self, input_ids: Sequence[int]) -> bool:
        """Check whether ``input_ids`` matches the begin-think token prefix."""
        if not input_ids:
            return False
        prefix_len = min(len(input_ids), len(self._begin_think_token_ids))
        return list(input_ids[:prefix_len]) == self._begin_think_token_ids[:prefix_len]

    def _find_end_think(self, input_ids: Sequence[int], start: int = 0) -> int:
        """Return the index of the end-think token sequence, or -1 if absent."""
        n = len(self._end_think_token_ids)
        for i in range(start, len(input_ids) - n + 1):
            if list(input_ids[i : i + n]) == self._end_think_token_ids:
                return i
        return -1

    def _contains_end_think(self, input_ids: Sequence[int]) -> bool:
        """Check whether the complete end-think token sequence appears."""
        return self._find_end_think(input_ids) != -1

    def _end_think_tail_prefix_len(self, input_ids: Sequence[int]) -> int:
        """Length of the longest trailing run that is a proper end-think prefix.

        Used while inside a reasoning block to hold back tokens that could still
        grow into the complete end-think sequence on the next step.
        """
        max_k = min(len(self._end_think_token_ids) - 1, len(input_ids))
        for k in range(max_k, 0, -1):
            if list(input_ids[-k:]) == self._end_think_token_ids[:k]:
                return k
        return 0

    def _reconstruct(self, ids: list[int], current_text: str) -> str:
        """Rebuild the stream buffer from token ids, falling back to text."""
        if ids:
            return self.model_tokenizer.decode(ids, skip_special_tokens=False)
        return strip_trailing_partial_marker(current_text)

    def _find_reasoning_end(
        self,
        ids: list[int],
        reconstructed: str,
    ) -> tuple[int | None, int | None]:
        """Locate the end-think boundary and return (char_start, tag_end)."""
        if ids:
            end_idx = self._find_end_think(ids, len(self._begin_think_token_ids))
            if end_idx == -1:
                return None, None
            end_char_start = len(
                self.model_tokenizer.decode(ids[:end_idx], skip_special_tokens=False)
            )
            return end_char_start, end_char_start + len(END_THINK_TAG)

        if (
            end_tag_start := reconstructed.find(END_THINK_TAG, self._stream_emit_pos)
        ) == -1:
            return None, None
        return end_tag_start, end_tag_start + len(END_THINK_TAG)

    def _emit_reasoning(
        self,
        reconstructed: str,
        end_pos: int,
    ) -> DeltaMessage | None:
        """Emit a reasoning delta up to ``end_pos`` if new bytes exist."""
        if end_pos > self._stream_emit_pos and (
            delta := reconstructed[self._stream_emit_pos : end_pos]
        ):
            self._stream_emit_pos = end_pos
            return DeltaMessage(reasoning=delta)
        return None

    def _emit_content(self, reconstructed: str) -> DeltaMessage | None:
        """Emit any remaining content delta."""
        if delta := reconstructed[self._stream_emit_pos :]:
            self._stream_emit_pos = len(reconstructed)
            return DeltaMessage(content=delta)
        return None

    def _has_begin_think(self, ids: list[int], reconstructed: str) -> bool:
        """Whether the stream opens with the complete begin-think sequence."""
        if ids:
            return self._starts_with_begin_think(ids)
        return reconstructed.startswith(BEGIN_THINK_TAG)

    def _is_begin_think_prefix(self, ids: list[int], reconstructed: str) -> bool:
        """Whether the stream is still an incomplete prefix of begin-think."""
        if ids:
            return self._starts_with_begin_think(ids) and len(ids) < len(
                self._begin_think_token_ids
            )
        return bool(reconstructed) and BEGIN_THINK_TAG.startswith(reconstructed)

    def _safe_reasoning_until(
        self,
        ids: list[int],
        reconstructed: str,
    ) -> int:
        """Largest buffer position that can be flushed without splitting a tag."""
        if ids:
            if hold := self._end_think_tail_prefix_len(ids):
                held = self.model_tokenizer.decode(
                    ids[len(ids) - hold :], skip_special_tokens=False
                )
                return len(reconstructed) - len(held)
            return len(reconstructed)
        return compute_safe_until(
            reconstructed,
            self._stream_emit_pos,
            [(END_THINK_TAG, "<|plamo:end_")],
        )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Return True when the reasoning phase is over."""
        return not (input_ids and not self._contains_end_think(input_ids))

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """Return True when the streaming output has left the reasoning block."""
        if not self._starts_with_begin_think(input_ids):
            # Non-reasoning output (including tool calls) never enters the
            # reasoning block, so report it as finished immediately.
            return True
        delta_ids = list(delta_ids)
        n = len(self._end_think_token_ids)
        delta_start = len(input_ids) - len(delta_ids)
        window_start = max(0, delta_start - (n - 1))
        window = input_ids[window_start:]
        for i in range(len(window) - n + 1):
            if list(window[i : i + n]) == self._end_think_token_ids:
                return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Content ids are derived from text by the orchestration layer."""
        return []

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract reasoning/content deltas from an incomplete streaming response.

        Decodes the full ``current_token_ids`` to reconstruct special tags that
        may be fragmented across chunks.  The state machine then emits only the
        reasoning text while inside the think block and only content after it.
        """
        ids = list(current_token_ids)
        reconstructed = self._reconstruct(ids, current_text)

        while True:
            if self._stream_phase == ReasoningParserStreamPhase.BEFORE_REASONING:
                if self._is_begin_think_prefix(ids, reconstructed):
                    break
                if self._has_begin_think(ids, reconstructed):
                    self._stream_emit_pos = len(BEGIN_THINK_TAG)
                    self._stream_phase = ReasoningParserStreamPhase.IN_REASONING
                    continue
                self._stream_phase = ReasoningParserStreamPhase.AFTER_REASONING
                continue

            if self._stream_phase == ReasoningParserStreamPhase.IN_REASONING:
                end_char_start, end_tag_end = self._find_reasoning_end(
                    ids, reconstructed
                )
                if end_char_start is not None:
                    if msg := self._emit_reasoning(reconstructed, end_char_start):
                        return msg
                    self._stream_emit_pos = end_tag_end
                    self._stream_phase = ReasoningParserStreamPhase.AFTER_REASONING
                    continue

                safe_until = self._safe_reasoning_until(ids, reconstructed)
                if msg := self._emit_reasoning(reconstructed, safe_until):
                    return msg
                break

            if self._stream_phase == ReasoningParserStreamPhase.AFTER_REASONING:
                if msg := self._emit_content(reconstructed):
                    return msg
                break

        return None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """Split a complete response into reasoning and content."""
        if not model_output.startswith(BEGIN_THINK_TAG):
            return None, strip_trailing_partial_marker(model_output)

        begin_tag_end = len(BEGIN_THINK_TAG)
        if (end_tag_start := model_output.find(END_THINK_TAG, begin_tag_end)) == -1:
            # Incomplete reasoning block: report the partial chain-of-thought as
            # reasoning rather than leaking it into content.
            return strip_trailing_partial_marker(model_output[begin_tag_end:]), None

        end_tag_end = end_tag_start + len(END_THINK_TAG)
        reasoning = strip_trailing_partial_marker(
            model_output[begin_tag_end:end_tag_start]
        )
        content = strip_trailing_partial_marker(model_output[end_tag_end:])
        return reasoning, content
