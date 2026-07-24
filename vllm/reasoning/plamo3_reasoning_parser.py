# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.tokenizers import TokenizerLike

from vllm.utils.plamo3_parser_common import (
    BEGIN_THINK_TAG,
    END_THINK_TAG,
    compute_safe_until,
    strip_trailing_partial_marker,
)

logger = init_logger(__name__)

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
    @property
    def reasoning_start_str(self) -> str:
        return BEGIN_THINK_TAG

    @property
    def reasoning_end_str(self) -> str:
        return END_THINK_TAG

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

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
        # vLLM 0.20.2 passes only delta_token_ids to is_reasoning_end, which is
        # too short to detect the multi-token END_THINK sequence. We keep the
        # full stream token ids updated in extract_reasoning_streaming and fall
        # back to them inside is_reasoning_end / extract_content_ids.
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

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not (ids := self._stream_token_ids or list(input_ids)):
            return False
        if (last_end := self._rfind_end_think(ids)) == -1:
            return False
        if (last_begin := self._rfind_begin_think(ids)) == -1:
            return True
        return last_end > last_begin

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
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
        # vLLM 0.20.2 passes delta_token_ids here; use the accumulated stream.
        ids = self._stream_token_ids or input_ids
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

        # Keep accumulated token ids in sync so is_reasoning_end /
        # extract_content_ids can work around vLLM 0.20.2's delta-only call.
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
                # If the head is neither the tag nor its prefix fragment,
                # treat everything as content
                self._stream_phase = ReasoningParserStreamPhase.AFTER_REASONING
                continue

            if self._stream_phase == ReasoningParserStreamPhase.IN_REASONING:
                # Search for the end tag after the last emitted position
                search_offset = self._stream_emit_pos
                end_tag_start = current_text.find(END_THINK_TAG, search_offset)
                if end_tag_start != -1:
                    end_tag_end = end_tag_start + len(END_THINK_TAG)
                    reasoning_delta = ""
                    if end_tag_start > self._stream_emit_pos:
                        reasoning_delta = current_text[
                            self._stream_emit_pos : end_tag_start
                        ]
                    content_delta = current_text[end_tag_end:]
                    self._stream_emit_pos = len(current_text)
                    self._stream_phase = ReasoningParserStreamPhase.AFTER_REASONING
                    if reasoning_delta or content_delta:
                        return DeltaMessage(
                            reasoning=reasoning_delta or None,
                            content=content_delta or None,
                        )
                    return None
                else:
                    # End tag not found yet. Emit only the safely determinable range.
                    # END_THINK_TAG starts with the single-token anchor "<|plamo:end_",
                    # so we only hold back content when the buffer tail
                    # matches a prefix of the tag starting at that anchor.
                    if self._stream_emit_pos < len(current_text):
                        safe_until = compute_safe_until(
                            current_text,
                            self._stream_emit_pos,
                            [(END_THINK_TAG, "<|plamo:end_")],
                        )
                        if safe_until > self._stream_emit_pos:
                            delta = current_text[self._stream_emit_pos : safe_until]
                            if delta:
                                self._stream_emit_pos = safe_until
                                return DeltaMessage(reasoning=delta)
                    break

            if self._stream_phase == ReasoningParserStreamPhase.AFTER_REASONING:
                delta = current_text[self._stream_emit_pos :]
                if delta:
                    self._stream_emit_pos = len(current_text)
                    return DeltaMessage(content=delta)
                break

        return None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Parameters:
        model_output: str
            The model-generated string to extract reasoning content from.

        request: ChatCompletionRequest
            The request object that was used to generate the model_output.

        Returns:
        tuple[Optional[str], Optional[str]]
            A tuple containing the reasoning content and the content.
        """
        if not model_output.startswith(BEGIN_THINK_TAG):
            # A lone partial begin-think anchor ("<|plamo:begin_") left by a
            # max_tokens cut lands here (it is not the full tag). Strip the
            # incomplete-marker artifact so it does not leak as content; real
            # content (which never contains "<|plamo:") is unaffected. Keep the
            # original empty-handling (return the string, not None) so this path
            # only loses the marker artifact, nothing else.
            return None, strip_trailing_partial_marker(model_output)
        begin_tag_end = len(BEGIN_THINK_TAG)
        end_tag_start = model_output.find(END_THINK_TAG, begin_tag_end)
        if end_tag_start == -1:
            # END_THINK not found: the turn was cut inside the reasoning span.
            # If the tail is a partial END_THINK (or other) marker from a
            # max_tokens cut, strip it so no special-token markup leaks into
            # reasoning/content either.
            return strip_trailing_partial_marker(model_output[begin_tag_end:]), None
        end_tag_end = end_tag_start + len(END_THINK_TAG)
        reasoning = model_output[begin_tag_end:end_tag_start]
        # Return content as a plain string (possibly ""), never None: this value
        # is fed to the tool parser downstream (vllm serving extracts tool calls
        # "exclusively from the content"), and vllm itself normalises the final
        # message.content to None for tool-call responses. Nulling here would be
        # redundant and risks the assert-content-is-not-None branches in vllm's
        # tool-choice handling.
        content = strip_trailing_partial_marker(model_output[end_tag_end:])
        return reasoning, content
