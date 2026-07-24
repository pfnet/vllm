# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from enum import Enum

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser

from vllm.utils.plamo3_parser_common import (
    BEGIN_TOOL_ARGS_TAG,
    BEGIN_TOOL_NAME_TAG,
    BEGIN_TOOL_REQUEST_TAG,
    BEGIN_TOOL_REQUESTS_TAG,
    END_TOOL_ARGS_TAG,
    END_TOOL_NAME_TAG,
    END_TOOL_REQUEST_TAG,
    END_TOOL_REQUESTS_TAG,
    EOT_TAG,
    compute_safe_until,
    strip_trailing_partial_marker,
    verify_grammar_token_ids,
)

logger = init_logger(__name__)


def parse_model_output(model_output: str) -> tuple[str, list[ToolCall]]:
    """Parse a complete model output into content and tool calls.

    Returns ``(content, tool_calls)`` where ``content`` is the text before the
    first tool-requests tag.
    """
    if (pos_begin_requests := model_output.find(BEGIN_TOOL_REQUESTS_TAG)) == -1:
        return model_output, []

    content = model_output[:pos_begin_requests]
    index = pos_begin_requests + len(BEGIN_TOOL_REQUESTS_TAG)
    tool_calls: list[ToolCall] = []

    while True:
        if not model_output.startswith(BEGIN_TOOL_REQUEST_TAG, index):
            if not tool_calls:
                return content, []
            break

        index += len(BEGIN_TOOL_REQUEST_TAG)

        if not model_output.startswith(BEGIN_TOOL_NAME_TAG, index):
            return content, []
        name_start = index + len(BEGIN_TOOL_NAME_TAG)
        if (name_end := model_output.find(END_TOOL_NAME_TAG, name_start)) == -1:
            return content, []
        tool_name = model_output[name_start:name_end]
        index = name_end + len(END_TOOL_NAME_TAG)

        if not model_output.startswith(BEGIN_TOOL_ARGS_TAG, index):
            return content, []
        args_start = index + len(BEGIN_TOOL_ARGS_TAG)
        if (args_end := model_output.find(END_TOOL_ARGS_TAG, args_start)) == -1:
            return content, []
        tool_arguments = model_output[args_start:args_end]
        index = args_end + len(END_TOOL_ARGS_TAG)

        if not model_output.startswith(END_TOOL_REQUEST_TAG, index):
            return content, []
        index += len(END_TOOL_REQUEST_TAG)

        tool_calls.append(
            ToolCall(function=FunctionCall(name=tool_name, arguments=tool_arguments))
        )

    if not model_output.startswith(END_TOOL_REQUESTS_TAG, index):
        return content, []

    return content, tool_calls


class ToolParserStreamPhase(Enum):
    CONTENT = "content"
    BEGIN_TOOL_REQUEST = "begin_tool_request"
    BEGIN_TOOL_NAME = "begin_tool_name"
    BEGIN_TOOL_ARGUMENTS = "begin_tool_arguments"
    IN_TOOL_ARGUMENTS = "in_tool_arguments"
    END_TOOL_REQUEST = "end_tool_request"
    MAYBE_NEXT_TOOL_OR_END = "maybe_next_tool_or_end"
    AFTER_TOOL_REQUESTS = "after_tool_requests"
    DONE = "done"


class Plamo3ToolParser(ToolParser):
    """Tool parser for PLaMo-3 explicit tool-call tags.

    Parses structures such as:

        <|plamo:begin_tool_requests:plamo|>...<|plamo:end_tool_requests:plamo|>

    The tags are not added special tokens, so they may be split across
    streaming chunks; the streaming parser works on the reconstructed text.
    """

    # Tool tag strings and the corresponding token-id sequences used for
    # streaming detection. Order matches the state-machine flow.
    _TAG_SPEC: list[tuple[str, str]] = [
        ("BEGIN_TOOL_REQUESTS", BEGIN_TOOL_REQUESTS_TAG),
        ("END_TOOL_REQUESTS", END_TOOL_REQUESTS_TAG),
        ("BEGIN_TOOL_REQUEST", BEGIN_TOOL_REQUEST_TAG),
        ("END_TOOL_REQUEST", END_TOOL_REQUEST_TAG),
        ("BEGIN_TOOL_NAME", BEGIN_TOOL_NAME_TAG),
        ("END_TOOL_NAME", END_TOOL_NAME_TAG),
        ("BEGIN_TOOL_ARGS", BEGIN_TOOL_ARGS_TAG),
        ("END_TOOL_ARGS", END_TOOL_ARGS_TAG),
        ("EOT", EOT_TAG),
    ]

    def __init__(self, tokenizer: TokenizerLike, tools=None):
        super().__init__(tokenizer, tools)

        # Verify the grammar anchors are at the expected vocabulary positions.
        verify_grammar_token_ids(tokenizer)

        # Cache token-id sequences for each special tag so streaming detection
        # can reason from token ids, independent of text reconstruction.
        self._tag_ids: dict[str, list[int]] = {}
        for tag_name, tag_str in self._TAG_SPEC:
            ids = list(tokenizer.encode(tag_str, add_special_tokens=False))
            if not ids:
                raise ValueError(
                    f"PLaMo3 tool parser failed to tokenize {tag_name}: "
                    "the tokenizer or vocabulary may be incompatible."
                )
            self._tag_ids[tag_name] = ids

        # Convenience accessors for tag ids derived from _TAG_SPEC.
        self._begin_tool_requests_ids = self._tag_ids["BEGIN_TOOL_REQUESTS"]
        self._end_tool_requests_ids = self._tag_ids["END_TOOL_REQUESTS"]
        self._begin_tool_request_ids = self._tag_ids["BEGIN_TOOL_REQUEST"]
        self._end_tool_request_ids = self._tag_ids["END_TOOL_REQUEST"]
        self._begin_tool_name_ids = self._tag_ids["BEGIN_TOOL_NAME"]
        self._end_tool_name_ids = self._tag_ids["END_TOOL_NAME"]
        self._begin_tool_args_ids = self._tag_ids["BEGIN_TOOL_ARGS"]
        self._end_tool_args_ids = self._tag_ids["END_TOOL_ARGS"]
        self._eot_ids = self._tag_ids["EOT"]

        # Streaming state: current parse/text position, corresponding token
        # position, accumulated token ids, emitted-content boundary, and the
        # tool call currently being assembled.
        self._stream_pos: int = 0
        self._stream_token_pos: int = 0
        self._stream_token_ids: list[int] = []
        self._stream_phase: ToolParserStreamPhase = ToolParserStreamPhase.CONTENT
        self._stream_content_emit_pos: int = 0
        self._stream_tool_index: int = 0
        self._stream_tool_id: str | None = None
        self._stream_function_name: str | None = None
        self._stream_arg_emit_pos: int | None = None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        content, tool_calls = parse_model_output(model_output)
        content = strip_trailing_partial_marker(content)
        return ExtractedToolCallInformation(
            tools_called=(len(tool_calls) != 0),
            tool_calls=tool_calls,
            content=content,
        )

    @staticmethod
    def _tokens_match_at(
        input_ids: Sequence[int], token_ids: Sequence[int], offset: int
    ) -> bool:
        """Return True when ``token_ids`` matches ``input_ids`` at ``offset``."""
        if offset < 0 or offset + len(token_ids) > len(input_ids):
            return False
        for i, token_id in enumerate(token_ids):
            if input_ids[offset + i] != token_id:
                return False
        return True

    @staticmethod
    def _skip_leading_bos(
        input_ids: Sequence[int], bos_id: int | None
    ) -> Sequence[int]:
        """Drop a leading BOS token id if present."""
        if bos_id is not None and len(input_ids) > 0 and input_ids[0] == bos_id:
            return input_ids[1:]
        return input_ids

    def _find_tag(
        self, input_ids: Sequence[int], tag_ids: list[int], start: int = 0
    ) -> int:
        """First token index at which ``tag_ids`` occurs at or after ``start``."""
        n = len(tag_ids)
        for i in range(start, len(input_ids) - n + 1):
            if self._tokens_match_at(input_ids, tag_ids, i):
                return i
        return -1

    def _tag_in_delta_window(
        self,
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        tag_ids: list[int],
    ) -> int:
        """Return the token index of a tag occurrence completable in delta.

        vLLM passes ``current_token_ids`` = all_token_ids (including delta as
        suffix) and ``delta_token_ids`` = the new tokens. We only need to check
        windows that can complete inside the delta: the last ``n-1`` tokens
        before the delta plus the delta itself.
        """
        delta_token_ids = list(delta_token_ids)
        n = len(tag_ids)
        delta_start = len(current_token_ids) - len(delta_token_ids)
        window_start = max(0, delta_start - (n - 1))
        for i in range(window_start, len(current_token_ids) - n + 1):
            if self._tokens_match_at(current_token_ids, tag_ids, i):
                return i
        return -1

    def _advance_tag(
        self,
        tag_str: str,
        tag_ids: list[int],
    ) -> None:
        """Advance both the text and token positions past a detected tag."""
        self._stream_pos += len(tag_str)
        self._stream_token_pos += len(tag_ids)

    def _tag_text_present_at(self, current_text: str, tag_str: str) -> bool:
        """Return True when ``tag_str`` is fully present at ``_stream_pos``."""
        return current_text.find(tag_str, self._stream_pos) == self._stream_pos

    def _emit_content_delta(
        self,
        buf: str,
        until: int,
    ) -> DeltaMessage | None:
        """Emit a content delta up to ``until`` if new bytes exist."""
        if until > self._stream_content_emit_pos and (
            delta := buf[self._stream_content_emit_pos : until]
        ):
            self._stream_content_emit_pos = until
            return DeltaMessage(content=delta)
        return None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract content/tool-call deltas from an incomplete streaming response.

        Tracks a small state machine over ``current_text`` so that leading
        content, tool names, and arguments are emitted separately without
        leaking the special tags.
        """
        # Keep accumulated token ids in sync so ID-based detection can work
        # around fragmented multi-token tags across streaming deltas.
        self._stream_token_ids = list(current_token_ids)

        while True:
            if self._stream_phase == ToolParserStreamPhase.CONTENT:
                next_begin = current_text.find(
                    BEGIN_TOOL_REQUESTS_TAG, self._stream_pos
                )
                next_eos = current_text.find(EOT_TAG, self._stream_pos)
                candidates = [p for p in [next_begin, next_eos] if p != -1]
                if candidates:
                    next_pos = min(candidates)
                    if msg := self._emit_content_delta(current_text, next_pos):
                        return msg
                else:
                    # EOT_TAG ("<|plamo:tag|>") is a single token and therefore
                    # never appears partially in the buffer, so it needs no
                    # hold-back and is omitted here.
                    safe_until = compute_safe_until(
                        current_text,
                        self._stream_content_emit_pos,
                        [(BEGIN_TOOL_REQUESTS_TAG, "<|plamo:begin_")],
                    )
                    if msg := self._emit_content_delta(current_text, safe_until):
                        return msg
                    break

                # Sync the structural parse position to the tag before
                # advancing. ``_stream_pos`` is not moved while content is
                # emitted (only ``_stream_content_emit_pos`` is), so when
                # content precedes the tag it still points at the content
                # start. ``_advance_tag`` then adds the tag length to the wrong
                # base and lands inside the content, so the following
                # BEGIN_TOOL_REQUEST is never detected and the tool call is
                # dropped. ``next_pos`` is the position of whichever tag the
                # branch below advances past.
                self._stream_pos = next_pos
                if next_eos != -1 and (next_begin == -1 or next_eos < next_begin):
                    self._advance_tag(EOT_TAG, self._eot_ids)
                    self._stream_phase = ToolParserStreamPhase.DONE
                else:
                    self._advance_tag(
                        BEGIN_TOOL_REQUESTS_TAG, self._begin_tool_requests_ids
                    )
                    self._stream_phase = ToolParserStreamPhase.BEGIN_TOOL_REQUEST
                continue

            if self._stream_phase == ToolParserStreamPhase.BEGIN_TOOL_REQUEST:
                if self._tag_text_present_at(current_text, BEGIN_TOOL_REQUEST_TAG):
                    self._advance_tag(
                        BEGIN_TOOL_REQUEST_TAG, self._begin_tool_request_ids
                    )
                    self._stream_tool_id = make_tool_call_id()
                    self._stream_function_name = None
                    self._stream_arg_emit_pos = None
                    self._stream_phase = ToolParserStreamPhase.BEGIN_TOOL_NAME
                    continue
                # If the tag is detected by token ids but the text is not yet
                # complete in the buffer, wait for more tokens.
                if (
                    self._tag_in_delta_window(
                        current_token_ids,
                        delta_token_ids,
                        self._begin_tool_request_ids,
                    )
                    != -1
                ):
                    break
                break

            if self._stream_phase == ToolParserStreamPhase.BEGIN_TOOL_NAME:
                if self._tag_text_present_at(current_text, BEGIN_TOOL_NAME_TAG):
                    name_start = self._stream_pos + len(BEGIN_TOOL_NAME_TAG)
                    if (
                        name_end := current_text.find(END_TOOL_NAME_TAG, name_start)
                    ) == -1:
                        break
                    self._stream_function_name = current_text[name_start:name_end]
                    self._stream_pos = name_end + len(END_TOOL_NAME_TAG)
                    self._stream_token_pos += len(self._begin_tool_name_ids)
                    # Skip over the tool name tokens; we do not track the
                    # exact token length of the variable tool name, but the
                    # next fixed tag detection resynchronises token_pos.
                    end_name_idx = self._find_tag(
                        current_token_ids,
                        self._end_tool_name_ids,
                        self._stream_token_pos,
                    )
                    if end_name_idx != -1:
                        self._stream_token_pos = end_name_idx + len(
                            self._end_tool_name_ids
                        )
                    self._stream_phase = ToolParserStreamPhase.BEGIN_TOOL_ARGUMENTS
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self._stream_tool_index,
                                type="function",
                                id=self._stream_tool_id,
                                function=DeltaFunctionCall(
                                    name=self._stream_function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                if (
                    self._tag_in_delta_window(
                        current_token_ids,
                        delta_token_ids,
                        self._begin_tool_name_ids,
                    )
                    != -1
                ):
                    break
                break

            if self._stream_phase == ToolParserStreamPhase.BEGIN_TOOL_ARGUMENTS:
                if self._tag_text_present_at(current_text, BEGIN_TOOL_ARGS_TAG):
                    self._advance_tag(BEGIN_TOOL_ARGS_TAG, self._begin_tool_args_ids)
                    self._stream_arg_emit_pos = self._stream_pos
                    self._stream_phase = ToolParserStreamPhase.IN_TOOL_ARGUMENTS
                    continue
                if (
                    self._tag_in_delta_window(
                        current_token_ids,
                        delta_token_ids,
                        self._begin_tool_args_ids,
                    )
                    != -1
                ):
                    break
                break

            if self._stream_phase == ToolParserStreamPhase.IN_TOOL_ARGUMENTS:
                if (
                    args_end_start := current_text.find(
                        END_TOOL_ARGS_TAG, self._stream_pos
                    )
                ) != -1:
                    if (
                        self._stream_arg_emit_pos is not None
                        and args_end_start > self._stream_arg_emit_pos
                        and (
                            delta := current_text[
                                self._stream_arg_emit_pos : args_end_start
                            ]
                        )
                    ):
                        self._stream_arg_emit_pos = args_end_start
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self._stream_tool_index,
                                    type="function",
                                    id=None,
                                    function=DeltaFunctionCall(
                                        arguments=delta
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                    self._advance_tag(END_TOOL_ARGS_TAG, self._end_tool_args_ids)
                    self._stream_phase = ToolParserStreamPhase.END_TOOL_REQUEST
                    continue

                # If END_TOOL_ARGS is detected by token ids but the text is
                # not complete yet, wait to avoid leaking tag text.
                if (
                    self._tag_in_delta_window(
                        current_token_ids,
                        delta_token_ids,
                        self._end_tool_args_ids,
                    )
                    != -1
                ):
                    break

                if (
                    self._stream_arg_emit_pos is not None
                    and (
                        safe_until := compute_safe_until(
                            current_text,
                            self._stream_arg_emit_pos,
                            [(END_TOOL_ARGS_TAG, "<|plamo:end_")],
                        )
                    )
                    > self._stream_arg_emit_pos
                    and (delta := current_text[self._stream_arg_emit_pos : safe_until])
                ):
                    self._stream_arg_emit_pos = safe_until
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self._stream_tool_index,
                                type="function",
                                id=None,
                                function=DeltaFunctionCall(arguments=delta).model_dump(
                                    exclude_none=True
                                ),
                            )
                        ]
                    )
                break

            if self._stream_phase == ToolParserStreamPhase.END_TOOL_REQUEST:
                if self._tag_text_present_at(current_text, END_TOOL_REQUEST_TAG):
                    self._advance_tag(
                        END_TOOL_REQUEST_TAG, self._end_tool_request_ids
                    )
                    self._stream_phase = ToolParserStreamPhase.MAYBE_NEXT_TOOL_OR_END
                    continue
                if (
                    self._tag_in_delta_window(
                        current_token_ids,
                        delta_token_ids,
                        self._end_tool_request_ids,
                    )
                    != -1
                ):
                    break
                break

            if self._stream_phase == ToolParserStreamPhase.MAYBE_NEXT_TOOL_OR_END:
                if self._tag_text_present_at(current_text, BEGIN_TOOL_REQUEST_TAG):
                    self._stream_tool_index += 1
                    self._stream_tool_id = make_tool_call_id()
                    self._stream_function_name = None
                    self._stream_arg_emit_pos = None
                    self._advance_tag(
                        BEGIN_TOOL_REQUEST_TAG, self._begin_tool_request_ids
                    )
                    self._stream_phase = ToolParserStreamPhase.BEGIN_TOOL_NAME
                    continue
                if (
                    end_pos := current_text.find(
                        END_TOOL_REQUESTS_TAG, self._stream_pos
                    )
                ) != -1:
                    self._stream_pos = end_pos + len(END_TOOL_REQUESTS_TAG)
                    self._stream_token_pos += len(self._end_tool_requests_ids)
                    self._stream_phase = ToolParserStreamPhase.AFTER_TOOL_REQUESTS
                    continue
                if (
                    self._tag_in_delta_window(
                        current_token_ids,
                        delta_token_ids,
                        self._end_tool_requests_ids,
                    )
                    != -1
                ):
                    break
                break

            if self._stream_phase == ToolParserStreamPhase.AFTER_TOOL_REQUESTS:
                if self._tag_text_present_at(current_text, EOT_TAG):
                    self._advance_tag(EOT_TAG, self._eot_ids)
                    self._stream_phase = ToolParserStreamPhase.DONE
                    continue
                break

            if self._stream_phase == ToolParserStreamPhase.DONE:
                break

        return None
