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
from vllm.reasoning.plamo3_reasoning_parser import compute_safe_until
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)

# PLaMo-3 tool-call tags.  These are not added special tokens, so they may be
# split across streaming chunks; the parser below works on the reconstructed
# text rather than on single token IDs.
BEGIN_TOOL_REQUESTS_TAG = "<|plamo:begin_tool_requests:plamo|>"
END_TOOL_REQUESTS_TAG = "<|plamo:end_tool_requests:plamo|>"
BEGIN_TOOL_REQUEST_TAG = "<|plamo:begin_tool_request:plamo|>"
END_TOOL_REQUEST_TAG = "<|plamo:end_tool_request:plamo|>"
BEGIN_TOOL_NAME_TAG = "<|plamo:begin_tool_name:plamo|>"
END_TOOL_NAME_TAG = "<|plamo:end_tool_name:plamo|>"
# Composite tag emitted by the model around JSON tool arguments.
BEGIN_TOOL_ARGS_TAG = (
    "<|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>"
)
END_TOOL_ARGS_TAG = "<|plamo:end_tool_arguments:plamo|>"
EOT_TAG = "<|plamo:tag|>"


_SPECIAL_TOKEN_PREFIX = "<|plamo:"

_ALL_TOOL_TAGS: tuple[str, ...] = (
    BEGIN_TOOL_REQUESTS_TAG,
    END_TOOL_REQUESTS_TAG,
    BEGIN_TOOL_REQUEST_TAG,
    END_TOOL_REQUEST_TAG,
    BEGIN_TOOL_NAME_TAG,
    END_TOOL_NAME_TAG,
    "<|plamo:begin_tool_arguments:plamo|>",
    END_TOOL_ARGS_TAG,
    EOT_TAG,
)


def strip_trailing_partial_marker(text: str) -> str:
    """Strip a trailing incomplete PLaMo-3 tool-tag fragment from content."""
    if (idx := text.rfind(_SPECIAL_TOKEN_PREFIX)) == -1:
        return text
    tail = text[idx:]
    for tag in _ALL_TOOL_TAGS:
        if tag.startswith(tail) and tail != tag:
            return text[:idx]
    return text


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

    def __init__(self, tokenizer: TokenizerLike, tools=None):
        super().__init__(tokenizer, tools)

        # Streaming state: current parse position, emitted-content boundary,
        # and the tool call currently being assembled.
        self._stream_pos: int = 0
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

                if next_eos != -1 and (next_begin == -1 or next_eos < next_begin):
                    self._stream_pos = next_eos + len(EOT_TAG)
                    self._stream_phase = ToolParserStreamPhase.DONE
                else:
                    self._stream_pos = next_begin + len(BEGIN_TOOL_REQUESTS_TAG)
                    self._stream_phase = ToolParserStreamPhase.BEGIN_TOOL_REQUEST
                continue

            if self._stream_phase == ToolParserStreamPhase.BEGIN_TOOL_REQUEST:
                if (
                    current_text.find(BEGIN_TOOL_REQUEST_TAG, self._stream_pos)
                    == self._stream_pos
                ):
                    self._stream_pos += len(BEGIN_TOOL_REQUEST_TAG)
                    self._stream_tool_id = make_tool_call_id()
                    self._stream_function_name = None
                    self._stream_arg_emit_pos = None
                    self._stream_phase = ToolParserStreamPhase.BEGIN_TOOL_NAME
                    continue
                break

            if self._stream_phase == ToolParserStreamPhase.BEGIN_TOOL_NAME:
                if (
                    current_text.find(BEGIN_TOOL_NAME_TAG, self._stream_pos)
                    == self._stream_pos
                ):
                    name_start = self._stream_pos + len(BEGIN_TOOL_NAME_TAG)
                    if (
                        name_end := current_text.find(END_TOOL_NAME_TAG, name_start)
                    ) == -1:
                        break
                    self._stream_function_name = current_text[name_start:name_end]
                    self._stream_pos = name_end + len(END_TOOL_NAME_TAG)
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
                break

            if self._stream_phase == ToolParserStreamPhase.BEGIN_TOOL_ARGUMENTS:
                if (
                    current_text.find(BEGIN_TOOL_ARGS_TAG, self._stream_pos)
                    == self._stream_pos
                ):
                    self._stream_pos += len(BEGIN_TOOL_ARGS_TAG)
                    self._stream_arg_emit_pos = self._stream_pos
                    self._stream_phase = ToolParserStreamPhase.IN_TOOL_ARGUMENTS
                    continue
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
                    self._stream_pos = args_end_start + len(END_TOOL_ARGS_TAG)
                    self._stream_phase = ToolParserStreamPhase.END_TOOL_REQUEST
                    continue

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
                if (
                    current_text.find(END_TOOL_REQUEST_TAG, self._stream_pos)
                    == self._stream_pos
                ):
                    self._stream_pos += len(END_TOOL_REQUEST_TAG)
                    self._stream_phase = ToolParserStreamPhase.MAYBE_NEXT_TOOL_OR_END
                    continue
                break

            if self._stream_phase == ToolParserStreamPhase.MAYBE_NEXT_TOOL_OR_END:
                if (
                    current_text.find(BEGIN_TOOL_REQUEST_TAG, self._stream_pos)
                    == self._stream_pos
                ):
                    self._stream_tool_index += 1
                    self._stream_tool_id = make_tool_call_id()
                    self._stream_function_name = None
                    self._stream_arg_emit_pos = None
                    self._stream_pos += len(BEGIN_TOOL_REQUEST_TAG)
                    self._stream_phase = ToolParserStreamPhase.BEGIN_TOOL_NAME
                    continue
                if (
                    end_pos := current_text.find(
                        END_TOOL_REQUESTS_TAG, self._stream_pos
                    )
                ) != -1:
                    self._stream_pos = end_pos + len(END_TOOL_REQUESTS_TAG)
                    self._stream_phase = ToolParserStreamPhase.AFTER_TOOL_REQUESTS
                    continue
                break

            if self._stream_phase == ToolParserStreamPhase.AFTER_TOOL_REQUESTS:
                if current_text.find(EOT_TAG, self._stream_pos) == self._stream_pos:
                    self._stream_pos += len(EOT_TAG)
                    self._stream_phase = ToolParserStreamPhase.DONE
                    continue
                break

            if self._stream_phase == ToolParserStreamPhase.DONE:
                break

        return None
