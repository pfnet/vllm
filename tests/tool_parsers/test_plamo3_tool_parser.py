# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from tests.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.plamo3_tool_parser import (
    BEGIN_TOOL_ARGS_TAG,
    BEGIN_TOOL_NAME_TAG,
    BEGIN_TOOL_REQUEST_TAG,
    BEGIN_TOOL_REQUESTS_TAG,
    END_TOOL_ARGS_TAG,
    END_TOOL_NAME_TAG,
    END_TOOL_REQUEST_TAG,
    END_TOOL_REQUESTS_TAG,
    EOT_TAG,
    Plamo3ToolParser,
    strip_trailing_partial_marker,
)

PLAMO3_MODEL_PATH = "/mnt/shared/models/plamo3-2604-31b-256k-id4281-non-reasoning-release-v3-gptq-4bit-qid0"  # noqa: E501


class _DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    def tokenize(self, text: str) -> list[str]:
        return [text] if text else []

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


@pytest.fixture
def dummy_tokenizer() -> _DummyTokenizer:
    return _DummyTokenizer()


@pytest.fixture(scope="module")
def plamo3_tokenizer() -> AutoTokenizer:
    if not Path(PLAMO3_MODEL_PATH).exists():
        pytest.skip(f"PLaMo-3 model not found at {PLAMO3_MODEL_PATH}")
    return AutoTokenizer.from_pretrained(PLAMO3_MODEL_PATH, trust_remote_code=True)


@pytest.fixture
def parser(dummy_tokenizer: _DummyTokenizer) -> Plamo3ToolParser:
    return ToolParserManager.get_tool_parser("plamo3")(dummy_tokenizer)


@pytest.fixture
def mock_request() -> Mock:
    req = Mock(spec=ChatCompletionRequest)
    req.chat_template_kwargs = {}
    return req


def _wrap_single_tool_call(name: str, args_json: str) -> str:
    return (
        f"{BEGIN_TOOL_REQUEST_TAG}"
        f"{BEGIN_TOOL_NAME_TAG}{name}{END_TOOL_NAME_TAG}"
        f"{BEGIN_TOOL_ARGS_TAG}{args_json}{END_TOOL_ARGS_TAG}"
        f"{END_TOOL_REQUEST_TAG}"
    )


def _wrap_tool_requests(body: str) -> str:
    return f"{BEGIN_TOOL_REQUESTS_TAG}{body}{END_TOOL_REQUESTS_TAG}{EOT_TAG}"


def _split_into_deltas(text: str) -> list[str]:
    """Split a model output into deltas aligned with PLaMo-3 tags."""
    tags = sorted(
        [
            BEGIN_TOOL_REQUESTS_TAG,
            END_TOOL_REQUESTS_TAG,
            BEGIN_TOOL_REQUEST_TAG,
            END_TOOL_REQUEST_TAG,
            BEGIN_TOOL_NAME_TAG,
            END_TOOL_NAME_TAG,
            BEGIN_TOOL_ARGS_TAG,
            END_TOOL_ARGS_TAG,
            EOT_TAG,
        ],
        key=len,
        reverse=True,
    )
    deltas: list[str] = []
    remaining = text
    while remaining:
        for tag in tags:
            if remaining.startswith(tag):
                deltas.append(tag)
                remaining = remaining[len(tag) :]
                break
        else:
            # Plain text until the next tag
            next_tag_pos = min(
                (remaining.find(tag) for tag in tags if remaining.find(tag) != -1),
                default=len(remaining),
            )
            deltas.append(remaining[:next_tag_pos])
            remaining = remaining[next_tag_pos:]
    return [d for d in deltas if d]


def test_extract_tool_calls_no_tools(
    parser: Plamo3ToolParser,
    mock_request: Mock,
) -> None:
    model_output = "This is normal text without tool calls."
    result = parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == model_output


def test_extract_tool_calls_incomplete_begin_tag_strips_fragment(
    parser: Plamo3ToolParser,
    mock_request: Mock,
) -> None:
    # max_tokens cut in the middle of BEGIN_TOOL_REQUESTS_TAG must not leak
    # the partial special-token fragment into content.
    model_output = "Intro<|plamo:begin_"
    result = parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == "Intro"


def test_extract_tool_calls_single(
    parser: Plamo3ToolParser,
    mock_request: Mock,
) -> None:
    args = json.dumps({"city": "Tokyo", "unit": "celsius"})
    body = _wrap_single_tool_call("get_current_weather", args)
    model_output = _wrap_tool_requests(body)

    result = parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_current_weather"
    assert result.tool_calls[0].function.arguments == args
    assert result.content == ""


def test_extract_tool_calls_parallel(
    parser: Plamo3ToolParser,
    mock_request: Mock,
) -> None:
    args1 = json.dumps({"text": "hello"})
    args2 = json.dumps({"a": 1, "b": 2})
    body = _wrap_single_tool_call("echo", args1) + _wrap_single_tool_call("sum", args2)
    model_output = _wrap_tool_requests(body)

    result = parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "echo"
    assert result.tool_calls[1].function.name == "sum"


def test_extract_tool_calls_with_leading_content(
    parser: Plamo3ToolParser,
    mock_request: Mock,
) -> None:
    args = json.dumps({"city": "Tokyo"})
    body = _wrap_single_tool_call("get_weather", args)
    model_output = "Intro text" + _wrap_tool_requests(body)

    result = parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called is True
    assert result.content == "Intro text"


def test_extract_tool_calls_malformed(
    parser: Plamo3ToolParser,
    mock_request: Mock,
) -> None:
    # Missing END_TOOL_REQUESTS_TAG
    model_output = f"{BEGIN_TOOL_REQUESTS_TAG}{BEGIN_TOOL_REQUEST_TAG}invalid"
    result = parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == ""


def test_streaming_single_tool_call(
    dummy_tokenizer: _DummyTokenizer,
    mock_request: Mock,
) -> None:
    parser = ToolParserManager.get_tool_parser("plamo3")(dummy_tokenizer)
    args = json.dumps({"city": "Tokyo"})
    full = _wrap_tool_requests(_wrap_single_tool_call("get_weather", args))
    content, tool_calls = run_tool_extraction(
        parser, _split_into_deltas(full), request=mock_request, streaming=True
    )
    assert content is None or content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == args


def test_streaming_tool_call_with_leading_content(
    dummy_tokenizer: _DummyTokenizer,
    mock_request: Mock,
) -> None:
    parser = ToolParserManager.get_tool_parser("plamo3")(dummy_tokenizer)
    args = json.dumps({"city": "Tokyo"})
    full = "Intro" + _wrap_tool_requests(_wrap_single_tool_call("get_weather", args))
    content, tool_calls = run_tool_extraction(
        parser, _split_into_deltas(full), request=mock_request, streaming=True
    )
    assert content == "Intro"
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"


def test_streaming_parallel_tool_calls(
    dummy_tokenizer: _DummyTokenizer,
    mock_request: Mock,
) -> None:
    parser = ToolParserManager.get_tool_parser("plamo3")(dummy_tokenizer)
    args1 = json.dumps({"x": 1})
    args2 = json.dumps({"y": 2})
    full = _wrap_tool_requests(
        _wrap_single_tool_call("first", args1) + _wrap_single_tool_call("second", args2)
    )
    content, tool_calls = run_tool_extraction(
        parser, _split_into_deltas(full), request=mock_request, streaming=True
    )
    assert content is None or content == ""
    assert len(tool_calls) == 2
    assert tool_calls[0].function.name == "first"
    assert tool_calls[1].function.name == "second"


def test_streaming_content_then_eot_no_partial_leak(
    dummy_tokenizer: _DummyTokenizer,
    mock_request: Mock,
) -> None:
    """A trailing EOT tag must not leak into content."""
    parser = ToolParserManager.get_tool_parser("plamo3")(dummy_tokenizer)
    full = "Hello world" + EOT_TAG
    # Split at PLaMo token boundaries so the EOT tag arrives as one unit.
    content, tool_calls = run_tool_extraction(
        parser, _split_into_deltas(full), request=mock_request, streaming=True
    )
    assert tool_calls == []
    assert content == "Hello world"


@pytest.mark.parametrize(
    "output,expected_tools,expected_content",
    [
        (
            _wrap_tool_requests(_wrap_single_tool_call("noop", json.dumps({}))),
            [ToolCall(function=FunctionCall(name="noop", arguments=json.dumps({})))],
            "",
        ),
        (
            "plain text",
            [],
            "plain text",
        ),
    ],
)
def test_parse_model_output_helper(output, expected_tools, expected_content) -> None:
    from vllm.tool_parsers.plamo3_tool_parser import parse_model_output

    content, tool_calls = parse_model_output(output)
    assert content == expected_content
    assert len(tool_calls) == len(expected_tools)
    for actual, expected in zip(tool_calls, expected_tools):
        assert actual.function.name == expected.function.name
        assert actual.function.arguments == expected.function.arguments


_PLAMO_PREFIX = "<|plamo:"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<|plamo:begin_", ""),
        ("<|plamo:begin_tool_requests:pla", ""),
        ("<|plamo:end_tool_arguments", ""),
        ("Hello<|plamo:begin_tool_requests", "Hello"),
    ],
)
def test_strip_trailing_partial_marker_tool(text, expected) -> None:
    assert strip_trailing_partial_marker(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello world",
        "ends with <|plamo",
        _wrap_tool_requests(_wrap_single_tool_call("noop", json.dumps({}))),
    ],
)
def test_strip_trailing_partial_marker_tool_leaves_complete(text) -> None:
    assert strip_trailing_partial_marker(text) == text


def test_no_tool_marker_leaks_on_truncation(
    parser: Plamo3ToolParser,
    mock_request: Mock,
) -> None:
    """Truncate representative tool-call outputs at every byte boundary and
    assert no PLaMo special-token markup survives in content or arguments.
    """
    tool_body = _wrap_single_tool_call("noop", json.dumps({}))
    full = _wrap_tool_requests(tool_body)
    for i in range(1, len(full) + 1):
        prefix = full[:i]
        result = parser.extract_tool_calls(prefix, request=mock_request)
        assert _PLAMO_PREFIX not in (result.content or "")
        for tc in result.tool_calls:
            assert _PLAMO_PREFIX not in tc.function.arguments
