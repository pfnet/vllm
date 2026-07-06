# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import (
    StreamingReasoningReconstructor,
    run_reasoning_extraction,
)
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.plamo3_reasoning_parser import (
    BEGIN_THINK_TAG,
    END_THINK_TAG,
    Plamo3ReasoningParser,
    strip_trailing_partial_marker,
)

# Token IDs for the PLaMo-3 think tags (mirrors the model tokenizer).
_BEGIN_THINK_TOKEN_IDS: list[int] = [256, 21279, 258]
_END_THINK_TOKEN_IDS: list[int] = [257, 21279, 258]

PLAMO3_MODEL_PATH = "/mnt/shared/models/plamo3-2604-31b-256k-id4281-non-reasoning-release-v3-gptq-4bit-qid0"  # noqa: E501


class _DummyTokenizer:
    """Minimal tokenizer for parser unit tests without a real model."""

    def __init__(self):
        self._vocab = {
            BEGIN_THINK_TAG: 256,
            END_THINK_TAG: 257,
            "thought": 100,
            " more thought": 101,
            "answer": 102,
            "prefix": 103,
            "text": 104,
        }
        self._id_to_token = {v: k for k, v in self._vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def tokenize(self, text: str) -> list[str]:
        # Keep token IDs empty for streaming tests so the parser falls back
        # to text-based reconstruction; real-tokenizer tests cover the
        # token-based path.
        return []

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return "".join(self._id_to_token.get(i, "") for i in token_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Return the canonical think-tag ids so the parser can initialise
        # its token-id checks; for everything else return [] to exercise
        # the text fallback path.
        if text == BEGIN_THINK_TAG:
            return list(_BEGIN_THINK_TOKEN_IDS)
        if text == END_THINK_TAG:
            return list(_END_THINK_TOKEN_IDS)
        return []


class _TokenAwareTokenizer:
    """Tokenizer that models the multi-token PLaMo-3 think tags.

    The think tags are not single tokens: ``_BEGIN_THINK_TOKEN_IDS`` and
    ``_END_THINK_TOKEN_IDS`` each decode to three sub-token pieces.  This
    tokenizer reproduces that structure so the token-id based streaming path
    can be exercised without the real model.
    """

    _ID_TO_TOKEN = {
        256: "<|plamo:begin_",
        257: "<|plamo:end_",
        21279: "think",
        258: ":plamo|>",
        1: "Plan",
        2: " step",
        3: " answer",
        4: " more",
    }
    _ANCHORS = {
        "<|plamo:begin_": [256],
        "<|plamo:end_": [257],
        "<|plamo:tag|>": [9],
    }

    def get_vocab(self) -> dict[str, int]:
        return {v: k for k, v in self._ID_TO_TOKEN.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == BEGIN_THINK_TAG:
            return list(_BEGIN_THINK_TOKEN_IDS)
        if text == END_THINK_TAG:
            return list(_END_THINK_TOKEN_IDS)
        return self._ANCHORS.get(text, [0])

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return "".join(self._ID_TO_TOKEN.get(i, "") for i in token_ids)


def _run_token_streaming(
    parser: Plamo3ReasoningParser,
    tokenizer: _TokenAwareTokenizer | _DummyTokenizer,
    token_ids: list[int],
) -> StreamingReasoningReconstructor:
    """Feed ``token_ids`` one at a time through the streaming parser."""
    reconstructor = StreamingReasoningReconstructor()
    previous_text = ""
    previous_tokens: list[int] = []
    for idx in range(1, len(token_ids) + 1):
        current_tokens = token_ids[:idx]
        current_text = tokenizer.decode(current_tokens, skip_special_tokens=False)
        delta_text = current_text[len(previous_text) :]
        delta_message = parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_tokens,
            current_tokens,
            [token_ids[idx - 1]],
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_tokens = current_tokens
    return reconstructor


@pytest.fixture
def token_tokenizer() -> _TokenAwareTokenizer:
    return _TokenAwareTokenizer()


def test_streaming_token_ids_basic_flow(token_tokenizer: _TokenAwareTokenizer) -> None:
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(token_tokenizer)
    token_ids = _BEGIN_THINK_TOKEN_IDS + [1, 2] + _END_THINK_TOKEN_IDS + [3]
    reconstructor = _run_token_streaming(parser, token_tokenizer, token_ids)
    assert reconstructor.reasoning == "Plan step"
    assert reconstructor.other_content == " answer"


def test_streaming_token_ids_content_before_begin(
    token_tokenizer: _TokenAwareTokenizer,
) -> None:
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(token_tokenizer)
    token_ids = [3] + _BEGIN_THINK_TOKEN_IDS + [1] + _END_THINK_TOKEN_IDS
    reconstructor = _run_token_streaming(parser, token_tokenizer, token_ids)
    assert reconstructor.reasoning is None
    assert reconstructor.other_content == token_tokenizer.decode(token_ids)


def test_streaming_token_ids_incomplete_no_end(
    token_tokenizer: _TokenAwareTokenizer,
) -> None:
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(token_tokenizer)
    token_ids = _BEGIN_THINK_TOKEN_IDS + [1, 2]
    reconstructor = _run_token_streaming(parser, token_tokenizer, token_ids)
    assert reconstructor.reasoning == "Plan step"
    assert reconstructor.other_content is None


def test_streaming_token_ids_empty_reasoning(
    token_tokenizer: _TokenAwareTokenizer,
) -> None:
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(token_tokenizer)
    token_ids = _BEGIN_THINK_TOKEN_IDS + _END_THINK_TOKEN_IDS + [3]
    reconstructor = _run_token_streaming(parser, token_tokenizer, token_ids)
    assert reconstructor.reasoning is None
    assert reconstructor.other_content == " answer"


@pytest.fixture
def dummy_tokenizer() -> _DummyTokenizer:
    return _DummyTokenizer()


@pytest.fixture(scope="module")
def plamo3_tokenizer() -> AutoTokenizer:
    if not Path(PLAMO3_MODEL_PATH).exists():
        pytest.skip(f"PLaMo-3 model not found at {PLAMO3_MODEL_PATH}")
    return AutoTokenizer.from_pretrained(PLAMO3_MODEL_PATH, trust_remote_code=True)


@pytest.fixture
def parser(dummy_tokenizer: _DummyTokenizer) -> Plamo3ReasoningParser:
    return ReasoningParserManager.get_reasoning_parser("plamo3")(dummy_tokenizer)


def test_reasoning_delimiter_properties(parser: Plamo3ReasoningParser) -> None:
    assert parser.reasoning_start_str == BEGIN_THINK_TAG
    assert parser.reasoning_end_str == END_THINK_TAG


def test_non_streaming_basic_extraction(parser: Plamo3ReasoningParser) -> None:
    reasoning_text = "Plan in progress..."
    content_text = "Final answer."
    model_output = f"{BEGIN_THINK_TAG}{reasoning_text}{END_THINK_TAG}{content_text}"

    reasoning, content = parser.extract_reasoning(model_output, request=None)
    assert reasoning == reasoning_text
    assert content == content_text


def test_non_streaming_no_begin_tag(parser: Plamo3ReasoningParser) -> None:
    model_output = "Plain text without thinking tags"
    reasoning, content = parser.extract_reasoning(model_output, request=None)
    assert reasoning is None
    assert content == model_output


def test_non_streaming_only_end_tag(parser: Plamo3ReasoningParser) -> None:
    model_output = END_THINK_TAG + "text"
    reasoning, content = parser.extract_reasoning(model_output, request=None)
    assert reasoning is None
    assert content == model_output


def test_non_streaming_no_end_tag(parser: Plamo3ReasoningParser) -> None:
    model_output = f"{BEGIN_THINK_TAG}incomplete reasoning"
    reasoning, content = parser.extract_reasoning(model_output, request=None)
    assert reasoning == "incomplete reasoning"
    assert content is None


def test_non_streaming_empty_content(parser: Plamo3ReasoningParser) -> None:
    model_output = f"{BEGIN_THINK_TAG}reasoning{END_THINK_TAG}"
    reasoning, content = parser.extract_reasoning(model_output, request=None)
    assert reasoning == "reasoning"
    assert content == ""


def test_extract_content_ids_always_empty(parser: Plamo3ReasoningParser) -> None:
    assert parser.extract_content_ids([1, 2, 3]) == []


def test_streaming_simple_flow(dummy_tokenizer: _DummyTokenizer) -> None:
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(dummy_tokenizer)
    deltas = [
        BEGIN_THINK_TAG,
        "thought",
        " more thought",
        END_THINK_TAG,
        "answer",
    ]
    reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)
    assert reasoning == "thought more thought"
    assert content == "answer"


def test_streaming_with_real_tokenizer(plamo3_tokenizer: AutoTokenizer) -> None:
    """Streaming extraction works when token IDs carry split special tags."""
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(plamo3_tokenizer)
    text = f"{BEGIN_THINK_TAG}first thought{END_THINK_TAG}output body"
    token_ids = plamo3_tokenizer.encode(text, add_special_tokens=False)
    # Feed the tokens one-by-one to simulate streaming fragmentation.
    reconstructor = StreamingReasoningReconstructor()
    previous_text = ""
    previous_tokens: list[int] = []
    for idx in range(1, len(token_ids) + 1):
        current_tokens = token_ids[:idx]
        delta_tokens = [token_ids[idx - 1]]
        current_text = plamo3_tokenizer.decode(
            current_tokens, skip_special_tokens=False
        )
        delta_text = plamo3_tokenizer.decode(delta_tokens, skip_special_tokens=False)
        delta_message = parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_tokens,
            current_tokens,
            delta_tokens,
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_tokens = current_tokens
    assert reconstructor.reasoning == "first thought"
    assert reconstructor.other_content == "output body"


def test_streaming_content_before_begin_tag(dummy_tokenizer: _DummyTokenizer) -> None:
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(dummy_tokenizer)
    deltas = [
        "prefix",
        "more",
        BEGIN_THINK_TAG,
        "thought",
        END_THINK_TAG,
        "suffix",
    ]
    reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)
    assert reasoning is None
    expected = "prefixmore" + BEGIN_THINK_TAG + "thought" + END_THINK_TAG + "suffix"
    assert content == expected


def test_streaming_incomplete_no_end_tag(dummy_tokenizer: _DummyTokenizer) -> None:
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(dummy_tokenizer)
    deltas = [BEGIN_THINK_TAG, "only thoughts"]
    reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)
    assert reasoning == "only thoughts"
    assert content is None


def test_streaming_uses_token_ids_not_lagging_text(
    dummy_tokenizer: _DummyTokenizer,
) -> None:
    """Token IDs are authoritative even when reconstructed text lags."""
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(dummy_tokenizer)
    # Token IDs already contain the full end tag, but simulated detokenized text
    # is still missing the closing part.
    current_token_ids = (
        list(_BEGIN_THINK_TOKEN_IDS) + [100] + list(_END_THINK_TOKEN_IDS)
    )
    current_text = f"{BEGIN_THINK_TAG}thought<|plamo:end_"
    delta_text = "<|plamo:end_"
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=[],
        current_token_ids=current_token_ids,
        delta_token_ids=[],
    )
    # The parser sees the end tag in token IDs and emits reasoning up to it.
    assert delta is not None
    assert delta.reasoning == "thought"
    assert delta.content is None


def test_streaming_tool_tag_transitions_to_content(
    dummy_tokenizer: _DummyTokenizer,
) -> None:
    """A non-reasoning tool tag should leave the reasoning phase."""
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(dummy_tokenizer)
    tool_tag = "<|plamo:begin_tool_requests:plamo|>"
    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text=tool_tag,
        delta_text=tool_tag,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
    )
    # The reasoning parser emits the tool markup as content so the tool parser
    # can take over in the orchestration layer.
    assert delta is not None
    assert delta.content == tool_tag
    assert delta.reasoning is None
    # The streaming end check now reports reasoning has ended.
    assert parser.is_reasoning_end_streaming([], []) is True


def test_is_reasoning_end_before_reasoning_tag(
    dummy_tokenizer: _DummyTokenizer,
) -> None:
    """is_reasoning_end tracks the end tag; streaming checks the full output.

    In the streaming path vLLM passes the entire generated sequence (including
    the current delta) as ``input_ids``, so ``delta_ids`` is not concatenated
    again.
    """
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(dummy_tokenizer)
    # Empty input.
    assert parser.is_reasoning_end([]) is True
    # Plain text tokens: not yet finished because the prompt never has tags.
    assert parser.is_reasoning_end([100, 200, 300]) is False
    # Begin tag without end tag: still in reasoning.
    assert parser.is_reasoning_end(list(_BEGIN_THINK_TOKEN_IDS) + [100, 200]) is False
    # End tag present: reasoning is finished.
    assert (
        parser.is_reasoning_end(
            list(_BEGIN_THINK_TOKEN_IDS) + [100] + list(_END_THINK_TOKEN_IDS)
        )
        is True
    )
    # Streaming: tokens that do not match the begin-think prefix are not
    # in a reasoning block.
    assert parser.is_reasoning_end_streaming([100], [200]) is True
    # Streaming: begin tag without end tag in the current delta means still
    # reasoning.
    assert (
        parser.is_reasoning_end_streaming(list(_BEGIN_THINK_TOKEN_IDS) + [100], [100])
        is False
    )
    # Streaming: end tag present in the current delta finishes reasoning.
    assert (
        parser.is_reasoning_end_streaming(
            list(_BEGIN_THINK_TOKEN_IDS) + [100] + list(_END_THINK_TOKEN_IDS),
            list(_END_THINK_TOKEN_IDS),
        )
        is True
    )


@pytest.mark.skipif(
    not Path(PLAMO3_MODEL_PATH).exists(),
    reason=f"PLaMo-3 model not found at {PLAMO3_MODEL_PATH}",
)
def test_tokenizer_end_think_token_ids(plamo3_tokenizer: AutoTokenizer) -> None:
    ids = plamo3_tokenizer.encode(END_THINK_TAG, add_special_tokens=False)
    assert ids == _END_THINK_TOKEN_IDS


@pytest.mark.skipif(
    not Path(PLAMO3_MODEL_PATH).exists(),
    reason=f"PLaMo-3 model not found at {PLAMO3_MODEL_PATH}",
)
def test_is_reasoning_end_streaming_with_real_tokenizer(
    plamo3_tokenizer: AutoTokenizer,
) -> None:
    parser = ReasoningParserManager.get_reasoning_parser("plamo3")(plamo3_tokenizer)
    # END_THINK appears inside the delta window of a reasoning block
    input_ids = _BEGIN_THINK_TOKEN_IDS + [1, 2, 3] + _END_THINK_TOKEN_IDS + [42]
    assert parser.is_reasoning_end_streaming(input_ids, _END_THINK_TOKEN_IDS) is True

    # END_THINK split across the previous delta boundary and current delta
    input_ids2 = _BEGIN_THINK_TOKEN_IDS + [1] + [257] + [21279, 258]
    assert parser.is_reasoning_end_streaming(input_ids2, [21279, 258]) is True

    # END_THINK split so only the last token arrives in the current delta
    input_ids2b = _BEGIN_THINK_TOKEN_IDS + [1, 257, 21279] + [258]
    assert parser.is_reasoning_end_streaming(input_ids2b, [258]) is True

    # Inside a reasoning block, END_THINK not in current delta
    input_ids3 = _BEGIN_THINK_TOKEN_IDS + [99]
    assert parser.is_reasoning_end_streaming(input_ids3, [99]) is False

    # Non-reasoning output does not contain think tags; the orchestration
    # should transition out of the reasoning phase immediately.
    input_ids4 = [42, 43]
    assert parser.is_reasoning_end_streaming(input_ids4, [43]) is True


_PLAMO_PREFIX = "<|plamo:"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<|plamo:begin_", ""),
        ("<|plamo:begin_think:pla", ""),
        ("<|plamo:end_", ""),
        ("Hello<|plamo:begin_", "Hello"),
    ],
)
def test_strip_trailing_partial_marker_reasoning(text, expected) -> None:
    assert strip_trailing_partial_marker(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello world",
        "ends with <|plamo",
        f"{BEGIN_THINK_TAG}thought{END_THINK_TAG}answer",
    ],
)
def test_strip_trailing_partial_marker_reasoning_leaves_complete(text) -> None:
    assert strip_trailing_partial_marker(text) == text


def test_no_reasoning_marker_leaks_on_truncation(parser: Plamo3ReasoningParser) -> None:
    """Truncate representative reasoning outputs at every byte boundary and
    assert no PLaMo special-token markup survives in reasoning or content.
    """
    outputs = [
        f"{BEGIN_THINK_TAG}step one.{END_THINK_TAG}The final answer is 42.",
        f"{BEGIN_THINK_TAG}thinking hard<|plamo:end_",
    ]
    for full in outputs:
        for i in range(1, len(full) + 1):
            prefix = full[:i]
            reasoning, content = parser.extract_reasoning(prefix, request=None)
            assert _PLAMO_PREFIX not in (reasoning or "")
            assert _PLAMO_PREFIX not in (content or "")
