# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import json
import logging
from unittest.mock import Mock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.utils.plamo3_parser_common import (
    BEGIN_THINK_TAG,
    BEGIN_TOOL_ARGS_TAG,
    BEGIN_TOOL_NAME_TAG,
    BEGIN_TOOL_REQUEST_TAG,
    BEGIN_TOOL_REQUESTS_TAG,
    END_THINK_TAG,
    END_TOOL_ARGS_TAG,
    END_TOOL_NAME_TAG,
    END_TOOL_REQUEST_TAG,
    END_TOOL_REQUESTS_TAG,
   EOT_TAG,
)

PlamoToolParser = ToolParserManager.get_tool_parser("plamo3")


class _DummyTokenizer:
    """Minimal tokenizer with PLaMo-3 special-token ID mappings for unit tests."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self.bos_token_id: int | None = 1

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def tokenize(self, text: str) -> list[str]:
        return [text] if text else []

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Grammar anchors (single-token prefixes/suffixes)
        if text == "<|plamo:begin_":
            return [256]
        if text == "<|plamo:end_":
            return [257]
        if text == ":plamo|>":
            return [258]
        # Reasoning tags
        if text == "<|plamo:begin_think:plamo|>":
            return [256, 21279, 258]
        if text == "<|plamo:end_think:plamo|>":
            return [257, 21279, 258]
        # Tool tags
        if text == "<|plamo:begin_tool_requests:plamo|>":
            return [256, 13672, 95, 31026, 258]
        if text == "<|plamo:end_tool_requests:plamo|>":
            return [257, 13672, 95, 31026, 258]
        if text == "<|plamo:begin_tool_request:plamo|>":
            return [256, 13672, 95, 2475, 258]
        if text == "<|plamo:end_tool_request:plamo|>":
            return [257, 13672, 95, 2475, 258]
        if text == "<|plamo:begin_tool_name:plamo|>":
            return [256, 13672, 50416, 258]
        if text == "<|plamo:end_tool_name:plamo|>":
            return [257, 13672, 50416, 258]
        # Composite BEGIN_TOOL_ARGS_TAG
        if text == (
            "<|plamo:begin_tool_arguments:plamo|>"
            "<|plamo:constrain|>json<|plamo:msg|>"
        ):
            return [256, 13672, 95, 19868, 258, 31, 349, 19]
        if text == "<|plamo:end_tool_arguments:plamo|>":
            return [257, 13672, 95, 19868, 258]
        if text == "<|plamo:constrain|>":
            return [31]
        if text == "<|plamo:msg|>":
            return [19]
        if text == "<|plamo:tag|>":
            return [16]
        # For streaming tests, unknown text (including tag fragments) is
        # treated as generating no token ids.
        return []

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return "".join(str(token_id) for token_id in token_ids)


class StreamingToolCallReconstructor:
    """Aggregate streaming DeltaMessage outputs for tool parser tests."""

    def __init__(self):
        self.content: str | None = None
        self.tool_deltas: list[DeltaMessage] = []

    def append_delta(self, delta: DeltaMessage | None):
        if delta is None:
            return
        if delta.content is not None:
            self.content = (
                delta.content if self.content is None else self.content + delta.content
            )
        if delta.tool_calls:
            self.tool_deltas.extend(delta.tool_calls)


def run_tool_parser_streaming(
    parser,
    deltas: list[str],
    request=None,
    delta_token_ids: list[list[int] | None] | None = None,
):
    """Drive ``parser.extract_tool_calls_streaming`` across ``deltas``.

    Token ids are reconstructed from the dummy tokenizer's ``encode`` method
    by default, so the parser receives realistic ``current_token_ids`` and
    ``delta_token_ids``. For tag fragments the dummy tokenizer cannot encode,
    the caller can pass explicit ``delta_token_ids`` for each delta.
    """
    req = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingToolCallReconstructor()
    previous_text = ""
    previous_token_ids: list[int] = []
    for i, delta_text in enumerate(deltas):
        if (
            delta_token_ids is not None
            and i < len(delta_token_ids)
            and delta_token_ids[i] is not None
        ):
            ids = list(delta_token_ids[i])
        else:
            ids = list(
                parser.model_tokenizer.encode(
                    delta_text, add_special_tokens=False
                )
            )
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + ids
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=ids,
            request=req,
        )
        reconstructor.append_delta(msg)
        previous_text = current_text
        previous_token_ids = current_token_ids
    return reconstructor


@pytest.fixture
def tokenizer():
    return _DummyTokenizer()




@pytest.fixture
def parser(tokenizer):
    return PlamoToolParser(tokenizer)


@pytest.fixture
def mock_request():
    # ChatCompletionRequest の最低限参照される属性のみを持つモック
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


def test_extract_tool_calls_no_tools(parser, mock_request):
    model_output = "これはツール呼び出しの無い通常テキストです。"
    result = parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == model_output


@pytest.mark.parametrize(
    "content,tool_bodies,expected_calls",
    [
        (
            "前置き。",
            _wrap_single_tool_call(
                "get_current_weather",
                json.dumps({"city": "Tokyo", "unit": "celsius"}),
            ),
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps({"city": "Tokyo", "unit": "celsius"}),
                    )
                )
            ],
        ),
        (
            "ヘッダ本文。",
            _wrap_single_tool_call(
                "echo",
                json.dumps({"text": "hello"}),
            )
            + _wrap_single_tool_call(
                "sum",
                json.dumps({"a": 1, "b": 2}),
            ),
            [
                ToolCall(
                    function=FunctionCall(
                        name="echo", arguments=json.dumps({"text": "hello"})
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="sum", arguments=json.dumps({"a": 1, "b": 2})
                    )
                ),
            ],
        ),
    ],
)
def test_extract_tool_calls_basic(
    parser, content, tool_bodies, expected_calls, mock_request
):
    model_output = content + _wrap_tool_requests(tool_bodies)
    result = parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called is True
    assert len(result.tool_calls) == len(expected_calls)
    for actual, expected in zip(result.tool_calls, expected_calls):
        assert actual.type == "function"
        assert actual.function.name == expected.function.name
        assert json.loads(actual.function.arguments) == json.loads(
            expected.function.arguments
        )
    assert result.content == content


def test_extract_tool_calls_empty_arguments(parser, mock_request):
    content = "説明。"
    tool_bodies = _wrap_single_tool_call("noop", "{}")
    result = parser.extract_tool_calls(
        content + _wrap_tool_requests(tool_bodies), request=mock_request
    )
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "noop"
    assert result.tool_calls[0].function.arguments == "{}"
    assert result.content == content


def test_extract_tool_calls_malformed(parser, mock_request):
    # ENDタグ不足などの不正フォーマットは安全側に倒してtool_calls=[]
    malformed = (
        f"前文。{BEGIN_TOOL_REQUESTS_TAG}{BEGIN_TOOL_REQUEST_TAG}"
        f"{BEGIN_TOOL_NAME_TAG}broken{END_TOOL_NAME_TAG}"
        f'{BEGIN_TOOL_ARGS_TAG}{{"a": 1}}'  # END_TOOL_ARGS_TAGを欠落
    )
    result = parser.extract_tool_calls(malformed, request=mock_request)
    assert not result.tools_called
    # 少なくとも例外は出ず、listで返る
    assert isinstance(result.tool_calls, list)


def test_extract_tool_calls_with_thinking_tags(parser, mock_request):
    # 推論タグはcontentとして扱われる（Plamo独自タグ）
    think = f"{BEGIN_THINK_TAG}考え中...{END_THINK_TAG}\n"
    content = think + "天気を確認します。"
    tool_bodies = _wrap_single_tool_call(
        "get_weather",
        json.dumps({"city": "上海", "date": "2025-08-01"}, ensure_ascii=False),
    )
    model_output = content + _wrap_tool_requests(tool_bodies)
    result = parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments)["city"] == "上海"
    assert result.content == content


def test_extract_tool_calls_incomplete_requests_block(parser, mock_request):
    # END_TOOL_REQUESTS_TAG が欠落 → tool_calls=[] で安全側
    content = "前置きテキスト。"
    body = _wrap_single_tool_call("ping", json.dumps({"x": 1}))
    malformed = content + BEGIN_TOOL_REQUESTS_TAG + body  # END_TOOL_REQUESTS_TAG を欠落
    result = parser.extract_tool_calls(malformed, request=mock_request)
    assert result.tool_calls == []
    assert result.tools_called is False
    assert result.content == content


def test_streaming_with_content_before_requests(parser, mock_request):
    # BEGIN_TOOL_REQUESTS_TAG 直前までのcontentがデルタで返る
    content = "案内文。"
    delta = content + BEGIN_TOOL_REQUESTS_TAG
    msg = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=delta,
        delta_text=delta,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )
    assert msg is not None and hasattr(msg, "content")
    assert msg.content == content


def test_streaming_tool_call_after_content_emits_name_and_args(tokenizer, mock_request):
    """content が先行しても、続く tool call の name/arguments が emit される。

    回帰テスト: CONTENT フェーズで _stream_pos がタグ位置へ同期されておらず、
    content が非空だと _advance_tag の基点がずれて後続の BEGIN_TOOL_REQUEST を
    取りこぼし、tool_call が空になっていた（json_schema 有効時の streaming で
    "required must produce at least one tool_call" が失敗する e2e 回帰）。
    content は fenced JSON（json_schema 有効時の content body）を模す。
    """
    parser = PlamoToolParser(tokenizer)
    content = '```json\n{"name": "Alice", "age": 30}\n```'
    tool = _wrap_single_tool_call("get_weather", json.dumps({"city": "Tokyo"}))
    full = content + _wrap_tool_requests(tool)

    got_content = ""
    names, args = [], []
    prev = ""
    for chunk in _iter_tokens(full):
        cur = prev + chunk
        msg = _call_streaming(parser, prev, cur, chunk, mock_request)
        if msg is not None:
            if msg.content:
                got_content += msg.content
            for d in msg.tool_calls or []:
                if d.function is None:
                    continue
                if d.function.name:
                    names.append(d.function.name)
                if d.function.arguments:
                    args.append(d.function.arguments)
        prev = cur

    assert got_content == content
    assert names == ["get_weather"]
    assert json.loads("".join(args)) == {"city": "Tokyo"}


def test_streaming_content_delta(parser, mock_request):
    # EOTタグが来た時点で手前のcontentがflushされる
    delta = "こんにちは" + EOT_TAG
    msg = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=delta,
        delta_text=delta,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )
    assert msg is not None and hasattr(msg, "content")
    assert msg.content == "こんにちは"


def test_streaming_name_then_args(parser, mock_request):
    # ツール名通知→引数の部分的通知→引数終了 までの流れを検証
    parts = [
        BEGIN_TOOL_REQUESTS_TAG
        + BEGIN_TOOL_REQUEST_TAG
        + BEGIN_TOOL_NAME_TAG
        + "sum"
        + END_TOOL_NAME_TAG,
        BEGIN_TOOL_ARGS_TAG + '{"a":1,"b":',
        "2}"
        + END_TOOL_ARGS_TAG
        + END_TOOL_REQUEST_TAG
        + END_TOOL_REQUESTS_TAG
        + EOT_TAG,
    ]

    # 1) ツール名が出た段階でDeltaToolCall(name=...)が返る
    msg1 = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=parts[0],
        delta_text=parts[0],
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )
    assert msg1 is not None and msg1.tool_calls, "ツール名のDeltaが返るはず"
    tc = msg1.tool_calls[0]
    assert tc.type == "function"
    assert tc.function is not None
    assert tc.function.name == "sum"

    # 2) 引数途中チャンク: END_TOOL_ARGS_TAG の prefix で終わっていないので即 emit される
    msg2 = parser.extract_tool_calls_streaming(
        previous_text=parts[0],
        current_text=parts[0] + parts[1],
        delta_text=parts[1],
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )
    assert msg2 is not None and msg2.tool_calls
    assert msg2.tool_calls[0].function is not None
    args_so_far = msg2.tool_calls[0].function.arguments or ""
    assert args_so_far == '{"a":1,"b":'

    # 3) 引数の残りと終了タグを流すと、残差のargumentsチャンクが返る
    msg3 = parser.extract_tool_calls_streaming(
        previous_text=parts[0] + parts[1],
        current_text="".join(parts),
        delta_text=parts[2],
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=mock_request,
    )
    assert msg3 is not None and msg3.tool_calls
    assert msg3.tool_calls[0].function is not None
    # 終了タグ直前の差分が返る（"2}"）
    assert (msg3.tool_calls[0].function.arguments or "") == "2}"


# ---------------------------------------------------------------------------
# EOS (EOT_TAG) 到着時のバッファフラッシュに関するテスト
# ---------------------------------------------------------------------------

# "<|plamo:begin_"、"<|plamo:end_"、"<|plamo:tag|>" は単一トークンという前提。
# この前提のもとでは、これらの文字列がバッファ末尾に中途半端に現れることはない。
_SINGLE_TOKENS = ("<|plamo:begin_", "<|plamo:end_", "<|plamo:tag|>")


def _iter_tokens(s: str):
    """単一トークン境界を尊重しながら s をチャンク単位で yield する。

    _SINGLE_TOKENS に該当する部分文字列はひとまとまりで yield し、
    それ以外の文字は 1 文字ずつ yield する。
    """
    i = 0
    while i < len(s):
        for tok in _SINGLE_TOKENS:
            if s[i : i + len(tok)] == tok:
                yield s[i : i + len(tok)]
                i += len(tok)
                break
        else:
            yield s[i]
            i += 1


def _call_streaming(parser, previous_text, current_text, delta_text, request):
    """extract_tool_calls_streaming のラッパー（token_ids は空リスト固定）。"""
    return parser.extract_tool_calls_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )


def test_streaming_begin_tag_prefix_held_content(tokenizer, mock_request):
    """BEGIN_TOOL_REQUESTS_TAG の prefix でコンテンツが保留される。

    "<|plamo:begin_" は単一トークンなので途中状態では現れない。
    バッファ末尾が "<|plamo:begin_" + 後続文字列の一部で終わる場合、
    その部分は保留され、前にあるコンテンツのみ即 emit される。
    """
    parser = PlamoToolParser(tokenizer)
    anchor = "<|plamo:begin_"
    # anchor は単一トークンなので一度に届く。
    # その後に続く残余文字列 "tool" は別トークンで届きうる。
    partial_suffix = BEGIN_TOOL_REQUESTS_TAG[len(anchor) : len(anchor) + 4]  # "tool"
    content_before = "Hello"
    delta1 = content_before + anchor + partial_suffix  # "Hello<|plamo:begin_tool"

    # バッファ末尾が "<|plamo:begin_tool" → anchor より前の "Hello" は即 emit、
    # "<|plamo:begin_tool" は保留される。
    msg1 = _call_streaming(parser, "", delta1, delta1, mock_request)
    assert msg1 is not None, "anchor より前の content は即 emit される"
    assert msg1.content == content_before


def test_streaming_eos_with_no_content_returns_none(tokenizer, mock_request):
    """コンテンツなしで EOT だけが来た場合は None を返す（余分な emit をしない）。"""
    parser = PlamoToolParser(tokenizer)

    msg = _call_streaming(parser, "", EOT_TAG, EOT_TAG, mock_request)
    # コンテンツがないので DeltaMessage は返らない
    assert msg is None


def test_streaming_eos_arrives_as_single_token(tokenizer, mock_request):
    """EOT_TAG は単一トークンとして届くため、途中分割は発生しない。

    コンテンツが先に届き、続いて EOT_TAG が完全な形で届く場合の動作を確認する。
    """
    parser = PlamoToolParser(tokenizer)
    content = "テスト"

    # チャンク1: コンテンツのみ → 即 emit
    msg1 = _call_streaming(parser, "", content, content, mock_request)
    assert msg1 is not None, "content は即 emit される"
    assert msg1.content == content

    # チャンク2: EOT_TAG が単一トークンとして完全な形で届く → 追加コンテンツなし
    current2 = content + EOT_TAG
    msg2 = _call_streaming(parser, content, current2, EOT_TAG, mock_request)
    assert msg2 is None, "EOT 到着時に追加 content はない（既に emit 済み）"


def test_streaming_eos_after_tool_call_emits_no_content(tokenizer, mock_request):
    """ツール呼び出しが完結した後に EOT が来ても、コンテンツの delta は返らない。

    単一トークン前提（"<|plamo:begin_"、"<|plamo:end_"、"<|plamo:tag|>" は各 1 トークン）
    のもとで、_iter_tokens を使ってトークン境界を尊重したチャンクで送信する。
    """
    parser = PlamoToolParser(tokenizer)
    tool_body = _wrap_single_tool_call("ping", json.dumps({"x": 1}))
    full = _wrap_tool_requests(tool_body)  # BEGIN...END...EOT

    collected = []
    prev = ""
    for chunk in _iter_tokens(full):
        cur = prev + chunk
        msg = _call_streaming(parser, prev, cur, chunk, mock_request)
        if msg is not None:
            collected.append(msg)
        prev = cur

    # ツール名・引数の DeltaMessage は含まれるが、content delta は含まれない
    content_deltas = [m for m in collected if m.content]
    assert content_deltas == [], (
        f"ツール呼び出しのみのシーケンスでコンテンツ delta が発生: {content_deltas}"
    )


def test_streaming_content_emits_without_eot_when_no_tag_prefix(
    tokenizer, mock_request
):
    """タグの prefix で終わっていないコンテンツは EOT なしで即座に emit されるべき。

    skip_special_tokens=True や stop_token_ids 使用時など EOT_TAG が
    delta_text に現れない場合でも、コンテンツが失われてはならない。
    """
    parser = PlamoToolParser(tokenizer)
    content = "Hello"  # どのタグの prefix でも終わっていない

    msg = _call_streaming(parser, "", content, content, mock_request)

    # 期待: EOT なしでも即座に emit される
    # 現状: safe_until = 5 - 34 = 0 → None のまま（バグ）
    assert msg is not None, (
        "タグ prefix で終わっていないコンテンツは即座に emit されるべき"
    )
    assert msg.content == content


def test_streaming_id_based_begin_tool_requests_fragmented(tokenizer, mock_request):
    """BEGIN_TOOL_REQUESTS_TAG が delta 境界をまたいでも ID ベースで検出される。

    タグの token id 列が複数 delta に分割されて届く場合、content は即座に emit
    され、タグテキストが content に漏れない。
    """
    parser = PlamoToolParser(tokenizer)
    ids = tokenizer.encode(BEGIN_TOOL_REQUESTS_TAG, add_special_tokens=False)
    assert len(ids) == 5

    # delta1: content + タグの途中文字列（anchor 直後）
    delta1 = "前置き" + BEGIN_TOOL_REQUESTS_TAG[: len("<|plamo:begin_t")]
    # delta2: タグの残り
    delta2 = BEGIN_TOOL_REQUESTS_TAG[len("<|plamo:begin_t") :]

    recon = run_tool_parser_streaming(
        parser,
        [delta1, delta2],
        request=mock_request,
        delta_token_ids=[ids[:3], ids[3:]],
    )
    assert recon.content == "前置き"


def test_streaming_id_based_end_tool_args_fragmented(tokenizer, mock_request):
    """END_TOOL_ARGS_TAG が delta 境界をまたいでも ID ベースで検出される。

    引数 delta がタグ直前で止まり、タグテキストは arguments に漏れない。
    """
    parser = PlamoToolParser(tokenizer)
    prefix = (
        BEGIN_TOOL_REQUESTS_TAG
        + BEGIN_TOOL_REQUEST_TAG
        + BEGIN_TOOL_NAME_TAG
        + "sum"
        + END_TOOL_NAME_TAG
        + BEGIN_TOOL_ARGS_TAG
    )
    args = '{"a":1,"b":2}'
    # Split END_TOOL_ARGS_TAG in the middle.
    end_ids = tokenizer.encode(END_TOOL_ARGS_TAG, add_special_tokens=False)
    assert len(end_ids) == 5
    split_idx = len(end_ids) // 2
    delta1 = prefix + args + END_TOOL_ARGS_TAG[: len(END_TOOL_ARGS_TAG) // 2]
    delta2 = (
        END_TOOL_ARGS_TAG[len(END_TOOL_ARGS_TAG) // 2 :]
        + END_TOOL_REQUEST_TAG
        + END_TOOL_REQUESTS_TAG
        + EOT_TAG
    )

    recon = run_tool_parser_streaming(
        parser,
        [delta1, delta2],
        request=mock_request,
        delta_token_ids=[
            tokenizer.encode(delta1, add_special_tokens=False),
            end_ids[split_idx:]
            + tokenizer.encode(
                END_TOOL_REQUEST_TAG + END_TOOL_REQUESTS_TAG + EOT_TAG,
                add_special_tokens=False,
            ),
        ],
    )
    arg_deltas = [
        d.function.arguments
        for d in recon.tool_deltas
        if d.function is not None and d.function.arguments is not None
    ]
    assert args in "".join(arg_deltas)
    assert END_TOOL_ARGS_TAG not in "".join(arg_deltas)


def test_streaming_id_based_tag_spanning_window(tokenizer, mock_request):
    """タグの先頭 4 token が前回 delta、最後の 1 token が今回 delta の場合に検出される。"""
    parser = PlamoToolParser(tokenizer)
    ids = tokenizer.encode(BEGIN_TOOL_REQUESTS_TAG, add_special_tokens=False)
    assert len(ids) == 5

    # Use text boundaries that split after 4 tokens. The dummy tokenizer maps
    # each distinct substring to a single token only when passed exactly, so we
    # split at known character boundaries for the test tag.
    split_char = len(BEGIN_TOOL_REQUESTS_TAG) - 1
    delta1 = "前置き" + BEGIN_TOOL_REQUESTS_TAG[:split_char]
    delta2 = BEGIN_TOOL_REQUESTS_TAG[split_char:]

    recon = run_tool_parser_streaming(
        parser,
        [delta1, delta2],
        request=mock_request,
        delta_token_ids=[ids[:-1], ids[-1:]],
    )
    assert recon.content == "前置き"
    assert not any(
        END_TOOL_REQUESTS_TAG in (d.function.arguments or "")
        for d in recon.tool_deltas
    )


def test_streaming_id_based_wrong_middle_token_not_detected(tokenizer, mock_request):
    """タグと似たトークン列だが途中が異なる場合、誤検出されない。"""
    parser = PlamoToolParser(tokenizer)
    # Build a sequence whose token ids match BEGIN_TOOL_REQUESTS_TAG at the
    # anchors but differ in the middle token.
    ids = list(tokenizer.encode(BEGIN_TOOL_REQUESTS_TAG, add_special_tokens=False))
    ids[2] = 99999  # corrupt the middle token

    recon = run_tool_parser_streaming(
        parser,
        ["前置き"],
        request=mock_request,
        delta_token_ids=[ids],
    )
    assert recon.content == "前置き"


def test_streaming_content_accumulated_across_chunks_emits_per_chunk(
    tokenizer, mock_request
):
    """タグ prefix で終わっていないチャンクは各々即 emit される。

    EOT を待たずに各チャンクのコンテンツが順次 emit されることを確認する。
    """
    parser = PlamoToolParser(tokenizer)
    chunks = ["AB", "CD", "EF"]

    prev = ""
    emitted = []
    for chunk in chunks:
        cur = prev + chunk
        msg = _call_streaming(parser, prev, cur, chunk, mock_request)
        if msg is not None and msg.content:
            emitted.append(msg.content)
        prev = cur

    assert emitted == chunks, f"各チャンクが即 emit されるべき: {emitted}"


def test_streaming_begin_tag_prefix_held_across_chunks(tokenizer, mock_request):
    """BEGIN タグの prefix で終わるチャンクのみ保留され、タグ確定後にフラッシュされる。

    単一トークン前提のもとでは "<|plamo:begin_" は常に一括で届くため、
    バッファ末尾が "<|plamo:begin_" + 後続文字列の一部で終わる状況が保留対象。
    """
    parser = PlamoToolParser(tokenizer)
    anchor = "<|plamo:begin_"

    # チャンク1: 通常コンテンツ → 即 emit
    chunk1 = "Hello"
    msg1 = _call_streaming(parser, "", chunk1, chunk1, mock_request)
    assert msg1 is not None and msg1.content == chunk1

    # チャンク2: コンテンツ + anchor + 後続の一部 → "World" のみ emit、残りは保留
    partial_suffix = BEGIN_TOOL_REQUESTS_TAG[len(anchor) : len(anchor) + 4]  # "tool"
    chunk2 = "World" + anchor + partial_suffix
    prev2 = chunk1
    cur2 = prev2 + chunk2
    msg2 = _call_streaming(parser, prev2, cur2, chunk2, mock_request)
    # "World" は即 emit、"<|plamo:begin_tool" は保留
    assert msg2 is not None and msg2.content == "World"

    # チャンク3: BEGIN_TOOL_REQUESTS_TAG の残り → タグ確定、追加コンテンツなし
    rest_of_tag = BEGIN_TOOL_REQUESTS_TAG[len(anchor) + len(partial_suffix) :]
    cur3 = cur2 + rest_of_tag
    msg3 = _call_streaming(parser, cur2, cur3, rest_of_tag, mock_request)
    assert msg3 is None
