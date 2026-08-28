# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from unittest.mock import Mock

import pytest

from tests.reasoning.test_plamo3_reasoning_parser import _DummyTokenizer
from tests.tool_parsers.utils import StreamingToolReconstructor
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    FunctionCall,
    ToolCall,
)
from vllm.reasoning.plamo3_reasoning_parser import (
    _ALL_SPECIAL_TAGS,
    BEGIN_THINK_TAG,
    END_THINK_TAG,
    Plamo3ReasoningParser,
    strip_trailing_partial_marker,
)
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
)


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
    reconstructor = StreamingToolReconstructor()
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
                parser.model_tokenizer.encode(delta_text, add_special_tokens=False)
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
        if msg is not None:
            reconstructor.append_delta(msg)
        previous_text = current_text
        previous_token_ids = current_token_ids
    return reconstructor


@pytest.fixture
def tokenizer():
    return _DummyTokenizer()


@pytest.fixture
def parser(tokenizer):
    return Plamo3ToolParser(tokenizer)


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


def test_extract_tool_calls_two_requests(parser, mock_request):
    """Non-streaming counterpart of
    ``test_streaming_two_tool_requests_increment_index_and_id``.

    Parses two tool_request blocks at once and verifies that names and arguments
    are restored in order.
    """
    body = _wrap_single_tool_call(
        "echo", json.dumps({"text": "hi"})
    ) + _wrap_single_tool_call("sum", json.dumps({"a": 1, "b": 2}))
    model_output = _wrap_tool_requests(body)
    result = parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "echo"
    assert json.loads(result.tool_calls[0].function.arguments) == {"text": "hi"}
    assert result.tool_calls[1].function.name == "sum"
    assert json.loads(result.tool_calls[1].function.arguments) == {"a": 1, "b": 2}


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
    parser = Plamo3ToolParser(tokenizer)
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

    # 2) 引数途中チャンク: END_TOOL_ARGS_TAG の prefix で終わっていないので
    # 即 emit される
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


def test_streaming_eos_with_no_content_returns_none(tokenizer, mock_request):
    """コンテンツなしで EOT だけが来た場合は None を返す（余分な emit をしない）。"""
    parser = Plamo3ToolParser(tokenizer)

    msg = _call_streaming(parser, "", EOT_TAG, EOT_TAG, mock_request)
    # コンテンツがないので DeltaMessage は返らない
    assert msg is None


def test_streaming_eos_arrives_as_single_token(tokenizer, mock_request):
    """EOT_TAG は単一トークンとして届くため、途中分割は発生しない。

    コンテンツが先に届き、続いて EOT_TAG が完全な形で届く場合の動作を確認する。
    """
    parser = Plamo3ToolParser(tokenizer)
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

    単一トークン前提（"<|plamo:begin_"、"<|plamo:end_"、
    "<|plamo:tag|>" は各 1 トークン）のもとで、
    _iter_tokens を使ってトークン境界を尊重したチャンクで送信する。
    """
    parser = Plamo3ToolParser(tokenizer)
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


@pytest.mark.parametrize(
    "split_char",
    [
        len(BEGIN_TOOL_REQUESTS_TAG[: len("<|plamo:begin_t")]),  # anchor 直後
        len(BEGIN_TOOL_REQUESTS_TAG) - 1,  # 末尾 1 文字
    ],
    ids=["split_after_anchor", "split_last_char"],
)
def test_streaming_begin_tool_requests_fragmented_no_leak(
    tokenizer, mock_request, split_char
):
    """BEGIN_TOOL_REQUESTS_TAG が delta 境界をまたいでも content に漏れない。

    タグ文字列が複数 delta に分割されて届く場合、タグより前の content は即座に
    emit され、タグの断片は compute_safe_until で保留される。
    2 つの代表分割点（anchor 直後 / 末尾 1 文字）で検証する。
    """
    parser = Plamo3ToolParser(tokenizer)
    ids = tokenizer.encode(BEGIN_TOOL_REQUESTS_TAG, add_special_tokens=False)
    assert len(ids) == 5

    delta1 = "前置き" + BEGIN_TOOL_REQUESTS_TAG[:split_char]
    delta2 = BEGIN_TOOL_REQUESTS_TAG[split_char:]
    # タグのトークン境界は分割点に依存しないので、単純に前半/後半で分割する。
    delta_ids1 = ids[: len(ids) // 2]
    delta_ids2 = ids[len(ids) // 2 :]

    recon = run_tool_parser_streaming(
        parser,
        [delta1, delta2],
        request=mock_request,
        delta_token_ids=[delta_ids1, delta_ids2],
    )
    assert recon.other_content == "前置き"


def test_streaming_end_tool_args_fragmented_no_leak(tokenizer, mock_request):
    """END_TOOL_ARGS_TAG が delta 境界をまたいでも arguments に漏れない。

    引数 delta がタグ直前で止まり、タグの断片は compute_safe_until で保留される。
    """
    parser = Plamo3ToolParser(tokenizer)
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
    arguments = "".join(call.function.arguments for call in recon.tool_calls)
    assert args in arguments
    assert END_TOOL_ARGS_TAG not in arguments


def test_streaming_token_ids_do_not_affect_text_parsing(tokenizer, mock_request):
    """パースはテキストベースで行われ、token id 列の内容に影響されない。"""
    parser = Plamo3ToolParser(tokenizer)
    # Pass token ids resembling BEGIN_TOOL_REQUESTS_TAG (with a corrupt middle
    # token); the text contains no tag, so everything is content.
    ids = list(tokenizer.encode(BEGIN_TOOL_REQUESTS_TAG, add_special_tokens=False))
    ids[2] = 99999  # corrupt the middle token

    recon = run_tool_parser_streaming(
        parser,
        ["前置き"],
        request=mock_request,
        delta_token_ids=[ids],
    )
    assert recon.other_content == "前置き"


def test_streaming_content_accumulated_across_chunks_emits_per_chunk(
    tokenizer, mock_request
):
    """タグ prefix で終わっていないチャンクは各々即 emit される。

    EOT を待たずに各チャンクのコンテンツが順次 emit されることを確認する。
    """
    parser = Plamo3ToolParser(tokenizer)
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
    parser = Plamo3ToolParser(tokenizer)
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


def test_streaming_two_tool_requests_increment_index_and_id(parser, mock_request):
    """Verifies that moving to the second tool_request increments index and
    issues a new id (streaming-side coverage of the MAYBE_NEXT_TOOL_OR_END
    transition). Also verifies that each tool_call arguments are correctly
    accumulated and restored.

    Regression test: END_TOOL_ARGS advancement previously added the tag length
    to the argument *start* position, so the parser stalled in END_TOOL_REQUEST
    and every tool call after the first was dropped from streams.
    """
    full_text = (
        BEGIN_TOOL_REQUESTS_TAG
        + _wrap_single_tool_call("echo", json.dumps({"text": "hi"}))
        + _wrap_single_tool_call("sum", json.dumps({"a": 1, "b": 2}))
        + END_TOOL_REQUESTS_TAG
        + EOT_TAG
    )

    all_deltas = []
    prev = ""
    for chunk in _iter_tokens(full_text):
        cur = prev + chunk
        msg = _call_streaming(parser, prev, cur, chunk, mock_request)
        if msg is not None and msg.tool_calls:
            all_deltas.extend(msg.tool_calls)
        prev = cur

    # Accumulate name / id / arguments per index
    name_by_index: dict[int, str] = {}
    id_by_index: dict[int, str] = {}
    args_by_index: dict[int, str] = {}
    for d in all_deltas:
        idx = d.index
        if d.function and d.function.name:
            name_by_index[idx] = d.function.name
            id_by_index[idx] = d.id
        if d.function and d.function.arguments:
            args_by_index[idx] = args_by_index.get(idx, "") + d.function.arguments

    assert sorted(name_by_index) == [0, 1]
    assert name_by_index[0] == "echo"
    assert name_by_index[1] == "sum"
    assert all(i is not None for i in id_by_index.values())
    assert id_by_index[0] != id_by_index[1], (
        "a new id should be issued for the second tool_request"
    )
    # Arguments must be fully restored
    assert json.loads(args_by_index[0]) == {"text": "hi"}
    assert json.loads(args_by_index[1]) == {"a": 1, "b": 2}


@pytest.mark.parametrize(
    "content,tool_bodies,expected",
    [
        (
            "",
            _wrap_single_tool_call("sum", '{"a":1,"b":2}'),
            [("sum", {"a": 1, "b": 2})],
        ),
        (
            "Before tools.",
            _wrap_single_tool_call("echo", '{"text":"hello"}'),
            [("echo", {"text": "hello"})],
        ),
        (
            "",
            _wrap_single_tool_call("first", '{"value":1}')
            + _wrap_single_tool_call("second", '{"value":2}'),
            [("first", {"value": 1}), ("second", {"value": 2})],
        ),
    ],
)
def test_streaming_complete_batch_is_fully_emitted(
    tokenizer,
    mock_request,
    content,
    tool_bodies,
    expected,
):
    """A final multi-token batch must not leave parsed deltas buffered.

    Stop strings and speculative decoding can deliver a whole tool-call block
    in one delta; the last parse call is the only chance to emit it, so all
    parsed pieces must be batched into a single DeltaMessage.
    """
    parser = Plamo3ToolParser(tokenizer)
    model_output = content + _wrap_tool_requests(tool_bodies)

    msg = _call_streaming(parser, "", model_output, model_output, mock_request)

    assert msg is not None
    assert msg.content == (content or None)
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == len(expected)
    for index, (tool_call, (name, arguments)) in enumerate(
        zip(msg.tool_calls, expected)
    ):
        assert tool_call.index == index
        assert tool_call.id is not None
        assert tool_call.function is not None
        assert tool_call.function.name == name
        assert json.loads(tool_call.function.arguments or "") == arguments


# ---------------------------------------------------------------------------
# Truncated special-token markers must never leak into non-streaming content.
#
# When max_tokens cuts generation mid-marker, the model can emit just the single
# ``<|plamo:begin_`` anchor token (id 256) before a length stop. The
# non-streaming parsers then find no complete tag and would pass that partial
# marker through as content. ``strip_trailing_partial_marker`` removes such
# truncation artifacts; this section pins the helper and its wiring into the
# reasoning + tool parsers.
# ---------------------------------------------------------------------------

_PLAMO_PREFIX = "<|plamo:"


# Alias so the truncation tests can request either name.
@pytest.fixture
def request_stub(mock_request):
    return mock_request


# --- Helper unit tests (pure, no tokenizer) -------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # The lone anchor token (the actual max_tokens=1 artifact).
        ("<|plamo:begin_", ""),
        # Partial begin-think between the anchor and the full tag.
        ("<|plamo:begin_think:pla", ""),
        # Partial end / EOT fragments.
        ("<|plamo:end_", ""),
        ("<|plamo:ta", ""),  # proper prefix of "<|plamo:tag|>"
        # Just the shared prefix.
        ("<|plamo:", ""),
        # Partial marker after real content: only the artifact is dropped.
        ("Hello<|plamo:begin_", "Hello"),
        ("answer text<|plamo:begin_tool_requests:pla", "answer text"),
    ],
)
def test_strips_trailing_partial_marker(text, expected):
    assert strip_trailing_partial_marker(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello world",
        "x < y and a <| b",  # stray '<' / '<|' are NOT the special prefix
        "ends with <|plamo",  # shorter than the full "<|plamo:" discriminator
    ],
)
def test_leaves_non_partial_text_unchanged(text):
    assert strip_trailing_partial_marker(text) == text


def test_special_tags_are_prefix_free():
    # strip_trailing_partial_marker's "incomplete vs complete" test (tail != tag)
    # is only sound if no tag is a proper prefix of another. The module enforces
    # this at import; assert it here too so the invariant is visible and covered.
    for a in _ALL_SPECIAL_TAGS:
        for b in _ALL_SPECIAL_TAGS:
            if a != b:
                assert not b.startswith(a), f"{a!r} is a proper prefix of {b!r}"


def test_whole_partial_marker_strips_to_empty_string_not_none():
    # The no-think path returns a *string* (possibly "") for content, matching
    # the pre-fix contract — it must not start mapping "" to None.
    assert strip_trailing_partial_marker("<|plamo:begin_") == ""


@pytest.mark.parametrize(
    "text",
    [
        # A *complete* tag is only a truncation artifact's opposite: it must be
        # left intact (tail == tag is excluded; no atomic tag is a proper prefix
        # of another, so a complete tag is never mistaken for an incomplete one).
        BEGIN_TOOL_REQUESTS_TAG,
        END_THINK_TAG,
        EOT_TAG,
        # Complete tag at the end of real text.
        f"answer{EOT_TAG}",
        f"text{BEGIN_TOOL_REQUESTS_TAG}",
        # Complete tag followed by more text (not a trailing fragment at all).
        f"a{BEGIN_THINK_TAG}b",
    ],
)
def test_complete_tags_are_preserved(text):
    assert strip_trailing_partial_marker(text) == text


# --- Parser wiring --------------------------------------------------------


def test_tool_parser_drops_lone_partial_marker(tokenizer, request_stub):
    parser = Plamo3ToolParser(tokenizer)
    result = parser.extract_tool_calls("<|plamo:begin_", request_stub)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert _PLAMO_PREFIX not in (result.content or "")
    assert (result.content or "") == ""


def test_reasoning_parser_drops_lone_partial_begin(tokenizer, request_stub):
    parser = Plamo3ReasoningParser(
        tokenizer, chat_template_kwargs={"enable_thinking": False}
    )
    reasoning, content = parser.extract_reasoning("<|plamo:begin_", request_stub)
    assert reasoning is None
    # When thinking is off, IdentityReasoningParser is used: reasoning
    # is not separated and the model output passes through as content
    # verbatim (no partial-marker strip).
    assert content == "<|plamo:begin_"


def test_reasoning_parser_empty_content_after_think_is_empty_string(
    tokenizer, request_stub
):
    # END_THINK with nothing after -> content is "" (a string), not None.
    parser = Plamo3ReasoningParser(tokenizer)
    reasoning, content = parser.extract_reasoning(
        f"{BEGIN_THINK_TAG}x{END_THINK_TAG}", request_stub
    )
    assert reasoning == "x"
    assert content == ""


def test_reasoning_parser_strips_trailing_partial_after_content(
    tokenizer, request_stub
):
    # Full reasoning + content, then a truncated trailing marker.
    text = f"{BEGIN_THINK_TAG}thought{END_THINK_TAG}answer<|plamo:begin_"
    reasoning, content = _parser_extract(tokenizer, request_stub, text)
    assert reasoning == "thought"
    assert content == "answer"
    assert _PLAMO_PREFIX not in (content or "")


def _parser_extract(tokenizer, request_stub, text):
    return Plamo3ReasoningParser(tokenizer).extract_reasoning(text, request_stub)


def test_reasoning_parser_plain_content_preserved(tokenizer, request_stub):
    # No think block at all -> everything is content, untouched.
    parser = Plamo3ReasoningParser(
        tokenizer, chat_template_kwargs={"enable_thinking": False}
    )
    reasoning, content = parser.extract_reasoning(
        "just content, no markers", request_stub
    )
    assert reasoning is None
    assert content == "just content, no markers"


def test_reasoning_span_keeps_internal_complete_tag(tokenizer, request_stub):
    # A complete tag inside the reasoning span (between BEGIN/END_THINK) is
    # extracted verbatim — the strip only touches a trailing partial marker,
    # never the reasoning span.
    text = f"{BEGIN_THINK_TAG}before {EOT_TAG} after{END_THINK_TAG}answer"
    reasoning, content = Plamo3ReasoningParser(tokenizer).extract_reasoning(
        text, request_stub
    )
    assert reasoning == f"before {EOT_TAG} after"
    assert content == "answer"


def test_reasoning_truncated_mid_end_think_strips_from_reasoning(
    tokenizer, request_stub
):
    # Cut inside a partial END_THINK ("<|plamo:end_") with no full END tag:
    # the partial marker must not leak into the reasoning field.
    text = f"{BEGIN_THINK_TAG}thinking hard<|plamo:end_"
    reasoning, content = Plamo3ReasoningParser(tokenizer).extract_reasoning(
        text, request_stub
    )
    assert reasoning == "thinking hard"
    assert content is None
    assert _PLAMO_PREFIX not in (reasoning or "")


def _full_outputs():
    """Representative complete model outputs (without the trailing EOT, which
    vLLM removes as a stop token before the parsers run)."""
    tc = (
        f"{BEGIN_TOOL_REQUESTS_TAG}"
        f"<|plamo:begin_tool_request:plamo|>"
        f"<|plamo:begin_tool_name:plamo|>noop<|plamo:end_tool_name:plamo|>"
        f"<|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>{{}}<|plamo:end_tool_arguments:plamo|>"
        f"<|plamo:end_tool_request:plamo|>"
        f"<|plamo:end_tool_requests:plamo|>"
    )
    return {
        "reasoning+content": (
            f"{BEGIN_THINK_TAG}let me think{END_THINK_TAG}Here is the answer."
        ),
        "reasoning+toolcall": f"{BEGIN_THINK_TAG}deciding{END_THINK_TAG}{tc}",
        "noreason+content": "Hello there, this is content.",
        "noreason+toolcall": tc,
    }


@pytest.mark.parametrize("name", list(_full_outputs().keys()))
def test_no_marker_leaks_at_any_truncation_point(tokenizer, request_stub, name):
    """Exhaustive sweep: truncate each representative output at *every* byte
    boundary (a superset of token boundaries) and assert no "<|plamo:"
    special-token markup survives in reasoning, content, or tool_call.arguments.

    Each output is run through the pipeline that matches its model kind: the
    reasoning model serves with the reasoning parser in front of the tool
    parser; the non-reasoning model has no reasoning parser, so the tool parser
    sees the raw output directly."""
    rp = Plamo3ReasoningParser(tokenizer)
    tp = Plamo3ToolParser(tokenizer)
    full = _full_outputs()[name]
    is_reasoning = full.startswith(BEGIN_THINK_TAG)
    for i in range(1, len(full) + 1):
        prefix = full[:i]
        if is_reasoning:
            reasoning, content = rp.extract_reasoning(prefix, request_stub)
        else:
            # Non-reasoning model: no reasoning parser is loaded.
            reasoning, content = None, prefix
        res = tp.extract_tool_calls(content or "", request_stub)
        _assert_clean(reasoning, "reasoning", prefix)
        _assert_clean(res.content, "content", prefix)
        for k, tc in enumerate(res.tool_calls):
            _assert_clean(tc.function.arguments, f"args[{k}]", prefix)


def _assert_clean(value, label, prefix):
    v = value or ""
    assert _PLAMO_PREFIX not in v, (
        f"marker leaked into {label} when truncated at {prefix[-24:]!r}: {v!r}"
    )


def test_normal_tool_call_unaffected(tokenizer, request_stub):
    # Regression: a complete tool call still parses and content stays clean.
    text = (
        f"{BEGIN_TOOL_REQUESTS_TAG}"
        f"<|plamo:begin_tool_request:plamo|>"
        f"<|plamo:begin_tool_name:plamo|>noop<|plamo:end_tool_name:plamo|>"
        f"<|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>{{}}<|plamo:end_tool_arguments:plamo|>"
        f"<|plamo:end_tool_request:plamo|>"
        f"<|plamo:end_tool_requests:plamo|>{EOT_TAG}"
    )
    result = Plamo3ToolParser(tokenizer).extract_tool_calls(text, request_stub)
    assert result.tools_called is True
    assert [tc.function.name for tc in result.tool_calls] == ["noop"]
    assert (result.content or "") == ""
