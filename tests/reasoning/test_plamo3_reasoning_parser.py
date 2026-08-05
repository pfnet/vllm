# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.utils.plamo3_parser_common import BEGIN_THINK_TAG, END_THINK_TAG


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


class StreamingReasoningReconstructor:
    def __init__(self):
        self.reasoning: str | None = None
        self.other_content: str | None = None

    def append_delta(self, delta: DeltaMessage | None):
        if delta is None:
            return
        if delta.reasoning is not None:
            self.reasoning = (
                delta.reasoning
                if self.reasoning is None
                else self.reasoning + delta.reasoning
            )
        if delta.content is not None:
            self.other_content = (
                delta.content
                if self.other_content is None
                else self.other_content + delta.content
            )


def run_reasoning_extraction(
    reasoning_parser: ReasoningParser,
    model_output: list[str],
    request: ChatCompletionRequest | None = None,
    streaming: bool = False,
) -> tuple[str | None, str | None]:
    if streaming:
        reconstructor = run_reasoning_extraction_streaming(
            reasoning_parser, model_output, request
        )
        return reconstructor.reasoning, (reconstructor.other_content or None)
    else:
        req = request or ChatCompletionRequest(messages=[], model="test-model")
        return reasoning_parser.extract_reasoning("".join(model_output), req)


def run_reasoning_extraction_streaming(
    reasoning_parser: ReasoningParser,
    model_deltas: list[str],
    request: ChatCompletionRequest | None = None,
) -> StreamingReasoningReconstructor:
    reconstructor = StreamingReasoningReconstructor()
    previous_text = ""
    previous_tokens: list[int] = []
    for delta in model_deltas:
        tokens = reasoning_parser.model_tokenizer.tokenize(delta)
        token_ids = [
            reasoning_parser.vocab.get(t) for t in tokens if t in reasoning_parser.vocab
        ]
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_ids
        msg = reasoning_parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta,
            previous_tokens,
            current_tokens,
            token_ids,
        )
        if msg is not None:
            reconstructor.append_delta(msg)
        previous_text = current_text
        previous_tokens = current_tokens
    return reconstructor


@pytest.fixture
def tokenizer():
    return _DummyTokenizer()



PlamoReasoningParser = ReasoningParserManager.get_reasoning_parser("plamo3")


@pytest.fixture
def parser(tokenizer):
    return PlamoReasoningParser(tokenizer)


def test_non_streaming_basic_extraction(parser):
    # 先頭にBEGINタグ → 推論本文 → ENDタグ → 残りがcontent
    reasoning_text = "プランを検討中…"
    content_text = "最終回答です。"
    model_output = f"{BEGIN_THINK_TAG}{reasoning_text}{END_THINK_TAG}{content_text}"

    reasoning, content = parser.extract_reasoning(model_output, request=None)  # type: ignore[arg-type]
    assert reasoning == reasoning_text
    assert content == content_text


def test_reasoning_delimiter_properties(parser):
    assert parser.reasoning_start_str == BEGIN_THINK_TAG
    assert parser.reasoning_end_str == END_THINK_TAG


def test_non_streaming_no_begin_tag(parser):
    # 先頭にBEGINタグが無い場合は全体がcontent扱い
    model_output = "通常のテキスト（思考タグ無し）"
    reasoning, content = parser.extract_reasoning(model_output, request=None)  # type: ignore[arg-type]
    assert reasoning is None
    assert content == model_output


def test_streaming_simple_flow(tokenizer):
    parser = PlamoReasoningParser(tokenizer)
    deltas = [
        BEGIN_THINK_TAG,
        "考え",
        "を積み上げ",
        END_THINK_TAG,
        "ユーザーへの回答",
    ]

    reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)
    assert reasoning == "考えを積み上げ"
    assert content == "ユーザーへの回答"


def test_streaming_partial_begin_tag(tokenizer):
    parser = PlamoReasoningParser(tokenizer)
    # BEGINタグをわざと分割して徐々に渡す
    parts = [
        BEGIN_THINK_TAG[:10],
        BEGIN_THINK_TAG[10:20],
        BEGIN_THINK_TAG[20:],
        "最初の思考",
        END_THINK_TAG,
        "出力本文",
    ]
    recon = run_reasoning_extraction_streaming(parser, parts)
    assert recon.reasoning == "最初の思考"
    assert recon.other_content == "出力本文"


def test_extract_content_ids_returns_tokens_after_end_tag(parser):
    # vLLM uses these ids when switching from reasoning parsing to tool parsing.
    tokenizer = parser.model_tokenizer
    begin_ids = tokenizer.encode(BEGIN_THINK_TAG, add_special_tokens=False)
    end_ids = tokenizer.encode(END_THINK_TAG, add_special_tokens=False)
    assert parser.extract_content_ids(begin_ids + [42]) == []
    assert parser.extract_content_ids(begin_ids + [42] + end_ids + [99]) == [99]


def test_streaming_end_tag_fragmented_no_leak(tokenizer):
    parser = PlamoReasoningParser(tokenizer)
    # END_THINK_TAG の anchor "<|plamo:end_" は単一トークンなので分割されない。
    # anchor 以降の部分 "think:plamo|>" が複数チャンクで届くケースをテスト。
    et = END_THINK_TAG  # "<|plamo:end_think:plamo|>"
    anchor = "<|plamo:end_"
    rest = et[len(anchor) :]  # "think:plamo|>"
    parts = [
        BEGIN_THINK_TAG,
        "ABC",
        anchor,  # 単一トークンとして一括で届く
        rest[:5],  # "think"
        rest[5:],  # ":plamo|>"
        "OK",
    ]
    reasoning, content = run_reasoning_extraction(parser, parts, streaming=True)
    assert reasoning == "ABC"
    assert content == "OK"


def test_streaming_content_before_begin_tag(tokenizer):
    parser = PlamoReasoningParser(tokenizer)
    # BEGINより前に通常テキストがある場合、以降も含めて全てcontent扱い
    deltas = [
        "前置き",
        "さらに",
        BEGIN_THINK_TAG,
        "思考（無視されるはず）",
        END_THINK_TAG,
        "後半",
    ]
    reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)
    assert reasoning is None
    assert (
        content
        == "前置きさらに"
        + BEGIN_THINK_TAG
        + "思考（無視されるはず）"
        + END_THINK_TAG
        + "後半"
    )


def test_streaming_incomplete_no_end_tag(tokenizer):
    parser = PlamoReasoningParser(tokenizer)
    # ENDタグが来ないまま終了 → END_THINK_TAG の prefix で終わっていないので即 emit
    deltas = [BEGIN_THINK_TAG, "考えだけで終わる"]
    reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)
    assert reasoning == "考えだけで終わる"
    assert content is None


def test_non_streaming_only_end_tag(parser):
    # BEGINが無くENDのみ → 全体をcontentとして扱う
    model_output = END_THINK_TAG + "テキスト"
    reasoning, content = parser.extract_reasoning(model_output, request=None)  # type: ignore[arg-type]
    assert reasoning is None
    assert content == model_output


def test_manager_registration_by_name(tokenizer):
    # レジストリ経由で 'plamo3' 名称取得 → 基本抽出が動く
    parser_cls = ReasoningParserManager.get_reasoning_parser("plamo3")
    parser = parser_cls(tokenizer)
    reasoning, content = parser.extract_reasoning(
        BEGIN_THINK_TAG + "r" + END_THINK_TAG + "c",
        request=None,  # type: ignore[arg-type]
    )
    assert reasoning == "r"
    assert content == "c"


def test_delta_message_kinds(tokenizer):
    # reasoningとcontentのデルタが相互に混在しないこと
    parser = PlamoReasoningParser(tokenizer)

    # 1) BEGIN流す → None（タグ完了待ち） or reasoning空は返らない
    msg = parser.extract_reasoning_streaming(
        previous_text="",
        current_text=BEGIN_THINK_TAG,
        delta_text=BEGIN_THINK_TAG,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
    )
    assert msg is None

    # 2) reasoning文字列を流す → END_THINK_TAG の prefix で終わっていないので即 emit
    msg = parser.extract_reasoning_streaming(
        previous_text=BEGIN_THINK_TAG,
        current_text=BEGIN_THINK_TAG + "abc",
        delta_text="abc",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
    )
    assert msg is not None and msg.reasoning == "abc" and msg.content is None

    # 3) ENDを流す → "abc" は step 2 で emit 済みなので追加 reasoning delta はなし
    msg = parser.extract_reasoning_streaming(
        previous_text=BEGIN_THINK_TAG + "abc",
        current_text=BEGIN_THINK_TAG + "abc" + END_THINK_TAG,
        delta_text=END_THINK_TAG,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
    )
    assert msg is None

    # 4) content文字を流す → contentデルタ
    msg = parser.extract_reasoning_streaming(
        previous_text=BEGIN_THINK_TAG + "abc" + END_THINK_TAG,
        current_text=BEGIN_THINK_TAG + "abc" + END_THINK_TAG + "X",
        delta_text="X",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
    )
    assert msg is not None and msg.content == "X" and msg.reasoning is None


# ---------------------------------------------------------------------------
# アンカーベース safe_until に関するテスト
# ---------------------------------------------------------------------------


def _call_reasoning_streaming(parser, previous_text, current_text, delta_text):
    """extract_reasoning_streaming のラッパー（token_ids は空リスト固定）。"""
    return parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
    )


def test_reasoning_content_emits_without_end_tag_when_no_prefix(tokenizer):
    """END_THINK_TAG の prefix で終わっていない reasoning は即座に emit されるべき。

    skip_special_tokens=True 等で EOT が来ない場合でも reasoning が失われてはならない。
    """
    parser = PlamoReasoningParser(tokenizer)
    # BEGIN タグを送って IN_REASONING フェーズに遷移
    _call_reasoning_streaming(parser, "", BEGIN_THINK_TAG, BEGIN_THINK_TAG)

    content = "思考テキスト"  # END_THINK_TAG の prefix で終わっていない
    prev = BEGIN_THINK_TAG
    cur = prev + content
    msg = _call_reasoning_streaming(parser, prev, cur, content)

    assert msg is not None, (
        "END_THINK_TAG prefix で終わっていない reasoning は即 emit されるべき"
    )
    assert msg.reasoning == content


def test_reasoning_chunks_emitted_per_chunk(tokenizer):
    """END_THINK_TAG prefix で終わっていない各チャンクは逐次 emit される。"""
    parser = PlamoReasoningParser(tokenizer)
    _call_reasoning_streaming(parser, "", BEGIN_THINK_TAG, BEGIN_THINK_TAG)

    chunks = ["AB", "CD", "EF"]
    prev = BEGIN_THINK_TAG
    emitted = []
    for chunk in chunks:
        cur = prev + chunk
        msg = _call_reasoning_streaming(parser, prev, cur, chunk)
        if msg is not None and msg.reasoning:
            emitted.append(msg.reasoning)
        prev = cur

    assert emitted == chunks, f"各チャンクが即 emit されるべき: {emitted}"


def test_reasoning_end_tag_prefix_held(tokenizer):
    """END_THINK_TAG の anchor prefix で始まる断片は確定まで保留される。

    "<|plamo:end_" は単一トークンなので途中状態では現れない。
    バッファ末尾が "<|plamo:end_" + 後続の一部で終わる場合、その部分は保留される。
    """
    parser = PlamoReasoningParser(tokenizer)
    _call_reasoning_streaming(parser, "", BEGIN_THINK_TAG, BEGIN_THINK_TAG)

    anchor = "<|plamo:end_"
    # anchor + END_THINK_TAG の続きの一部
    partial_suffix = END_THINK_TAG[len(anchor) : len(anchor) + 5]  # "think"
    reasoning_before = "思考"
    delta = reasoning_before + anchor + partial_suffix

    prev = BEGIN_THINK_TAG
    cur = prev + delta
    msg = _call_reasoning_streaming(parser, prev, cur, delta)

    # "思考" は即 emit、"<|plamo:end_think" は保留
    assert msg is not None and msg.reasoning == reasoning_before


def test_non_streaming_incomplete_reasoning(parser):
    # ENDタグ無しの場合は reasoning として返す
    reasoning_text = "途中で終わった思考"
    model_output = f"{BEGIN_THINK_TAG}{reasoning_text}"
    reasoning, content = parser.extract_reasoning(model_output, request=None)  # type: ignore[arg-type]
    assert reasoning == reasoning_text
    assert content is None


def test_reasoning_end_tag_prefix_then_full_tag(tokenizer):
    """END_THINK_TAG の prefix 保留後、完全なタグが来ると reasoning が確定する。"""
    parser = PlamoReasoningParser(tokenizer)
    _call_reasoning_streaming(parser, "", BEGIN_THINK_TAG, BEGIN_THINK_TAG)

    anchor = "<|plamo:end_"
    partial_suffix = END_THINK_TAG[len(anchor) : len(anchor) + 5]  # "think"
    reasoning_before = "思考"
    delta1 = reasoning_before + anchor + partial_suffix

    prev = BEGIN_THINK_TAG
    cur1 = prev + delta1
    _call_reasoning_streaming(parser, prev, cur1, delta1)

    # END_THINK_TAG の残りを送る
    rest_of_tag = END_THINK_TAG[len(anchor) + len(partial_suffix) :]
    cur2 = cur1 + rest_of_tag
    msg2 = _call_reasoning_streaming(parser, cur1, cur2, rest_of_tag)

    # タグ確定時に保留していた reasoning がフラッシュされる（または None）
    # END_THINK_TAG 直前の保留分 "<|plamo:end_think" は reasoning には含まれない
    # → このチャンクで reasoning delta は None（タグ部分はスキップ）
    assert msg2 is None or (msg2.reasoning is None or msg2.reasoning == "")


def test_is_reasoning_end_detects_non_reasoning_and_end_tag(tokenizer):
    """is_reasoning_end は生成トークン id から判定する。"""
    parser = PlamoReasoningParser(tokenizer)
    begin_ids = tokenizer.encode(BEGIN_THINK_TAG, add_special_tokens=False)
    end_ids = tokenizer.encode(END_THINK_TAG, add_special_tokens=False)

    # vLLM also calls is_reasoning_end() on prompt token ids. A normal prompt
    # without think tags must not bypass the streaming reasoning parser.
    assert parser.is_reasoning_end([]) is False
    assert parser.is_reasoning_end([100, 200, 300]) is False

    # A begin-think prefix without END_THINK is still inside reasoning.
    assert parser.is_reasoning_end(begin_ids[:1]) is False
    assert parser.is_reasoning_end(begin_ids + [42]) is False

    # Once END_THINK appears, the reasoning phase is complete.
    assert parser.is_reasoning_end(begin_ids + [42] + end_ids) is True
    # Prompts that explicitly end thinking (enable_thinking=False) still
    # short-circuit reasoning extraction.
    assert parser.is_reasoning_end([100] + end_ids) is True
    # END_THINK present without BEGIN_THINK => reasoning is complete regardless
    # of END_THINK position (matches vLLM BaseThinkingReasoningParser semantics:
    # the most recent end/begin marker wins).
    assert parser.is_reasoning_end([100] + end_ids + [42]) is True

    # is_reasoning_end_streaming scans only the delta window; a delta without a
    # completed END_THINK is not the end of reasoning.
    assert parser.is_reasoning_end_streaming([100], [100]) is False
    assert parser.is_reasoning_end_streaming(begin_ids + [42], [42]) is False
    # A leading BOS token must not be mistaken for the end of reasoning.
    bos_id = tokenizer.bos_token_id
    assert bos_id is not None
    assert parser.is_reasoning_end_streaming([bos_id] + begin_ids + [42], [42]) is False

    # Once streaming has started, END_THINK anywhere in the delta ends
    # reasoning even if content follows, because vLLM 0.20.x calls
    # is_reasoning_end on delta_token_ids only.
    parser2 = PlamoReasoningParser(tokenizer)
    _call_reasoning_streaming(parser2, "", BEGIN_THINK_TAG, BEGIN_THINK_TAG)
    assert parser2.is_reasoning_end([100] + end_ids + [42]) is True


def test_is_reasoning_end_streaming_with_three_token_tag(tokenizer):
    """END_THINK_TAG が3トークン構成の場合、is_reasoning_end_streaming が境界をまたいで検出できる。

    vLLM は should_advance 呼び出し時に input_ids = all_token_ids（delta を末尾に含む）、
    delta_ids = all_token_ids[num_computed_tokens:] を渡す。
    つまり input_ids[-len(delta_ids):] == delta_ids が成立する。
    """
    parser = PlamoReasoningParser(tokenizer)
    begin_ids = tokenizer.encode(BEGIN_THINK_TAG, add_special_tokens=False)
    end_ids = tokenizer.encode(END_THINK_TAG, add_special_tokens=False)
    assert len(end_ids) == 3, "END_THINK_TAG must tokenize to three ids for this test"
    end0, end1, end2 = end_ids

    # Three one-token steps
    # step1: first END_THINK token -> END_THINK not complete yet
    input_ids1 = [*begin_ids, 42, end0]
    assert parser.is_reasoning_end_streaming(input_ids1, input_ids1[-1:]) is False

    # step2: second END_THINK token -> still incomplete
    input_ids2 = [*begin_ids, 42, end0, end1]
    assert parser.is_reasoning_end_streaming(input_ids2, input_ids2[-1:]) is False

    # step3: third END_THINK token -> END_THINK complete
    input_ids3 = [*begin_ids, 42, end0, end1, end2]
    assert parser.is_reasoning_end_streaming(input_ids3, input_ids3[-1:]) is True

    # All three END_THINK tokens generated at once
    input_ids4 = [*begin_ids, 42, end0, end1, end2]
    assert parser.is_reasoning_end_streaming(input_ids4, input_ids4[-3:]) is True

    # Only the first two END_THINK tokens have arrived
    input_ids5 = [*begin_ids, 42, end0, end1]
    assert parser.is_reasoning_end_streaming(input_ids5, input_ids5[-2:]) is False


def test_is_reasoning_end_streaming_no_redetect_after_end(tokenizer):
    """END_THINK がデルタウィンドウより前に完全出現済みの場合は False を返す。

    ウィンドウ計算が「過去に確定した END_THINK」を再検出しないことを確認する。
    """
    parser = PlamoReasoningParser(tokenizer)
    begin_ids = tokenizer.encode(BEGIN_THINK_TAG, add_special_tokens=False)
    end_ids = tokenizer.encode(END_THINK_TAG, add_special_tokens=False)

    # END_THINK is in the past, delta is only the trailing token 99.
    # window = input_ids[-3:] excludes the complete END_THINK sequence.
    input_ids = [*begin_ids, 42, *end_ids, 99]
    assert parser.is_reasoning_end_streaming(input_ids, input_ids[-1:]) is False

    # Same check when END_THINK is further in the past.
    input_ids2 = [*begin_ids, 42, *end_ids, 10, 20, 30, 40]
    assert parser.is_reasoning_end_streaming(input_ids2, input_ids2[-1:]) is False


def test_streaming_content_not_lost_at_reasoning_end(tokenizer):
    """END_THINK 後の content がデルタ境界によらず欠損しないことを確認する。

    END_THINK とその後の content が同一デルタで届いた場合、その content は同じ
    DeltaMessage で（pending だった reasoning があればそれと一緒に）emit される。
    content を次のコールに遅延させると、vLLM の named tool_choice streaming 経路では
    is_reasoning_end が累積トークン列で reasoning 終了を検知して parser をバイパスするため、
    遅延した content が回収されず失われる
    （tests/tool_parser/test_named_tool_reasoning_content_loss.py 参照）。
    """
    # Case A: END_THINK と content が別デルタ（典型的なトークン毎ストリーミング）
    parser = PlamoReasoningParser(tokenizer)
    _call_reasoning_streaming(parser, "", BEGIN_THINK_TAG, BEGIN_THINK_TAG)
    _call_reasoning_streaming(parser, BEGIN_THINK_TAG, BEGIN_THINK_TAG + "思考", "思考")
    msg = _call_reasoning_streaming(
        parser,
        BEGIN_THINK_TAG + "思考",
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG,
        END_THINK_TAG,
    )
    assert msg is None
    msg = _call_reasoning_streaming(
        parser,
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG,
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG + "回答",
        "回答",
    )
    assert msg is not None and msg.content == "回答"

    # Case B: END_THINK + content in the same delta. Reasoning was already
    # emitted earlier, so this call emits the trailing content immediately in
    # the same DeltaMessage (same-delta convention).
    parser2 = PlamoReasoningParser(tokenizer)
    _call_reasoning_streaming(parser2, "", BEGIN_THINK_TAG, BEGIN_THINK_TAG)
    _call_reasoning_streaming(
        parser2, BEGIN_THINK_TAG, BEGIN_THINK_TAG + "思考", "思考"
    )
    msg = _call_reasoning_streaming(
        parser2,
        BEGIN_THINK_TAG + "思考",
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG + "回答",
        END_THINK_TAG + "回答",
    )
    assert msg is not None and msg.content == "回答"
    # Content was already emitted above; the follow-up call yields nothing.
    msg = _call_reasoning_streaming(
        parser2,
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG,
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG + "回答",
        "回答",
    )
    assert msg is None

    # Case C: reasoning + END_THINK in one delta, content in the next delta.
    parser3 = PlamoReasoningParser(tokenizer)
    _call_reasoning_streaming(parser3, "", BEGIN_THINK_TAG, BEGIN_THINK_TAG)
    msg = _call_reasoning_streaming(
        parser3,
        BEGIN_THINK_TAG,
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG,
        "思考" + END_THINK_TAG,
    )
    assert msg is not None and msg.reasoning == "思考"
    msg = _call_reasoning_streaming(
        parser3,
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG,
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG + "回答",
        "回答",
    )
    assert msg is not None and msg.content == "回答"

    # Case D: reasoning + END_THINK + content all in the same delta. Reasoning
    # and content are emitted together in one DeltaMessage (same-delta).
    parser4 = PlamoReasoningParser(tokenizer)
    _call_reasoning_streaming(parser4, "", BEGIN_THINK_TAG, BEGIN_THINK_TAG)
    msg = _call_reasoning_streaming(
        parser4,
        BEGIN_THINK_TAG,
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG + "回答",
        "思考" + END_THINK_TAG + "回答",
    )
    assert msg is not None and msg.reasoning == "思考" and msg.content == "回答"
    # Everything was already emitted above; the follow-up call yields nothing.
    msg = _call_reasoning_streaming(
        parser4,
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG + "回答",
        BEGIN_THINK_TAG + "思考" + END_THINK_TAG + "回答",
        "",
    )
    assert msg is None


def test_init_rejects_empty_think_tag_tokenization():
    class EmptyThinkTagTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {}

        def tokenize(self, text: str) -> list[str]:
            return [text] if text else []

        def convert_tokens_to_string(self, tokens: list[str]) -> str:
            return "".join(tokens)

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if text == "<|plamo:begin_":
                return [256]
            if text == "<|plamo:end_":
                return [257]
            if text == "<|plamo:tag|>":
                return [9]
            return []

    with pytest.raises(ValueError, match="failed to tokenize think tags"):
        PlamoReasoningParser(EmptyThinkTagTokenizer())
def _make_parser(tokenizer, chat_template_kwargs):
    return PlamoReasoningParser(tokenizer, chat_template_kwargs=chat_template_kwargs)


# When enable_thinking=True the chat template injects the begin-think tag, so
# the parser starts in the reasoning phase without requiring the model to emit
# it. Cover the streaming/non-streaming and begin/end-tag matrix.
@pytest.mark.parametrize(
    ("streaming", "begin_tag", "end_tag"),
    [
        (False, False, True),
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, False, False),
        (True, True, True),
    ],
    ids=[
        "non_stream-no_begin-end",
        "non_stream-no_begin-no_end",
        "non_stream-begin-end",
        "stream-no_begin-end",
        "stream-no_begin-no_end",
        "stream-begin-end",
    ],
)
def test_enable_thinking_extraction(tokenizer, streaming, begin_tag, end_tag):
    parser = _make_parser(tokenizer, {"enable_thinking": True})
    reasoning_text = "プランを検討中…"
    content_text = "最終回答です。"

    begin = BEGIN_THINK_TAG if begin_tag else ""
    end = END_THINK_TAG if end_tag else ""
    body = f"{begin}{reasoning_text}{end}{content_text}"

    # END tag absent => everything is reasoning, content is None.
    expected_reasoning = reasoning_text if end_tag else body
    expected_content = content_text if end_tag else None

    if streaming:
        deltas = []
        if begin_tag:
            deltas.append(BEGIN_THINK_TAG)
        deltas.append(reasoning_text)
        if end_tag:
            deltas.append(END_THINK_TAG)
        deltas.append(content_text)
        reasoning, content = run_reasoning_extraction(parser, deltas, streaming=True)
    else:
        reasoning, content = parser.extract_reasoning(body, request=None)  # type: ignore[arg-type]

    assert reasoning == expected_reasoning
    assert content == expected_content
