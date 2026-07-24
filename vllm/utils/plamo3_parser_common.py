# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
# END_THINK_TAG tokenizes to these three token IDs in the fixed PLaMo-3 vocabulary:
#   <|plamo:end_  (257) + think (21279) + :plamo|> (258)
END_THINK_TOKEN_IDS: list[int] = [257, 21279, 258]

# Markdown-style JSON code fence emitted around content when the user supplies
# a JSON schema via structured_outputs.json. The grammar forces the model to
# produce these exact byte sequences around its JSON payload, and the tool
# parser strips them back off before exposing message.content.
JSON_FENCE_HEAD = "```json\n"
JSON_FENCE_TAIL = "\n```"


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

# strip_trailing_partial_marker() distinguishes an *incomplete* marker (a proper
# prefix of a tag) from a *complete* tag via the ``tail != tag`` test. That is
# only sound while no tag is a proper prefix of another: otherwise a complete
# short tag sitting at the tail would also satisfy ``longer_tag.startswith(tail)``
# with ``tail != longer_tag`` and be wrongly stripped as "incomplete" — silently
# eating a legitimate complete tag from content/reasoning. Enforce the
# prefix-free invariant at import (and with an explicit raise, so it holds under
# ``python -O`` too) rather than relying on a comment.
for _a in _ALL_SPECIAL_TAGS:
    for _b in _ALL_SPECIAL_TAGS:
        if _a != _b and _b.startswith(_a):
            raise RuntimeError(
                f"PLaMo special tag {_a!r} is a proper prefix of {_b!r}; "
                "strip_trailing_partial_marker would wrongly strip a complete "
                f"{_a!r}. Keep _ALL_SPECIAL_TAGS prefix-free."
            )
del _a, _b


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


# Mapping from token string to expected token ID. The grammar in
# plamo3_structured_output.py references these IDs via the lark/llguidance
# <[ID]> syntax (e.g. _B=<[256]>, _E=<[257]>, _C=<[258]>) to anchor
# PLaMo-3's multi-token begin/end markers to specific vocabulary positions.
# If the tokenizer assigns different IDs the generated grammar is silently
# wrong, so these are verified once at server startup.
_GRAMMAR_TOKEN_IDS: list[tuple[str, int]] = [
    ("<|plamo:begin_", 256),
    ("<|plamo:end_", 257),
    (":plamo|>", 258),
]


def verify_grammar_token_ids(tokenizer: object) -> None:
    """Verify each token string in ``_GRAMMAR_TOKEN_IDS`` encodes to the expected ID.

    The grammar uses the lark/llguidance ``<[ID]>`` syntax to reference
    specific tokens by their vocabulary index. If the tokenizer assigns
    different IDs to these tokens the generated grammar is silently wrong.

    Raises ``ValueError`` if any token encodes to an unexpected ID.
    Silently skips tokenizers that do not expose an ``encode`` method
    (e.g. dummy tokenizers used in unit tests).
    """
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return
    for token_str, expected_id in _GRAMMAR_TOKEN_IDS:
        ids = encode(token_str, add_special_tokens=False)
        if len(ids) != 1 or ids[0] != expected_id:
            raise ValueError(
                f"Token {token_str!r} must encode to [{expected_id}] for the grammar "
                f"to work correctly, but it encodes to {ids}. "
                f"The tokenizer or vocabulary may have changed."
            )


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
