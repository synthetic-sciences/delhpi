"""Unit tests for OpenAIEmbeddingProvider input sanitization.

The OpenAI tokenizer rejects tiktoken special-token literals (e.g.
``<|endoftext|>``) by default with a ``disallowed special token`` error,
and the REST API exposes no equivalent of the python client's
``disallowed_special=()`` flag. We sanitize input chunks at the provider
boundary; these tests pin that behavior.
"""

from __future__ import annotations

import pytest

# These tests exercise ``OpenAIEmbeddingProvider._sanitize_special_tokens``
# as a classmethod — no provider instance is constructed and no network/env
# state is touched.
from synsc.embeddings.providers import OpenAIEmbeddingProvider


@pytest.mark.parametrize(
    "literal",
    [
        "<|endoftext|>",
        "<|endofprompt|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|im_sep|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
    ],
)
def test_sanitize_replaces_known_special_token(literal: str) -> None:
    text = f"prefix {literal} suffix"
    out = OpenAIEmbeddingProvider._sanitize_special_tokens(text)
    assert literal not in out
    assert "[ANGLE_BRACKET_TOKEN_FILTERED]" in out
    assert out.startswith("prefix ")
    assert out.endswith(" suffix")


def test_sanitize_passthrough_when_no_special_token() -> None:
    text = "def foo():\n    return 42"
    assert OpenAIEmbeddingProvider._sanitize_special_tokens(text) == text


def test_sanitize_handles_multiple_occurrences() -> None:
    text = "<|endoftext|> middle <|endoftext|> end"
    out = OpenAIEmbeddingProvider._sanitize_special_tokens(text)
    assert "<|endoftext|>" not in out
    assert out.count("[ANGLE_BRACKET_TOKEN_FILTERED]") == 2


def test_sanitize_handles_empty_string() -> None:
    assert OpenAIEmbeddingProvider._sanitize_special_tokens("") == ""


def test_sanitize_leaves_unrelated_angle_brackets_alone() -> None:
    text = "vec<int> x; if (a < |b|) { return; }"
    assert OpenAIEmbeddingProvider._sanitize_special_tokens(text) == text
