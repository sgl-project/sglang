"""
Unit tests for the XGrammar bridge attached to MistralCommon tokenizers.

XGrammar's from_huggingface path cannot read a MistralCommon tokenizer, so
patch_mistral_common_tokenizer attaches an init_xgrammar hook that builds
TokenizerInfo from the inner Tekkenizer's raw byte pieces. These tests drive that
hook with a stub Tekkenizer so they need no model download.

Usage:
    python -m pytest test_mistral_common_xgrammar.py -v
"""

import unittest

from sglang.srt.utils.hf_transformers.mistral_utils import (
    patch_mistral_common_tokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

VOCAB_SIZE = 300
NUM_SPECIAL = 8


class _StubTekkenizer:
    """Minimal stand-in for mistral_common's Tekkenizer."""

    def __init__(self, vocab_size=VOCAB_SIZE, num_special=NUM_SPECIAL):
        self._vocab_size = vocab_size
        self.num_special_tokens = num_special
        self._all_special_tokens = [
            {"rank": 0, "token_str": "<unk>", "is_control": True},
            {"rank": 1, "token_str": "<s>", "is_control": True},
            {"rank": 2, "token_str": "</s>", "is_control": True},
        ]

    def id_to_byte_piece(self, token_id, special_token_policy=None):
        if token_id < self.num_special_tokens:
            return b""
        # id == num_special is the NUL byte-fallback token, mirroring real Tekken.
        if token_id == self.num_special_tokens:
            return b"\x00"
        return bytes([token_id % 256])


class _StubMistralTokenizer:
    """Stands in for MistralCommonBackend: name must contain 'MistralCommon'."""

    def __init__(self, tekken=None):
        inner = type("InstructTokenizer", (), {"tokenizer": tekken})()
        self.tokenizer = type("MistralTokenizer", (), {"instruct_tokenizer": inner})()
        self.eos_token_id = 2
        self.chat_template = "x"

    def add_special_tokens(self, *args, **kwargs):
        return 0

    def convert_tokens_to_ids(self, val):
        return 0

    def decode(self, *args, **kwargs):
        return ""

    def batch_decode(self, *args, **kwargs):
        return []

    def apply_chat_template(self, *args, **kwargs):
        return []


class _MistralCommonStub(_StubMistralTokenizer):
    pass


class TestMistralCommonXGrammar(unittest.TestCase):
    def _patched(self, tekken=None):
        tok = _MistralCommonStub(tekken if tekken is not None else _StubTekkenizer())
        return patch_mistral_common_tokenizer(tok)

    def test_hook_is_attached(self):
        """The patch must expose init_xgrammar so XGrammar takes the special path."""
        self.assertTrue(hasattr(self._patched(), "init_xgrammar"))

    def test_builds_tokenizer_info_over_full_vocab(self):
        """TokenizerInfo must cover every id, not just the decodable ones."""
        info, stop_tokens = self._patched().init_xgrammar()

        self.assertIsNotNone(info)
        self.assertEqual(info.vocab_size, VOCAB_SIZE)
        self.assertEqual(stop_tokens, [2])

    def test_json_schema_compiles_and_constrains(self):
        """A compiled JSON grammar must allow '{' first and reject a letter."""
        from xgrammar import GrammarCompiler, GrammarMatcher, allocate_token_bitmask

        info, _ = self._patched().init_xgrammar()
        grammar = GrammarCompiler(tokenizer_info=info).compile_json_schema(
            '{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}'
        )
        mask = allocate_token_bitmask(1, info.vocab_size)
        GrammarMatcher(grammar).fill_next_token_bitmask(mask)

        def allowed(token_id):
            # XGrammar sets a bit when the token is permitted.
            return bool((int(mask[0][token_id // 32]) >> (token_id % 32)) & 1)

        # The stub maps id -> bytes([id % 256]), so a byte's token id is its ordinal.
        self.assertTrue(allowed(ord("{")))
        self.assertFalse(allowed(ord("z")))

    def test_returns_none_without_a_tekkenizer(self):
        """Missing inner tokenizer must degrade to the documented (None, None)."""
        info, stop_tokens = self._patched(tekken=object()).init_xgrammar()

        self.assertIsNone(info)
        self.assertIsNone(stop_tokens)


if __name__ == "__main__":
    unittest.main()
