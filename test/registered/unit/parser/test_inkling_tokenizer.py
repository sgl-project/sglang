"""Unit tests for srt/parser/inkling_tokenizer.py — no server, no model.

Covers the Inkling framing-token contracts:
- the special-token -> id overlay table (vocab-dictated external literals),
- the control-token alphabet shared by the reasoning parser and the
  tool-call detector (must be a superset of the framing tokens),
- normalize_special_token's dual spelling acceptance and failure mode,
- encode_text/encode_special keeping framing strictly in the overlay
  (the base tokenizer must never inject special tokens itself).
"""

import unittest
from unittest.mock import Mock

from sglang.srt.parser.inkling_tokenizer import (
    INKLING_CONTROL_TOKENS,
    INKLING_SPECIAL_TOKEN_IDS,
    INKLING_SPECIAL_TOKENS,
    MESSAGE_MODEL,
    MESSAGE_SYSTEM,
    MESSAGE_TOOL,
    MESSAGE_USER,
    ROLE_MESSAGE_TOKENS,
    InklingTokenizer,
    normalize_special_token,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestTokenTables(CustomTestCase):
    def test_special_token_ids_match_the_model_vocab(self):
        """External-source literals: these ids are dictated by the Inkling
        vocab, not by this repo — a silently mistyped id would produce
        corrupt framing that no other test catches."""
        self.assertEqual(
            INKLING_SPECIAL_TOKEN_IDS,
            {
                "<|endoftext|>": 199999,
                "<|message_user|>": 200000,
                "<|message_model|>": 200001,
                "<|message_system|>": 200002,
                "<|message_tool|>": 200003,
                "<|content_text|>": 200004,
                "<|content_image|>": 200005,
                "<|content_model_end_sampling|>": 200006,
                "<|content_thinking|>": 200008,
                "<|end_message|>": 200010,
                "<|content_audio_input|>": 200020,
                "<|content_tool_error|>": 200022,
                "<|content_xml|>": 200024,
                "<|audio_end|>": 200043,
                "<|content_invoke_tool_json|>": 200049,
                "<|content_invoke_tool_text|>": 200057,
            },
        )

    def test_control_alphabet_covers_all_framing_tokens(self):
        """Completeness contract: the reasoning parser and the tool-call
        detector key on INKLING_CONTROL_TOKENS — a framing token missing
        from it lets malformed headers slip through one of the two parsers.
        The alphabet is the framing set plus the two control-only tokens."""
        self.assertTrue(INKLING_CONTROL_TOKENS.issuperset(INKLING_SPECIAL_TOKENS))
        self.assertEqual(
            INKLING_CONTROL_TOKENS - INKLING_SPECIAL_TOKENS,
            {"<|content_invoke_tool|>", "<|model_trigger_generation|>"},
        )

    def test_every_chat_role_maps_to_a_framing_token(self):
        """Bookkeeping: adding a chat role without a message token (or vice
        versa) silently breaks conversation rendering."""
        self.assertEqual(
            ROLE_MESSAGE_TOKENS,
            {
                "user": MESSAGE_USER,
                "assistant": MESSAGE_MODEL,
                "system": MESSAGE_SYSTEM,
                "tool": MESSAGE_TOOL,
            },
        )


class TestNormalizeSpecialToken(CustomTestCase):
    def test_accepts_both_spellings(self):
        """Contract: callers may pass either the bare name or the delimited
        form; both normalize to the delimited token."""
        self.assertEqual(normalize_special_token("message_user"), MESSAGE_USER)
        self.assertEqual(normalize_special_token(MESSAGE_USER), MESSAGE_USER)

    def test_unknown_token_raises_with_the_offending_name(self):
        """Negative branch: an unknown token must fail loudly (naming the
        token) rather than being passed through as plain text."""
        with self.assertRaises(KeyError) as ctx:
            normalize_special_token("not_a_token")
        self.assertIn("not_a_token", str(ctx.exception))


class TestInklingTokenizer(CustomTestCase):
    def test_encode_text_never_lets_base_tokenizer_add_specials(self):
        """Derived property: framing must come only from the overlay map, so
        plain text is always encoded with add_special_tokens=False — a base
        tokenizer injecting its own BOS/EOS would corrupt the framing."""
        base = Mock()
        base.encode.return_value = [1, 2, 3]
        tokenizer = InklingTokenizer(tokenizer=base)
        self.assertEqual(tokenizer.encode_text("hi"), [1, 2, 3])
        base.encode.assert_called_once_with("hi", add_special_tokens=False)

    def test_encode_text_rejects_non_str(self):
        """Negative branch: token-id lists must not be double-encoded."""
        tokenizer = InklingTokenizer(tokenizer=Mock())
        with self.assertRaises(TypeError):
            tokenizer.encode_text([1, 2, 3])

    def test_encode_special_uses_override_map_when_provided(self):
        """Field contract: a per-model special_token_ids overlay takes
        precedence over the builtin table (models may remap framing ids)."""
        tokenizer = InklingTokenizer(
            tokenizer=Mock(), special_token_ids={MESSAGE_USER: 42}
        )
        self.assertEqual(tokenizer.encode_special("message_user"), 42)

    def test_encode_special_defaults_to_builtin_table(self):
        tokenizer = InklingTokenizer(tokenizer=Mock())
        self.assertEqual(
            tokenizer.encode_special(MESSAGE_USER),
            INKLING_SPECIAL_TOKEN_IDS[MESSAGE_USER],
        )


if __name__ == "__main__":
    unittest.main()
