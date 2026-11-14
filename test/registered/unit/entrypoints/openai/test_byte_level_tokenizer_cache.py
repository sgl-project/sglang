"""Unit tests for the byte-level tokenizer verdict cache used by logprob bytes.

``_is_byte_level_tokenizer`` runs once per streamed chunk and again for every
logprob token (``token_id_to_bytes``). Its slow path calls ``get_vocab()``, which
rebuilds the full vocab dict. Only True verdicts used to be cached, so a
tokenizer that probes False (no boolean ``is_byte_level``, non-byte-level
pieces) re-ran the probe on every call: a 4000-token ``logprobs`` stream spent
~80 s of the tokenizer-manager event loop inside ``get_vocab()`` and stalled
every other response on the server. The verdict must be decided once per
tokenizer, whichever way it goes.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest

from sglang.srt.entrypoints.openai import utils as openai_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ProbeCountingTokenizer:
    """Just enough tokenizer for the slow path: no ``is_byte_level`` attribute,
    a vocab, and ``convert_ids_to_tokens``. Counts ``get_vocab()`` calls."""

    def __init__(self, pieces):
        self._pieces = pieces
        self.get_vocab_calls = 0

    def get_vocab(self):
        self.get_vocab_calls += 1
        return {p: i for i, p in enumerate(self._pieces)}

    def convert_ids_to_tokens(self, token_id):
        return self._pieces[token_id]


# SentencePiece-style pieces: "▁" (U+2581) is not in the GPT-2 byte decoder.
_SENTENCEPIECE = ["<unk>", "<s>", "</s>", "▁the", "▁é", "▁world", "ing", "▁a"]
# GPT-2 byte-level pieces: "Ġ" (U+0120) is the byte decoder's image of a space.
_BYTE_LEVEL = ["!", '"', "#", "Ġthe", "Ġworld", "ing", "Ġa", "Ã©"]


class TestByteLevelVerdictCache(CustomTestCase):
    def setUp(self):
        openai_utils._BYTE_LEVEL_VERDICT.clear()

    def test_negative_verdict_is_probed_once(self):
        tok = _ProbeCountingTokenizer(_SENTENCEPIECE)
        for _ in range(1000):
            self.assertFalse(openai_utils._is_byte_level_tokenizer(tok))
        self.assertEqual(tok.get_vocab_calls, 1)

    def test_positive_verdict_is_probed_once(self):
        tok = _ProbeCountingTokenizer(_BYTE_LEVEL)
        for _ in range(1000):
            self.assertTrue(openai_utils._is_byte_level_tokenizer(tok))
        self.assertEqual(tok.get_vocab_calls, 1)

    def test_token_id_to_bytes_does_not_reprobe_per_token(self):
        """The per-token call site: a non-byte-level tokenizer returns None (so
        callers fall back to the display text) without re-reading the vocab."""
        tok = _ProbeCountingTokenizer(_SENTENCEPIECE)
        for _ in range(500):
            self.assertIsNone(openai_utils.token_id_to_bytes(tok, 4))
        self.assertEqual(tok.get_vocab_calls, 1)

    def test_fast_path_attribute_is_respected_and_cached(self):
        class _Fast(_ProbeCountingTokenizer):
            is_byte_level = False

        tok = _Fast(_BYTE_LEVEL)  # pieces look byte-level; the attribute wins
        self.assertFalse(openai_utils._is_byte_level_tokenizer(tok))
        self.assertFalse(openai_utils._is_byte_level_tokenizer(tok))
        self.assertEqual(tok.get_vocab_calls, 0)

    def test_distinct_tokenizers_get_their_own_verdicts(self):
        sp = _ProbeCountingTokenizer(_SENTENCEPIECE)
        bl = _ProbeCountingTokenizer(_BYTE_LEVEL)
        self.assertFalse(openai_utils._is_byte_level_tokenizer(sp))
        self.assertTrue(openai_utils._is_byte_level_tokenizer(bl))
        self.assertFalse(openai_utils._is_byte_level_tokenizer(sp))
        self.assertEqual((sp.get_vocab_calls, bl.get_vocab_calls), (1, 1))


if __name__ == "__main__":
    unittest.main()
