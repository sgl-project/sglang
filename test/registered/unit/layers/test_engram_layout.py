"""Golden checks for the DeepSeek-V4.1 engram hash layout.

The compressed token map depends on the `tokenizers` normalizers and the prime
layout on the local primality test; both feed every hash multiplier, so a silent
change rehashes the whole table.
"""

import types
import unittest

import torch
from tokenizers import Tokenizer, models

from sglang.srt.layers.engram import (
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Tokenizer:
    """The two attributes build_compressed_token_map reads from an HF tokenizer."""

    def __init__(self, tokens):
        vocab = {tok: i for i, tok in enumerate(tokens)}
        self.backend_tokenizer = Tokenizer(models.WordLevel(vocab, unk_token=tokens[0]))
        self._size = len(tokens)

    def __len__(self):
        return self._size


# Case, accents, compatibility forms and whitespace runs collapse; distinct words stay.
_TOKENS = [
    "[UNK]",
    "Hello",
    "hello",
    "HELLO",
    "café",
    "café",
    "cafe",
    " ",
    "  ",
    "\t",
    "a b",
    "a  b",
    "ﬁ",
    "fi",
    "x",
]
_COMPRESSED = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6]


def _naive_is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


class TestEngramLayout(unittest.TestCase):
    def test_compressed_token_map(self):
        lookup, size = build_compressed_token_map(_Tokenizer(_TOKENS))
        self.assertEqual(lookup, _COMPRESSED)
        self.assertEqual(size, 7)

    def test_layout_primes(self):
        config = types.SimpleNamespace(
            engram_layer_ids=(1, 2),
            engram_num_embeddings=(64, 64),
            engram_max_ngram_size=3,
            engram_n_heads=2,
            engram_head_dim=8,
            engram_vocab_size=1000,
        )
        layout = EngramLayout.from_config(config)
        # (layer, n-gram size, head) order from one ascending sequence above 999.
        self.assertEqual(
            layout.primes,
            (((1009, 1013), (1019, 1021)), ((1031, 1033), (1039, 1049))),
        )
        flat = [p for layer in layout.primes for size in layer for p in size]
        self.assertTrue(all(_naive_is_prime(p) for p in flat))
        self.assertEqual(flat, sorted(set(flat)))
        self.assertIsNone(
            EngramLayout.from_config(types.SimpleNamespace(engram_layer_ids=()))
        )

    def test_hash_multipliers(self):
        multipliers = compute_hash_multipliers((1, 2), 3, 1000)
        self.assertEqual(multipliers.shape, (2, 3))
        self.assertTrue(bool((multipliers % 2 == 1).all()))
        bound = (torch.iinfo(torch.int64).max // 1000) // 2 * 2 + 1
        self.assertTrue(bool((multipliers < bound).all()))
        self.assertTrue(
            torch.equal(multipliers, compute_hash_multipliers((1, 2), 3, 1000))
        )


if __name__ == "__main__":
    unittest.main()
