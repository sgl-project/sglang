"""Unit tests for the per-id text cache in TokenizerManager.detokenize_logprob_tokens.

Turns red if a cached text ever differs from decoding that id on its own, or if
ids that were already decoded go through the tokenizer again.
"""

import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CountingTokenizer:
    """Decodes each single-id sequence to a distinct text and counts work."""

    def __init__(self):
        self.calls = 0
        self.decoded_ids = []

    def batch_decode(self, sequences):
        self.calls += 1
        texts = []
        for seq in sequences:
            assert len(seq) == 1, "each id must be decoded on its own"
            self.decoded_ids.append(int(seq[0]))
            texts.append(f"<t{int(seq[0])}>")
        return texts


class _TokenizerManagerStub:
    detokenize_logprob_tokens = TokenizerManager.detokenize_logprob_tokens
    detokenize_top_logprobs_tokens = TokenizerManager.detokenize_top_logprobs_tokens

    def __init__(self):
        self.tokenizer = _CountingTokenizer()
        self._logprob_token_text_cache = {}


def _uncached(vals, ids):
    return [(v, i, f"<t{int(i)}>") for v, i in zip(vals, ids)]


class TestDetokenizeLogprobTokens(CustomTestCase):
    def test_matches_per_id_decode(self):
        tm = _TokenizerManagerStub()
        vals = [-0.1, -0.2, -0.3, -0.4, -0.5]
        ids = [7, 3, 7, 9, 3]
        self.assertEqual(
            tm.detokenize_logprob_tokens(vals, ids, decode_to_text=True),
            _uncached(vals, ids),
        )

    def test_decodes_each_new_id_once(self):
        tm = _TokenizerManagerStub()
        tm.detokenize_logprob_tokens([-1.0] * 4, [5, 6, 5, 6], decode_to_text=True)
        self.assertEqual(tm.tokenizer.calls, 1)
        self.assertEqual(tm.tokenizer.decoded_ids, [5, 6])

        out = tm.detokenize_logprob_tokens([-2.0, -3.0], [6, 5], decode_to_text=True)
        self.assertEqual(out, _uncached([-2.0, -3.0], [6, 5]))
        self.assertEqual(tm.tokenizer.calls, 1)

        tm.detokenize_logprob_tokens([-1.0, -1.0], [5, 8], decode_to_text=True)
        self.assertEqual(tm.tokenizer.calls, 2)
        self.assertEqual(tm.tokenizer.decoded_ids, [5, 6, 8])

    def test_top_logprobs_rows_share_the_cache(self):
        tm = _TokenizerManagerStub()
        val_rows = [[-0.1, -2.0], None, [-0.3, -1.0], [-0.2, -4.0]]
        idx_rows = [[11, 22], None, [22, 11], [np.int64(11), np.int64(33)]]
        out = tm.detokenize_top_logprobs_tokens(val_rows, idx_rows, decode_to_text=True)
        self.assertEqual(
            out,
            [
                _uncached(val_rows[0], idx_rows[0]),
                None,
                _uncached(val_rows[2], idx_rows[2]),
                _uncached(val_rows[3], idx_rows[3]),
            ],
        )
        self.assertEqual(sorted(tm.tokenizer.decoded_ids), [11, 22, 33])

    def test_no_text_skips_tokenizer(self):
        tm = _TokenizerManagerStub()
        out = tm.detokenize_logprob_tokens([-0.5], [4], decode_to_text=False)
        self.assertEqual(out, [(-0.5, 4, None)])
        self.assertEqual(tm.tokenizer.calls, 0)


if __name__ == "__main__":
    unittest.main()
