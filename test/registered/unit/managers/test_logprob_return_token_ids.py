"""Unit tests for return_token_ids_in_logprobs / empty-idx logprob detokenization."""

import unittest

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class _FakeTokenizer:
    """Decode each single-id list to a deterministic ``tok<id>`` string."""

    def batch_decode(self, ids_lists):
        return [f"tok{ids[0]}" for ids in ids_lists]


class _TMStub:
    """Minimal stand-in for ``self`` (only the logprob detokenize paths run)."""

    detokenize_logprob_tokens = TokenizerManager.detokenize_logprob_tokens
    detokenize_top_logprobs_tokens = TokenizerManager.detokenize_top_logprobs_tokens

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer


class TestReturnTokenIdsInLogprobs(CustomTestCase):
    def setUp(self):
        self.tm = _TMStub()
        self.tm_text = _TMStub(tokenizer=_FakeTokenizer())

    # ---- flat (regular / per-request) path ------------------------------

    def test_flat_default_keeps_triplets(self):
        """Default return_token_ids=True keeps [logprob, token_id, None].

        Also guards the regression where removing the default value made callers
        that omit the argument raise TypeError.
        """
        out = self.tm.detokenize_logprob_tokens([-1.0, -2.0], [10, 20], False)
        self.assertEqual(out, [(-1.0, 10, None), (-2.0, 20, None)])

    def test_flat_drop_token_ids_returns_scalars(self):
        """return_token_ids=False, no text -> raw logprob scalars."""
        out = self.tm.detokenize_logprob_tokens(
            [-1.0, -2.0], [10, 20], False, return_token_ids=False
        )
        self.assertEqual(out, [-1.0, -2.0])

    def test_flat_text_with_token_ids(self):
        """decode_to_text=True, return_token_ids=True -> (logprob, id, text)."""
        out = self.tm_text.detokenize_logprob_tokens([-1.0, -2.0], [10, 20], True)
        self.assertEqual(out, [(-1.0, 10, "tok10"), (-2.0, 20, "tok20")])

    def test_flat_text_drop_token_ids(self):
        """decode_to_text=True, return_token_ids=False -> (logprob, text).

        The combination fixed in this change: text is preserved even though the
        token id is dropped.
        """
        out = self.tm_text.detokenize_logprob_tokens(
            [-1.0, -2.0], [10, 20], True, return_token_ids=False
        )
        self.assertEqual(out, [(-1.0, "tok10"), (-2.0, "tok20")])

    # ---- per-position (top / token_ids) path ----------------------------

    def test_top_default_keeps_triplets_with_none(self):
        """Per-position default keeps triplets; empty rows stay None."""
        val = [[], [-1.0, -2.0], [-3.0, -4.0]]
        idx = [[], [10, 20], [30, 40]]
        out = self.tm.detokenize_top_logprobs_tokens(val, idx, False)
        self.assertEqual(out[0], None)
        self.assertEqual(out[1], [(-1.0, 10, None), (-2.0, 20, None)])
        self.assertEqual(out[2], [(-3.0, 30, None), (-4.0, 40, None)])

    def test_top_drop_token_ids_returns_scalar_rows(self):
        """Per-position return_token_ids=False -> each row is a flat scalar list."""
        val = [[], [-1.0, -2.0], [-3.0, -4.0]]
        idx = [[], [10, 20], [30, 40]]
        out = self.tm.detokenize_top_logprobs_tokens(
            val, idx, False, return_token_ids=False
        )
        self.assertEqual(out, [None, [-1.0, -2.0], [-3.0, -4.0]])

    def test_top_tolerates_empty_idx_after_source_drop(self):
        """idx dropped at the scheduler source -> empty idx list must not crash.

        With return_token_ids=False the idx is never needed, so a non-empty val
        paired with an empty idx still collapses to scalar rows. This mirrors
        ``output_streamer.accept`` appending ``[]`` for the dropped idx.
        """
        val = [[-1.0, -2.0, -3.0]]
        idx = []  # dropped in output_streamer.accept()
        out = self.tm.detokenize_top_logprobs_tokens(
            val, idx, False, return_token_ids=False
        )
        self.assertEqual(out, [[-1.0, -2.0, -3.0]])

    def test_top_text_drop_token_ids(self):
        """Per-position decode_to_text=True, return_token_ids=False -> (logprob, text)."""
        val = [[-1.0, -2.0]]
        idx = [[10, 20]]
        out = self.tm_text.detokenize_top_logprobs_tokens(
            val, idx, True, return_token_ids=False
        )
        self.assertEqual(out, [[(-1.0, "tok10"), (-2.0, "tok20")]])

    def test_scalar_values_preserved_exactly(self):
        """The scalar path must preserve the exact logprob values/order."""
        val = [[-13.047, -8.953, -9.672]]
        idx = [[1, 2, 3]]
        out = self.tm.detokenize_top_logprobs_tokens(
            val, idx, False, return_token_ids=False
        )
        self.assertEqual(out, [[-13.047, -8.953, -9.672]])


if __name__ == "__main__":
    unittest.main()
