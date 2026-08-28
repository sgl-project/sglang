"""Unit tests for NGRAM capacity > max_trie_depth validation (#36495).

Validates that ``check_server_args`` rejects NGRAM speculative-decoding
configs where ``speculative_ngram_capacity <= speculative_ngram_max_trie_depth``,
which triggers a crash in the C++ ``Trie::squeeze`` during the first request.
"""
import unittest

from sglang.srt.server_args import ServerArgs


class TestNgramCapacityValidation(unittest.TestCase):
    """Verify the NGRAM capacity-depth invariant in check_server_args."""

    BASE_KWARGS = dict(
        model_path="dummy",
        served_model_name="dummy",
        speculative_num_draft_tokens=4,
        chunked_prefill_size=8192,
        page_size=1,
    )

    def test_capacity_eq_depth_rejected(self):
        """capacity == max_trie_depth must raise AssertionError."""
        args = ServerArgs(
            speculative_algorithm="NGRAM",
            speculative_ngram_max_trie_depth=4,
            speculative_ngram_capacity=4,
            **self.BASE_KWARGS,
        )
        with self.assertRaises(AssertionError) as ctx:
            args.check_server_args()
        self.assertIn("capacity", str(ctx.exception))
        self.assertIn("max_trie_depth", str(ctx.exception))

    def test_capacity_lt_depth_rejected(self):
        """capacity < max_trie_depth must raise AssertionError."""
        args = ServerArgs(
            speculative_algorithm="NGRAM",
            speculative_ngram_max_trie_depth=10,
            speculative_ngram_capacity=5,
            **self.BASE_KWARGS,
        )
        with self.assertRaises(AssertionError) as ctx:
            args.check_server_args()
        self.assertIn("capacity", str(ctx.exception))

    def test_capacity_gt_depth_accepted(self):
        """capacity > max_trie_depth must pass validation."""
        args = ServerArgs(
            speculative_algorithm="NGRAM",
            speculative_ngram_max_trie_depth=4,
            speculative_ngram_capacity=8,
            **self.BASE_KWARGS,
        )
        args.check_server_args()

    def test_default_ngram_values_accepted(self):
        """Default values (capacity=10M, depth=18) must pass."""
        args = ServerArgs(
            speculative_algorithm="NGRAM",
            **self.BASE_KWARGS,
        )
        args.check_server_args()

    def test_non_ngram_unaffected(self):
        """Non-NGRAM configs must not be affected by the new check."""
        args = ServerArgs(**self.BASE_KWARGS)
        args.check_server_args()


if __name__ == "__main__":
    unittest.main()
