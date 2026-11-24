"""Unit tests for the one-batch KV-capacity skip with ragged --input-len-list batches."""

import unittest

from sglang.benchmark.one_batch_server import should_skip_due_to_token_capacity
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRaggedTokenCapacity(CustomTestCase):
    def test_uniform_batch_counts_the_total(self):
        self.assertFalse(should_skip_due_to_token_capacity(4, 100, 10, 440))
        self.assertTrue(should_skip_due_to_token_capacity(4, 100, 10, 439))

    def test_ragged_batch_sums_its_lengths(self):
        self.assertFalse(should_skip_due_to_token_capacity(4, 0, 10, 440, "50,150"))
        self.assertTrue(should_skip_due_to_token_capacity(4, 0, 10, 439, "50,150"))

    def test_heaviest_dp_rank_must_fit_its_share(self):
        # Round-robin over 2 ranks: rank 0 holds 300 + 300, rank 1 holds 100 + 100.
        lens = "300,100"
        self.assertFalse(should_skip_due_to_token_capacity(4, 0, 0, 800, lens))
        self.assertTrue(
            should_skip_due_to_token_capacity(4, 0, 0, 800, lens, num_dp_ranks=2)
        )
        self.assertFalse(
            should_skip_due_to_token_capacity(4, 0, 0, 1200, lens, num_dp_ranks=2)
        )

    def test_uniform_batch_ignores_dp_ranks(self):
        self.assertFalse(
            should_skip_due_to_token_capacity(3, 100, 0, 300, num_dp_ranks=2)
        )


if __name__ == "__main__":
    unittest.main()
