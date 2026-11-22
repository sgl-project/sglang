"""The concurrent_prefill warmup must reach the batch shapes serving will see:
one prompt, full concurrency, and a round larger than one prefill chunk."""

import unittest

from sglang.srt.entrypoints.warmup import concurrent_prefill_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestConcurrentPrefillPlan(CustomTestCase):
    def test_rounds_cover_single_and_full_concurrency(self):
        for max_running, rounds in ((None, [1]), (4, [1, 4]), (32, [1, 4, 32])):
            plan = concurrent_prefill_plan(max_running)
            self.assertEqual([len(sizes) for sizes in plan], rounds)

    def test_each_round_spans_a_chunk_boundary_with_distinct_lengths(self):
        for sizes in concurrent_prefill_plan(32):
            self.assertGreater(sum(sizes), 32768)
            self.assertEqual(len(set(sizes)), len(sizes))


if __name__ == "__main__":
    unittest.main()
