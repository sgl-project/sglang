"""Union capture grids and failure restoration protect target/owner isolation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.speculative.dflash_cohort import (
    DFlashCohort,
    cohort_draft_graph_buckets,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class CohortGraphTests(CustomTestCase):
    def test_exact_union_counts_include_partial_idle_and_maximum(self):
        self.assertEqual(cohort_draft_graph_buckets(2, 1, [1]), (1, 2))
        self.assertEqual(cohort_draft_graph_buckets(4, 1, [1]), (1, 2, 3, 4))
        self.assertEqual(cohort_draft_graph_buckets(4, 2, [1]), (1, 2, 3, 4))
        self.assertEqual(cohort_draft_graph_buckets(2, 2, [1, 2]), (1, 2, 3, 4))
        for args in [(1, 1, [1]), (8, 1, [1]), (2, 0, [1]), (2, 1, []), (2, 1, [0])]:
            with self.assertRaises(ValueError):
                cohort_draft_graph_buckets(*args)

    def cohort(self):
        cohort = object.__new__(DFlashCohort)
        cohort.owner_count = 4
        cohort.owner_slots = 1
        cohort.group = SimpleNamespace(rank_in_group=0)
        cohort.worker = SimpleNamespace(
            device="cuda:0",
            block_size=12,
            _draft_worker=SimpleNamespace(
                init_cuda_graphs=Mock(), get_decode_cuda_graph_bs=lambda: [1, 2, 3, 4]
            ),
        )
        cohort.graph_batch_sizes = ()
        return cohort

    def test_capture_restores_target_grid_and_publishes_only_complete_capture(self):
        cohort = self.cohort()
        decode = SimpleNamespace(bs=[1], backend=Backend.FULL)
        original = decode.bs

        def capture(**kwargs):
            self.assertEqual(kwargs, dict(capture_decode_cuda_graph=True))
            self.assertEqual(decode.bs, [1, 2, 3, 4])

        cohort.worker._draft_worker.init_cuda_graphs.side_effect = capture
        with patch(
            "sglang.srt.speculative.dflash_cohort.get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=decode))
            ),
        ):
            cohort.init_draft_cuda_graphs()
        self.assertIs(decode.bs, original)
        self.assertEqual(cohort.graph_batch_sizes, (1, 2, 3, 4))

    def test_capture_failure_keeps_target_grid_and_no_replay_eligibility(self):
        cohort = self.cohort()
        decode = SimpleNamespace(bs=[1], backend=Backend.FULL)
        original = decode.bs
        cohort.worker._draft_worker.init_cuda_graphs.side_effect = RuntimeError(
            "capture failed"
        )
        with patch(
            "sglang.srt.speculative.dflash_cohort.get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=decode))
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                cohort.init_draft_cuda_graphs()
        self.assertIs(decode.bs, original)
        self.assertEqual(cohort.graph_batch_sizes, ())

    def test_partial_capture_is_rejected(self):
        cohort = self.cohort()
        cohort.worker._draft_worker.get_decode_cuda_graph_bs = lambda: [1]
        decode = SimpleNamespace(bs=[1], backend=Backend.FULL)
        with patch(
            "sglang.srt.speculative.dflash_cohort.get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=decode))
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "grid mismatch"):
                cohort.init_draft_cuda_graphs()
        self.assertEqual(decode.bs, [1])
        self.assertEqual(cohort.graph_batch_sizes, ())


if __name__ == "__main__":
    unittest.main()
