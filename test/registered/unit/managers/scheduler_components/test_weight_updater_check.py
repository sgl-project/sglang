"""Unit tests for TP-wide weight checker results."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import CheckWeightsReqInput
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Runner:
    def __init__(self, *, payload=None, error=None):
        self.payload = payload
        self.error = error

    def check_weights(self, **kwargs):
        if self.error is not None:
            raise ValueError(self.error)
        return self.payload


class TestWeightUpdaterCheck(CustomTestCase):
    def _check_on_two_ranks(self, action, runners):
        # Each worker reaches the same fake collective only after its local
        # checker finishes. A missing collective breaks the expected result.
        barrier = Barrier(2, timeout=3)
        gathered = [None, None]

        def all_gather_object(output, value, *, group):
            gathered[group] = value
            barrier.wait()
            output[:] = gathered
            barrier.wait()

        def check(rank):
            worker = SimpleNamespace(
                weight_update_runners=lambda: [("target", runners[rank])]
            )
            manager = SchedulerWeightUpdaterManager(
                tp_worker=worker,
                draft_worker=None,
                tp_cpu_group=rank,
                memory_saver_adapter=None,
                flush_cache=lambda: True,
                is_fully_idle=lambda: True,
            )
            return manager.check_weights(CheckWeightsReqInput(action=action))

        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_gather_object", side_effect=all_gather_object),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [executor.submit(check, rank) for rank in range(2)]
            return [future.result(timeout=5) for future in futures]

    def test_nonzero_rank_compare_failure_reaches_every_rank(self):
        results = self._check_on_two_ranks(
            "compare", [_Runner(), _Runner(error="weight mismatch")]
        )

        for result in results:
            self.assertFalse(result.success)
            self.assertIn("TP rank 1: weight mismatch", result.message)
            self.assertIsNone(result.payload)

    def test_checksum_failure_does_not_strand_other_rank(self):
        results = self._check_on_two_ranks(
            "checksum", [_Runner(payload=self._checksum(0)), _Runner(error="bad hash")]
        )

        for result in results:
            self.assertFalse(result.success)
            self.assertIn("TP rank 1: bad hash", result.message)

    def test_checksum_keeps_one_payload_per_rank(self):
        results = self._check_on_two_ranks(
            "checksum",
            [_Runner(payload=self._checksum(0)), _Runner(payload=self._checksum(1))],
        )

        for result in results:
            self.assertTrue(result.success)
            self.assertEqual(len(result.payload), 2)
            self.assertEqual(
                [p.parallelism_info[0].tp_rank for p in result.payload], [0, 1]
            )

    @staticmethod
    def _checksum(rank):
        return {
            "checksums": {"weight": str(rank)},
            "parallelism_info": {
                "role": "target",
                "tp_rank": rank,
                "tp_size": 2,
                "dp_rank": 0,
                "dp_size": 1,
                "pp_rank": 0,
                "pp_size": 1,
                "rank": rank,
                "size": 2,
            },
        }


if __name__ == "__main__":
    unittest.main()
