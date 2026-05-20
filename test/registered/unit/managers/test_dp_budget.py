"""Unit tests for DPBudget — field mapping regression guard.

This PR changed DPBudget.update_budget to read num_running_reqs +
num_waiting_reqs and num_total_tokens from LoadSnapshot.
These tests lock in that mapping. Pre-existing dispatch logic is not
retested here — it's covered by DP balance integration tests.
"""

import dataclasses
import unittest

from sglang.srt.managers.data_parallel_controller import DPBudget, LoadBalanceMethod
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()


register_cpu_ci(est_time=11, suite="base-a-test-cpu")


_BASE_LOAD = dataclasses.replace(
    LoadSnapshot(dp_rank=0),
    max_total_num_tokens=4096,
    max_running_requests=128,
)


def _load(**overrides) -> LoadSnapshot:
    return dataclasses.replace(_BASE_LOAD, **overrides)


class TestDPBudgetUpdateBudget(CustomTestCase):
    def test_maps_running_plus_waiting_to_total_requests(self):
        budget = DPBudget(dp_size=2)
        budget.update_budget(
            [
                _load(dp_rank=0, timestamp=1.0, num_running_reqs=3, num_waiting_reqs=2),
                _load(dp_rank=1, timestamp=1.0, num_running_reqs=5, num_waiting_reqs=1),
            ]
        )
        self.assertEqual(budget.total_requests, [5, 6])

    def test_maps_num_total_tokens_not_num_used_tokens(self):
        budget = DPBudget(dp_size=2)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_used_tokens=100, num_total_tokens=150
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_used_tokens=80, num_total_tokens=80
                ),
            ]
        )
        self.assertEqual(budget.total_tokens, [150, 80])

    def test_partial_update_only_affects_reported_rank(self):
        budget = DPBudget(dp_size=3)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_running_reqs=10, num_total_tokens=100
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_running_reqs=20, num_total_tokens=200
                ),
                _load(
                    dp_rank=2, timestamp=1.0, num_running_reqs=30, num_total_tokens=300
                ),
            ]
        )
        budget.update_budget(
            [
                _load(
                    dp_rank=1,
                    timestamp=2.0,
                    num_running_reqs=1,
                    num_waiting_reqs=1,
                    num_total_tokens=50,
                )
            ]
        )
        self.assertEqual(budget.total_requests, [10, 2, 30])
        self.assertEqual(budget.total_tokens, [100, 50, 300])

    def test_stale_reread_preserves_speculative_increments(self):
        budget = DPBudget(dp_size=2)
        budget.update_budget(
            [
                _load(dp_rank=0, timestamp=1.0, num_running_reqs=0, num_waiting_reqs=0),
                _load(dp_rank=1, timestamp=1.0, num_running_reqs=0, num_waiting_reqs=0),
            ]
        )
        budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.assertEqual(budget.total_requests, [1, 0])

        # Re-read same stale snapshot (same timestamp) — increment survives
        budget.update_budget(
            [
                _load(dp_rank=0, timestamp=1.0, num_running_reqs=0, num_waiting_reqs=0),
                _load(dp_rank=1, timestamp=1.0, num_running_reqs=0, num_waiting_reqs=0),
            ]
        )
        self.assertEqual(budget.total_requests, [1, 0])

    def test_fresh_snapshot_overwrites_speculative_increments(self):
        budget = DPBudget(dp_size=2)
        budget.update_budget(
            [
                _load(dp_rank=0, timestamp=1.0, num_running_reqs=0, num_waiting_reqs=0),
                _load(dp_rank=1, timestamp=1.0, num_running_reqs=0, num_waiting_reqs=0),
            ]
        )
        budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.assertEqual(budget.total_requests, [1, 1])

        # Fresh snapshot from scheduler (new timestamp) overwrites
        budget.update_budget(
            [_load(dp_rank=0, timestamp=2.0, num_running_reqs=1, num_waiting_reqs=0)]
        )
        self.assertEqual(budget.total_requests, [1, 1])


if __name__ == "__main__":
    unittest.main()
