"""Unit tests for DPBudget — field mapping regression guard.

This PR changed DPBudget.update_budget to read num_running_reqs +
num_waiting_reqs and num_total_tokens from the new GetLoadsReqOutput.
These tests lock in that mapping. Pre-existing dispatch logic is not
retested here — it's covered by DP balance integration tests.
"""

import dataclasses
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import DPBudget, LoadBalanceMethod
from sglang.srt.managers.io_struct import GetLoadsReqOutput, WatchLoadUpdateReq
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="stage-a-test-cpu")


_BASE_LOAD = GetLoadsReqOutput(
    dp_rank=0,
    timestamp=0.0,
    num_running_reqs=0,
    num_waiting_reqs=0,
    num_used_tokens=0,
    num_total_tokens=0,
    max_total_num_tokens=4096,
    token_usage=0.0,
    gen_throughput=0.0,
    cache_hit_rate=0.0,
    utilization=0.0,
    max_running_requests=128,
)


def _load(**overrides) -> GetLoadsReqOutput:
    return dataclasses.replace(_BASE_LOAD, **overrides)


class TestDPBudgetUpdateBudget(CustomTestCase):
    def test_maps_running_plus_waiting_to_total_requests(self):
        budget = DPBudget(dp_size=2)
        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(dp_rank=0, num_running_reqs=3, num_waiting_reqs=2),
                    _load(dp_rank=1, num_running_reqs=5, num_waiting_reqs=1),
                ]
            )
        )
        self.assertEqual(budget.total_requests, [5, 6])

    def test_maps_num_total_tokens_not_num_used_tokens(self):
        # Reads num_total_tokens (used + queued prefill), NOT num_used_tokens.
        # A silent swap here would break DP balance for long-prompt workloads.
        budget = DPBudget(dp_size=2)
        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(dp_rank=0, num_used_tokens=100, num_total_tokens=150),
                    _load(dp_rank=1, num_used_tokens=80, num_total_tokens=80),
                ]
            )
        )
        self.assertEqual(budget.total_tokens, [150, 80])

    def test_partial_update_only_affects_reported_rank(self):
        budget = DPBudget(dp_size=3)
        budget.total_requests = [10, 20, 30]
        budget.total_tokens = [100, 200, 300]
        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(
                        dp_rank=1,
                        num_running_reqs=1,
                        num_waiting_reqs=1,
                        num_total_tokens=50,
                    )
                ]
            )
        )
        self.assertEqual(budget.total_requests, [10, 2, 30])
        self.assertEqual(budget.total_tokens, [100, 50, 300])

    def test_dispatch_rotates_ties_and_accounts_dispatched_load(self):
        budget = DPBudget(dp_size=3)

        self.assertEqual(
            [
                budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)[0]
                for _ in range(4)
            ],
            [0, 1, 2, 0],
        )
        self.assertEqual(budget.total_requests, [2, 1, 1])

    def test_total_tokens_dispatch_accounts_estimated_tokens(self):
        budget = DPBudget(dp_size=2)

        self.assertEqual(
            budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=32)[0],
            0,
        )
        self.assertEqual(budget.total_requests, [1, 0])
        self.assertEqual(budget.total_tokens, [32, 0])

    def test_load_update_preserves_unacked_dispatched_load(self):
        budget = DPBudget(dp_size=2, reserved_tokens_per_req=512)

        target_rank, dispatch_seq, dispatch_cum_tokens = budget.dispatch(
            LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=1536
        )
        self.assertEqual(target_rank, 0)
        self.assertEqual(dispatch_seq, 1)
        self.assertEqual(dispatch_cum_tokens, 1536)

        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(
                        dp_rank=0,
                        num_running_reqs=0,
                        num_waiting_reqs=0,
                        num_total_tokens=0,
                        dp_dispatch_ack_seq=0,
                        dp_dispatch_ack_cum_tokens=0,
                    )
                ]
            )
        )
        self.assertEqual(budget.total_requests, [1, 0])
        self.assertEqual(budget.total_tokens, [1536, 0])

        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(
                        dp_rank=0,
                        num_running_reqs=1,
                        num_waiting_reqs=0,
                        num_total_tokens=1024,
                        num_pending_output_tokens=512,
                        dp_dispatch_ack_seq=1,
                        dp_dispatch_ack_cum_tokens=1536,
                    )
                ]
            )
        )
        self.assertEqual(budget.total_requests, [1, 0])
        self.assertEqual(budget.total_tokens, [1536, 0])

        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(
                        dp_rank=0,
                        num_running_reqs=0,
                        num_waiting_reqs=0,
                        num_total_tokens=0,
                        dp_dispatch_ack_seq=1,
                        dp_dispatch_ack_cum_tokens=1536,
                    )
                ]
            )
        )
        self.assertEqual(budget.total_requests, [0, 0])
        self.assertEqual(budget.total_tokens, [0, 0])

    def test_total_tokens_includes_pending_output_tokens(self):
        budget = DPBudget(dp_size=2, reserved_tokens_per_req=512)
        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(
                        dp_rank=0,
                        num_running_reqs=1,
                        num_waiting_reqs=1,
                        num_total_tokens=2048,
                        num_pending_output_tokens=1024,
                    ),
                    _load(
                        dp_rank=1,
                        num_running_reqs=1,
                        num_waiting_reqs=1,
                        num_total_tokens=2048,
                        num_pending_output_tokens=256,
                    ),
                ]
            )
        )
        self.assertEqual(budget.total_tokens, [3072, 2304])


def _scheduler_accounting_state():
    return SimpleNamespace(
        dp_dispatch_ack_seq=0,
        dp_dispatch_ack_cum_tokens=0,
        _dp_dispatch_accounted={},
    )


class TestSchedulerDPDispatchAccounting(CustomTestCase):
    def test_ack_advances_only_after_contiguous_accounting(self):
        scheduler = _scheduler_accounting_state()

        Scheduler.observe_dp_dispatch_accounted(
            scheduler,
            SimpleNamespace(dp_dispatch_seq=2, dp_dispatch_cum_tokens=300),
        )
        self.assertEqual(scheduler.dp_dispatch_ack_seq, 0)
        self.assertEqual(scheduler.dp_dispatch_ack_cum_tokens, 0)

        Scheduler.observe_dp_dispatch_accounted(
            scheduler,
            SimpleNamespace(dp_dispatch_seq=1, dp_dispatch_cum_tokens=100),
        )
        self.assertEqual(scheduler.dp_dispatch_ack_seq, 2)
        self.assertEqual(scheduler.dp_dispatch_ack_cum_tokens, 300)
        self.assertEqual(scheduler._dp_dispatch_accounted, {})

    def test_duplicate_or_unstamped_request_does_not_move_ack(self):
        scheduler = _scheduler_accounting_state()

        Scheduler.observe_dp_dispatch_accounted(
            scheduler,
            SimpleNamespace(dp_dispatch_seq=0, dp_dispatch_cum_tokens=0),
        )
        Scheduler.observe_dp_dispatch_accounted(
            scheduler,
            SimpleNamespace(dp_dispatch_seq=1, dp_dispatch_cum_tokens=100),
        )
        Scheduler.observe_dp_dispatch_accounted(
            scheduler,
            SimpleNamespace(dp_dispatch_seq=1, dp_dispatch_cum_tokens=200),
        )

        self.assertEqual(scheduler.dp_dispatch_ack_seq, 1)
        self.assertEqual(scheduler.dp_dispatch_ack_cum_tokens, 100)
        self.assertEqual(scheduler._dp_dispatch_accounted, {})


if __name__ == "__main__":
    unittest.main()
