"""Unit tests for DPBudget — field mapping and dispatch logic."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import DPBudget, LoadBalanceMethod
from sglang.srt.managers.io_struct import GetLoadsReqOutput, WatchLoadUpdateReq

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_loads_output(
    *,
    dp_rank: int,
    running: int = 0,
    waiting: int = 0,
    total: int = 0,
    used: int = 0,
) -> GetLoadsReqOutput:
    return GetLoadsReqOutput(
        dp_rank=dp_rank,
        timestamp=0.0,
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        num_used_tokens=used,
        num_total_tokens=total,
        max_total_num_tokens=4096,
        token_usage=0.0,
        gen_throughput=0.0,
        cache_hit_rate=0.0,
        utilization=0.0,
        max_running_requests=128,
    )


class TestDPBudgetUpdateBudget(CustomTestCase):
    def test_maps_running_plus_waiting_to_total_requests(self):
        budget = DPBudget(dp_size=2)
        update = WatchLoadUpdateReq(
            loads=[
                _make_loads_output(dp_rank=0, running=3, waiting=2),
                _make_loads_output(dp_rank=1, running=5, waiting=1),
            ]
        )
        budget.update_budget(update)
        self.assertEqual(budget.total_requests[0], 5)
        self.assertEqual(budget.total_requests[1], 6)

    def test_maps_num_total_tokens_to_total_tokens(self):
        # Reads num_total_tokens (used + pending prefill), not num_used_tokens.
        budget = DPBudget(dp_size=2)
        update = WatchLoadUpdateReq(
            loads=[
                _make_loads_output(dp_rank=0, used=100, total=150),
                _make_loads_output(dp_rank=1, used=80, total=80),
            ]
        )
        budget.update_budget(update)
        self.assertEqual(budget.total_tokens[0], 150)
        self.assertEqual(budget.total_tokens[1], 80)

    def test_partial_update_only_affects_reported_rank(self):
        budget = DPBudget(dp_size=3)
        budget.total_requests = [10, 20, 30]
        budget.total_tokens = [100, 200, 300]
        # Only update rank 1.
        update = WatchLoadUpdateReq(
            loads=[_make_loads_output(dp_rank=1, running=1, waiting=1, total=50)]
        )
        budget.update_budget(update)
        self.assertEqual(budget.total_requests, [10, 2, 30])
        self.assertEqual(budget.total_tokens, [100, 50, 300])


class TestDPBudgetDispatch(CustomTestCase):
    def test_dispatch_total_requests_picks_min(self):
        budget = DPBudget(dp_size=3)
        budget.total_requests = [5, 2, 8]
        target = budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.assertEqual(target, 1)
        # Heuristic: the chosen rank's request count is bumped by 1.
        self.assertEqual(budget.total_requests[1], 3)

    def test_dispatch_total_tokens_picks_min(self):
        budget = DPBudget(dp_size=3)
        budget.total_tokens = [500, 200, 800]
        budget.total_requests = [1, 1, 1]
        target = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS)
        self.assertEqual(target, 1)
        # Requests counter is still the one that gets bumped.
        self.assertEqual(budget.total_requests[1], 2)
        # total_tokens unchanged by dispatch itself.
        self.assertEqual(budget.total_tokens[1], 200)

    def test_dispatch_total_tokens_tiebreak_by_requests(self):
        budget = DPBudget(dp_size=3)
        budget.total_tokens = [100, 100, 100]
        budget.total_requests = [5, 1, 3]
        # All tokens equal -> fall back to running+waiting requests.
        target = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS)
        self.assertEqual(target, 1)

    def test_dispatch_unsupported_method_returns_none(self):
        budget = DPBudget(dp_size=2)
        self.assertIsNone(budget.dispatch(LoadBalanceMethod.ROUND_ROBIN))
        self.assertIsNone(budget.dispatch(LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM))


if __name__ == "__main__":
    unittest.main()
