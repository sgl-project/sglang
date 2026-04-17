"""Unit tests for DPBudget — field mapping and dispatch logic."""

import dataclasses
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import DPBudget, LoadBalanceMethod
from sglang.srt.managers.io_struct import GetLoadsReqOutput, WatchLoadUpdateReq

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


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

    def test_maps_num_total_tokens_to_total_tokens(self):
        # Reads num_total_tokens (used + pending prefill), not num_used_tokens.
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


class TestDPBudgetDispatch(CustomTestCase):
    def test_total_requests_picks_min_and_bumps(self):
        budget = DPBudget(dp_size=3)
        budget.total_requests = [5, 2, 8]
        target = budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.assertEqual(target, 1)
        # The chosen rank's request count is bumped by 1.
        self.assertEqual(budget.total_requests[1], 3)

    def test_total_tokens_picks_min_and_bumps_requests(self):
        budget = DPBudget(dp_size=3)
        budget.total_tokens = [500, 200, 800]
        budget.total_requests = [1, 1, 1]
        target = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS)
        self.assertEqual(target, 1)
        # Requests counter gets bumped; total_tokens itself is unchanged.
        self.assertEqual(budget.total_requests[1], 2)
        self.assertEqual(budget.total_tokens[1], 200)

    def test_total_tokens_tiebreak_by_requests(self):
        budget = DPBudget(dp_size=3)
        budget.total_tokens = [100, 100, 100]
        budget.total_requests = [5, 1, 3]
        target = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS)
        self.assertEqual(target, 1)

    def test_unsupported_methods_return_none(self):
        budget = DPBudget(dp_size=2)
        for method in (
            LoadBalanceMethod.ROUND_ROBIN,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM,
        ):
            with self.subTest(method=method):
                self.assertIsNone(budget.dispatch(method))


if __name__ == "__main__":
    unittest.main()
