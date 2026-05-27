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

from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    DPBudget,
    DPCacheAffinityMethod,
    LoadBalanceMethod,
)
from sglang.srt.managers.io_struct import GetLoadsReqOutput, WatchLoadUpdateReq

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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
        # Reads num_total_tokens (used + pending prefill), NOT num_used_tokens.
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


class FakeWorker:
    def __init__(self):
        self.reqs = []

    def send_pyobj(self, req):
        self.reqs.append(req)


def _controller(method=LoadBalanceMethod.TOTAL_TOKENS):
    controller = DataParallelController.__new__(DataParallelController)
    controller.workers = [FakeWorker(), FakeWorker()]
    controller.status = [True, True]
    controller.round_robin_counter = 0
    controller.dp_budget = DPBudget(dp_size=2)
    controller.load_balance_method = method
    controller.dp_cache_affinity_method = DPCacheAffinityMethod.ROUTING_KEY
    controller.routing_key_to_dp_rank = {}
    return controller


def _req(
    routing_key=None,
    routed_dp_rank=None,
    bootstrap_room=0,
    input_ids=None,
):
    return SimpleNamespace(
        routing_key=routing_key,
        routed_dp_rank=routed_dp_rank,
        bootstrap_room=bootstrap_room,
        input_ids=input_ids or [1],
    )


class TestDPRoutingKeyCacheAffinity(CustomTestCase):
    def test_same_routing_key_routes_to_same_dp_rank(self):
        controller = _controller()
        controller.dp_budget.total_tokens = [10, 0]

        controller.total_tokens_scheduler(_req(routing_key="session-a", input_ids=[1]))
        controller.dp_budget.total_tokens = [0, 10]
        controller.total_tokens_scheduler(_req(routing_key="session-a", input_ids=[2]))

        self.assertEqual([len(w.reqs) for w in controller.workers], [0, 2])
        self.assertEqual(controller.routing_key_to_dp_rank["session-a"], 1)

    def test_new_routing_keys_use_base_load_balancer(self):
        controller = _controller()
        controller.dp_budget.total_tokens = [10, 0]
        controller.total_tokens_scheduler(_req(routing_key="session-a"))

        controller.dp_budget.total_tokens = [0, 10]
        controller.total_tokens_scheduler(_req(routing_key="session-b"))

        self.assertEqual([len(w.reqs) for w in controller.workers], [1, 1])
        self.assertEqual(controller.routing_key_to_dp_rank["session-a"], 1)
        self.assertEqual(controller.routing_key_to_dp_rank["session-b"], 0)

    def test_missing_routing_key_falls_back_to_base_load_balancer(self):
        controller = _controller()
        controller.dp_budget.total_tokens = [10, 0]

        controller.total_tokens_scheduler(_req())

        self.assertEqual([len(w.reqs) for w in controller.workers], [0, 1])
        self.assertEqual(controller.routing_key_to_dp_rank, {})

    def test_routed_dp_rank_overrides_cache_affinity(self):
        controller = _controller()
        controller.routing_key_to_dp_rank["session-a"] = 0

        controller.total_tokens_scheduler(
            _req(routing_key="session-a", routed_dp_rank=1)
        )

        self.assertEqual([len(w.reqs) for w in controller.workers], [0, 1])
        self.assertEqual(controller.routing_key_to_dp_rank["session-a"], 0)


if __name__ == "__main__":
    unittest.main()
