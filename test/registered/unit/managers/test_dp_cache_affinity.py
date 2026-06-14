import unittest
from collections import OrderedDict
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import (  # noqa: E402
    DPCacheAffinityMethod,
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeWorker:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, req):
        self.sent.append(req)


def _req(
    rid: str,
    routing_key=None,
    num_tokens: int = 1,
    routed_dp_rank=None,
    bootstrap_room: int = 0,
):
    return SimpleNamespace(
        rid=rid,
        routing_key=routing_key,
        input_ids=list(range(num_tokens)),
        routed_dp_rank=routed_dp_rank,
        bootstrap_room=bootstrap_room,
    )


def _controller(
    load_balance_method=LoadBalanceMethod.TOTAL_TOKENS,
    dp_cache_affinity_method=DPCacheAffinityMethod.ROUTING_KEY,
    dp_size: int = 3,
    dp_cache_affinity_max_keys: int = 100_000,
):
    controller = DataParallelController.__new__(DataParallelController)
    controller.workers = [_FakeWorker() for _ in range(dp_size)]
    controller.status = [True] * dp_size
    controller.round_robin_counter = 0
    controller.dp_budget = DPBudget(dp_size)
    controller.load_balance_method = load_balance_method
    controller.dp_cache_affinity_method = dp_cache_affinity_method
    controller.dp_cache_affinity_max_keys = dp_cache_affinity_max_keys
    controller.routing_key_to_dp_rank = OrderedDict()
    return controller


class TestDPCacheAffinity(CustomTestCase):
    def test_routing_key_sticks_to_first_total_token_assignment(self):
        controller = _controller()
        controller.dp_budget.total_tokens = [50, 5, 20]

        first_req = _req("first", routing_key="session-a", num_tokens=7)
        controller.total_tokens_scheduler(first_req)

        self.assertEqual(controller.routing_key_to_dp_rank["session-a"], 1)
        self.assertEqual(controller.workers[1].sent, [first_req])
        self.assertEqual(controller.dp_budget.total_tokens, [50, 12, 20])

        controller.dp_budget.total_tokens = [0, 100, 0]
        second_req = _req("second", routing_key="session-a", num_tokens=3)
        controller.total_tokens_scheduler(second_req)

        self.assertEqual(controller.workers[1].sent, [first_req, second_req])
        self.assertEqual(controller.dp_budget.total_tokens, [0, 103, 0])

    def test_missing_routing_key_falls_back_to_load_balance_method(self):
        controller = _controller()
        controller.dp_budget.total_tokens = [10, 1, 2]

        req = _req("no-key", routing_key=None, num_tokens=4)
        controller.total_tokens_scheduler(req)

        self.assertEqual(controller.routing_key_to_dp_rank, {})
        self.assertEqual(controller.workers[1].sent, [req])
        self.assertEqual(controller.dp_budget.total_tokens, [10, 5, 2])

    def test_unavailable_mapped_rank_is_remapped(self):
        controller = _controller()
        controller.routing_key_to_dp_rank["session-a"] = 1
        controller.status[1] = False
        controller.dp_budget.total_tokens = [30, 0, 5]

        req = _req("remap", routing_key="session-a", num_tokens=2)
        controller.total_tokens_scheduler(req)

        self.assertEqual(controller.routing_key_to_dp_rank["session-a"], 2)
        self.assertEqual(controller.workers[2].sent, [req])
        self.assertEqual(controller.workers[1].sent, [])
        self.assertEqual(controller.dp_budget.total_tokens, [30, 0, 7])

    def test_explicit_routed_dp_rank_overrides_cache_affinity(self):
        controller = _controller()
        controller.routing_key_to_dp_rank["session-a"] = 1

        req = _req("direct", routing_key="session-a", routed_dp_rank=2)
        controller.total_tokens_scheduler(req)

        self.assertEqual(controller.routing_key_to_dp_rank["session-a"], 1)
        self.assertEqual(controller.workers[2].sent, [req])
        self.assertEqual(controller.workers[1].sent, [])

    def test_routing_key_affinity_can_use_round_robin_for_first_assignment(self):
        controller = _controller(load_balance_method=LoadBalanceMethod.ROUND_ROBIN)

        first_req = _req("first", routing_key="session-a")
        second_req = _req("second", routing_key="session-b")
        third_req = _req("third", routing_key="session-a")

        controller.round_robin_scheduler(first_req)
        controller.round_robin_scheduler(second_req)
        controller.round_robin_scheduler(third_req)

        self.assertEqual(controller.routing_key_to_dp_rank["session-a"], 0)
        self.assertEqual(controller.routing_key_to_dp_rank["session-b"], 1)
        self.assertEqual(controller.workers[0].sent, [first_req, third_req])
        self.assertEqual(controller.workers[1].sent, [second_req])

    def test_routing_key_affinity_evicts_least_recent_key(self):
        controller = _controller(
            load_balance_method=LoadBalanceMethod.ROUND_ROBIN,
            dp_cache_affinity_max_keys=2,
        )

        first_req = _req("first", routing_key="session-a")
        second_req = _req("second", routing_key="session-b")
        refresh_req = _req("refresh", routing_key="session-a")
        third_req = _req("third", routing_key="session-c")

        controller.round_robin_scheduler(first_req)
        controller.round_robin_scheduler(second_req)
        controller.round_robin_scheduler(refresh_req)
        controller.round_robin_scheduler(third_req)

        self.assertEqual(
            list(controller.routing_key_to_dp_rank.keys()),
            ["session-a", "session-c"],
        )
        self.assertNotIn("session-b", controller.routing_key_to_dp_rank)


if __name__ == "__main__":
    unittest.main()
