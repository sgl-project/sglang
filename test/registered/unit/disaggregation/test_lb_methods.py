"""Unit tests for DataParallelController dispatch + DPBudget state machine.

Covers the simpler LB algorithms:
  - round_robin (counter + active-worker skip)
  - follow_bootstrap_room (bootstrap_room modulo, prefill default)
  - total_requests (DPBudget argmin)

The most complex algorithm, total_tokens, is exercised end-to-end in
test/registered/disaggregation/test_disaggregation_dp_attention.py — its
tie-breaking on total_requests transitively covers total_requests state
as well.

Fragility note:
  The scheduler-method tests bypass ``DataParallelController.__init__``
  via ``__new__`` and inject only the attributes the schedulers read:
  ``workers``, ``status``, ``round_robin_counter``, ``dp_budget``. If a
  scheduler method starts reading a new instance attribute, add it here.
  ``maybe_external_dp_rank_routing`` is exercised as a real method (no
  mock) so its bypass-on-``routed_dp_rank`` logic stays covered.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_controller(dp_size: int) -> DataParallelController:
    """Build a DataParallelController bypassing __init__.

    Only fields actually read by the dispatch path are populated:
    ``workers`` (zmq socket replacements), ``status`` (active flags),
    ``round_robin_counter``, and ``dp_budget``.
    """
    ctl = DataParallelController.__new__(DataParallelController)
    ctl.workers = [MagicMock(name=f"worker_{i}") for i in range(dp_size)]
    ctl.status = [True] * dp_size
    ctl.round_robin_counter = 0
    ctl.dp_budget = DPBudget(dp_size=dp_size)
    return ctl


def _req(routed_dp_rank=None, bootstrap_room=None, input_ids=None):
    """Minimal Req stand-in for dispatch testing.

    Schedulers only read ``routed_dp_rank``, ``bootstrap_room``, and
    ``input_ids``. Duck-typing via SimpleNamespace avoids pinning to the
    full Req dataclass schema.
    """
    return SimpleNamespace(
        routed_dp_rank=routed_dp_rank,
        bootstrap_room=bootstrap_room,
        input_ids=input_ids or [],
    )


def _load(dp_rank, running=0, waiting=0, total_tokens=0):
    """Minimal GetLoadsReqOutput stand-in for DPBudget.update_budget."""
    return SimpleNamespace(
        dp_rank=dp_rank,
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        num_total_tokens=total_tokens,
    )


class TestDPBudget(CustomTestCase):
    """DPBudget is a self-contained state machine — no mocks needed."""

    def test_update_budget_sums_running_and_waiting(self):
        budget = DPBudget(dp_size=4)
        budget.update_budget(
            SimpleNamespace(
                loads=[
                    _load(dp_rank=0, running=2, waiting=1, total_tokens=100),
                    _load(dp_rank=1, running=1, waiting=0, total_tokens=50),
                ]
            )
        )
        self.assertEqual(budget.total_requests, [3, 1, 0, 0])
        self.assertEqual(budget.total_tokens, [100, 50, 0, 0])

    def test_total_requests_dispatch_picks_min_and_increments(self):
        budget = DPBudget(dp_size=3)
        budget.total_requests = [4, 2, 7]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.assertEqual(rank, 1)
        self.assertEqual(
            budget.total_requests[1],
            3,
            "dispatch should increment chosen worker's request count",
        )

    def test_total_tokens_dispatch_applies_estimated_tokens(self):
        budget = DPBudget(dp_size=3)
        budget.total_tokens = [100, 50, 200]
        budget.total_requests = [0, 0, 0]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=30)
        self.assertEqual(rank, 1, "should pick worker with min total_tokens")
        self.assertEqual(
            budget.total_tokens[1],
            80,
            "dispatch should add estimated_tokens to chosen worker",
        )
        self.assertEqual(
            budget.total_requests[1],
            1,
            "dispatch should also increment request count",
        )

    def test_total_tokens_tie_breaks_on_total_requests(self):
        budget = DPBudget(dp_size=3)
        budget.total_tokens = [50, 50, 50]
        budget.total_requests = [4, 2, 7]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=10)
        self.assertEqual(
            rank, 1, "tie on total_tokens should fall back to min total_requests"
        )

    def test_dispatch_returns_none_for_methods_not_handled(self):
        """Round-robin and follow_bootstrap_room dispatch elsewhere; DPBudget
        only handles the load-aware variants."""
        budget = DPBudget(dp_size=3)
        self.assertIsNone(budget.dispatch(LoadBalanceMethod.ROUND_ROBIN))
        self.assertIsNone(budget.dispatch(LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM))


class TestRoundRobinScheduler(CustomTestCase):
    def test_cycles_through_active_workers_in_order(self):
        ctl = _make_controller(dp_size=4)
        for _ in range(8):
            ctl.round_robin_scheduler(_req())
        # 8 reqs across 4 active workers — 2 each, in round-robin order
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(worker.send_pyobj.call_count, 2, f"worker {i} call count")

    def test_first_dispatch_picks_worker_zero(self):
        ctl = _make_controller(dp_size=4)
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()
        for i in (1, 2, 3):
            ctl.workers[i].send_pyobj.assert_not_called()

    def test_skips_inactive_workers(self):
        ctl = _make_controller(dp_size=4)
        ctl.status[1] = False
        ctl.status[3] = False
        for _ in range(6):
            ctl.round_robin_scheduler(_req())
        # Only workers 0 and 2 are active — should split 6 reqs evenly
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 3)
        ctl.workers[1].send_pyobj.assert_not_called()
        self.assertEqual(ctl.workers[2].send_pyobj.call_count, 3)
        ctl.workers[3].send_pyobj.assert_not_called()

    def test_routed_dp_rank_bypasses_counter(self):
        """External dp-rank routing should not advance the round-robin
        counter. Tests the real maybe_external_dp_rank_routing branch."""
        ctl = _make_controller(dp_size=4)
        ctl.round_robin_scheduler(_req(routed_dp_rank=2))
        ctl.workers[2].send_pyobj.assert_called_once()
        self.assertEqual(
            ctl.round_robin_counter,
            0,
            "external routing must not advance the round-robin counter",
        )
        # Subsequent round-robin req still lands on worker 0
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()


class TestFollowBootstrapRoomScheduler(CustomTestCase):
    def test_dispatches_by_bootstrap_room_modulo(self):
        ctl = _make_controller(dp_size=4)
        for room, expected_rank in [
            (0, 0),
            (1, 1),
            (4, 0),
            (5, 1),
            (100, 0),
            (101, 1),
        ]:
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=room))
            ctl.workers[expected_rank].send_pyobj.assert_called()

    def test_requires_bootstrap_room(self):
        ctl = _make_controller(dp_size=4)
        with self.assertRaises(AssertionError):
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=None))

    def test_routed_dp_rank_bypasses_bootstrap_room(self):
        ctl = _make_controller(dp_size=4)
        ctl.follow_bootstrap_room_scheduler(_req(routed_dp_rank=3, bootstrap_room=1))
        ctl.workers[3].send_pyobj.assert_called_once()
        ctl.workers[1].send_pyobj.assert_not_called()


class TestTotalRequestsScheduler(CustomTestCase):
    def test_dispatches_to_min_request_worker(self):
        ctl = _make_controller(dp_size=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req())
        ctl.workers[2].send_pyobj.assert_called_once()
        for i in (0, 1, 3):
            ctl.workers[i].send_pyobj.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_requests[2],
            2,
            "DPBudget must record the dispatch by incrementing the counter",
        )

    def test_routed_dp_rank_bypasses_budget(self):
        ctl = _make_controller(dp_size=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req(routed_dp_rank=0))
        ctl.workers[0].send_pyobj.assert_called_once()
        # DPBudget must not be touched when bypassed
        self.assertEqual(
            ctl.dp_budget.total_requests,
            [5, 3, 1, 4],
            "external routing must not mutate DPBudget state",
        )


class TestStatusAwarenessInconsistency(CustomTestCase):
    """Document a divergence: ``round_robin_scheduler`` skips workers whose
    ``status`` is False, but ``total_requests_scheduler`` /
    ``total_tokens_scheduler`` route purely by DPBudget — they do NOT
    consult ``self.status``. If a future change unifies this behaviour,
    this test will fail and force a reviewer to confirm intent."""

    def test_total_requests_ignores_status(self):
        ctl = _make_controller(dp_size=4)
        # Worker 2 is the global minimum AND marked inactive.
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.status[2] = False
        ctl.total_requests_scheduler(_req())
        # Current behaviour: still dispatches to the inactive worker.
        ctl.workers[2].send_pyobj.assert_called_once()


if __name__ == "__main__":
    unittest.main()
