"""DPBudget + DataParallelController dispatch tests.

`total_tokens` (the most complex algorithm) is exercised end-to-end in
test/registered/disaggregation/test_disaggregation_dp_attention.py; its
tie-break on `total_requests` transitively covers that state.

Fragility: scheduler tests bypass `DataParallelController.__init__` via
`__new__` and inject only the attrs the schedulers read (`workers`,
`status`, `round_robin_counter`, `dp_budget`). Update `_make_controller`
if a scheduler starts reading another attr. `maybe_external_dp_rank_routing`
is exercised as the real method, no mock.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec.structs

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


_BASE_LOAD = msgspec.structs.replace(
    LoadSnapshot(dp_rank=0),
    max_total_num_tokens=4096,
    max_running_requests=128,
)


def _load(**overrides) -> LoadSnapshot:
    return msgspec.structs.replace(_BASE_LOAD, **overrides)


def _make_controller(dp_size: int) -> DataParallelController:
    """Bypass __init__; inject only the attrs dispatch methods read."""
    ctl = DataParallelController.__new__(DataParallelController)
    ctl.workers = [MagicMock(name=f"worker_{i}") for i in range(dp_size)]
    ctl.status = [True] * dp_size
    ctl.round_robin_counter = 0
    ctl.dp_budget = DPBudget(dp_size=dp_size)
    return ctl


def _req(routed_dp_rank=None, bootstrap_room=None, input_ids=None):
    """Req stand-in; SimpleNamespace avoids pinning to the Req dataclass schema."""
    return SimpleNamespace(
        routed_dp_rank=routed_dp_rank,
        bootstrap_room=bootstrap_room,
        input_ids=input_ids or [],
    )


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


class TestDPBudgetDispatch(CustomTestCase):
    """DPBudget.dispatch picks a rank from current state and updates counters."""

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
        """External dp-rank routing must not advance the counter."""
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


# ==== prefix_affinity tests (feat/dp-prefix-affinity) ====
#
# Same fixture philosophy as above: bypass __init__ via __new__ and inject only
# what prefix_affinity_scheduler reads. Beyond the base attrs it also reads the
# four _affinity_* config attrs and req.routing_key, so _make_affinity_controller
# and _areq add those. Update them if the scheduler starts reading more.


def _make_affinity_controller(
    dp_size: int,
    fallback: str = "total_tokens",
    max_load_skew: float = 1.5,
    hash_tokens: int = 4096,
    disable_token_fallback: bool = False,
) -> DataParallelController:
    ctl = _make_controller(dp_size)
    ctl._affinity_fallback = LoadBalanceMethod.from_str(fallback)
    ctl._affinity_max_load_skew = max_load_skew
    ctl._affinity_hash_tokens = hash_tokens
    ctl._affinity_disable_token_fallback = disable_token_fallback
    return ctl


def _areq(routing_key=None, input_ids=None, routed_dp_rank=None):
    """Req stand-in that also carries routing_key (read by prefix_affinity)."""
    return SimpleNamespace(
        routed_dp_rank=routed_dp_rank,
        bootstrap_room=None,
        input_ids=input_ids or [],
        routing_key=routing_key,
    )


def _dispatched_rank(ctl) -> int:
    """Index of the single worker whose send_pyobj was called once."""
    called = [i for i, w in enumerate(ctl.workers) if w.send_pyobj.call_count == 1]
    assert len(called) == 1, f"expected exactly one dispatch, got {called}"
    return called[0]


class TestPrefixAffinityRoutingKey(CustomTestCase):
    def test_same_routing_key_pins_to_same_rank(self):
        """Cache-affinity: identical routing_key must always co-locate."""
        ctl = _make_affinity_controller(dp_size=8)
        ranks = set()
        for _ in range(20):
            c = _make_affinity_controller(dp_size=8)
            c.prefix_affinity_scheduler(_areq(routing_key="session-A"))
            ranks.add(_dispatched_rank(c))
        self.assertEqual(len(ranks), 1, "same routing_key must be deterministic")

    def test_distinct_routing_keys_spread_across_ranks(self):
        """The headline fix: many sessions sharing a giant prefix but with
        distinct routing_keys must spread across ranks, NOT collapse to one
        (the failure mode of first-N-token hashing in #26612)."""
        ctl = _make_affinity_controller(dp_size=8)
        shared_prefix = list(range(8192))  # identical huge system prompt
        used = set()
        for i in range(256):
            ctl.prefix_affinity_scheduler(
                _areq(routing_key=f"session-{i}", input_ids=shared_prefix)
            )
            # find which worker got THIS req (cumulative counts)
        used = {i for i, w in enumerate(ctl.workers) if w.send_pyobj.call_count > 0}
        self.assertEqual(
            used, set(range(8)), "distinct routing_keys must cover all live ranks"
        )

    def test_identical_key_under_load_spills_instead_of_hotspotting(self):
        """The overload guard's reason for existing: even when every request
        maps to the SAME key (here via the token-prefix fallback, no
        routing_key), accumulating load on the preferred rank makes the guard
        spill subsequent requests to other ranks instead of hotspotting one --
        the concurrency failure mode of plain hash routing (#26612)."""
        ctl = _make_affinity_controller(dp_size=8)
        shared_prefix = list(range(8192))
        for _ in range(64):
            ctl.prefix_affinity_scheduler(_areq(input_ids=shared_prefix))
        used = {i for i, w in enumerate(ctl.workers) if w.send_pyobj.call_count > 0}
        self.assertGreater(
            len(used), 1, "overload guard must spill a hot shared key across ranks"
        )
        # ...but the first request (no load yet) lands on the HRW-preferred rank,
        # so affinity still holds at low load.
        fresh = _make_affinity_controller(dp_size=8)
        fresh.prefix_affinity_scheduler(_areq(input_ids=shared_prefix))
        preferred = ctl._rendezvous_ranked(
            ctl._token_prefix_key(_areq(input_ids=shared_prefix), 4096), list(range(8))
        )[0]
        self.assertEqual(_dispatched_rank(fresh), preferred)


class TestPrefixAffinityHRWStability(CustomTestCase):
    def test_dropping_a_rank_only_moves_its_keys(self):
        """Rendezvous hashing invariant: taking one rank offline must not
        remap keys that were not on it (unlike hash % dp_size)."""
        keys = [f"session-{i}" for i in range(500)]

        full = _make_affinity_controller(dp_size=8)
        before = {}
        for k in keys:
            c = _make_affinity_controller(dp_size=8)
            c.prefix_affinity_scheduler(_areq(routing_key=k))
            before[k] = _dispatched_rank(c)

        dropped = 3
        after = {}
        for k in keys:
            c = _make_affinity_controller(dp_size=8)
            c.status[dropped] = False
            c.prefix_affinity_scheduler(_areq(routing_key=k))
            after[k] = _dispatched_rank(c)

        for k in keys:
            if before[k] != dropped:
                self.assertEqual(
                    after[k], before[k], f"key {k} moved but its rank was up"
                )
            else:
                self.assertNotEqual(after[k], dropped, "dropped rank still used")


class TestPrefixAffinityOverloadGuard(CustomTestCase):
    def test_overloaded_preferred_rank_spills_to_next_candidate(self):
        """When the HRW-preferred rank is over the load-skew threshold, the
        request must go to the next-best rank, not hotspot the preferred one."""
        ctl = _make_affinity_controller(dp_size=4, max_load_skew=1.5)
        key = "session-hot"
        order = ctl._rendezvous_ranked(key, [0, 1, 2, 3])
        preferred = order[0]
        # Make only the preferred rank overloaded (>1.5x average).
        loads = [0, 0, 0, 0]
        loads[preferred] = 100
        ctl.dp_budget.total_tokens = list(loads)
        ctl.dp_budget.total_requests = [0, 0, 0, 0]

        ctl.prefix_affinity_scheduler(_areq(routing_key=key, input_ids=[1, 2, 3]))

        ctl.workers[preferred].send_pyobj.assert_not_called()
        chosen = _dispatched_rank(ctl)
        self.assertEqual(chosen, order[1], "should spill to the next HRW candidate")
        self.assertEqual(
            ctl.dp_budget.total_requests[chosen], 1, "budget must record the dispatch"
        )

    def test_single_live_rank_is_never_overloaded(self):
        """Overload guard must not deadlock when only one rank is live."""
        ctl = _make_affinity_controller(dp_size=4)
        for r in (1, 2, 3):
            ctl.status[r] = False
        ctl.dp_budget.total_tokens = [10_000, 0, 0, 0]
        ctl.prefix_affinity_scheduler(_areq(routing_key="k", input_ids=[1]))
        ctl.workers[0].send_pyobj.assert_called_once()


class TestPrefixAffinityFallback(CustomTestCase):
    def test_token_prefix_fallback_is_deterministic(self):
        """No routing_key but token fallback enabled: identical leading tokens
        must co-locate."""
        prefix = list(range(4096))
        ranks = set()
        for _ in range(10):
            c = _make_affinity_controller(dp_size=8)
            c.prefix_affinity_scheduler(_areq(input_ids=prefix))
            ranks.add(_dispatched_rank(c))
        self.assertEqual(len(ranks), 1)

    def test_keyless_with_token_fallback_disabled_uses_load_balancer(self):
        """No key + token fallback disabled -> defer to the load-aware fallback
        (total_tokens picks the least-loaded rank) rather than pinning."""
        ctl = _make_affinity_controller(
            dp_size=4, fallback="total_tokens", disable_token_fallback=True
        )
        ctl.dp_budget.total_tokens = [5, 1, 3, 4]
        ctl.dp_budget.total_requests = [0, 0, 0, 0]
        ctl.prefix_affinity_scheduler(_areq(input_ids=[]))
        ctl.workers[1].send_pyobj.assert_called_once()

    def test_external_routed_dp_rank_bypasses_affinity(self):
        ctl = _make_affinity_controller(dp_size=4)
        ctl.prefix_affinity_scheduler(
            _areq(routed_dp_rank=2, routing_key="ignored", input_ids=[1, 2])
        )
        ctl.workers[2].send_pyobj.assert_called_once()
        self.assertEqual(
            ctl.dp_budget.total_requests,
            [0, 0, 0, 0],
            "external routing must not touch affinity budget",
        )


if __name__ == "__main__":
    unittest.main()
