"""DPBudget + DataParallelController dispatch tests.

The e2e counterpart, over the real scheduler load-report path, is
test/registered/disaggregation/test_disaggregation_dp_attention.py.

Fragility: scheduler tests bypass `DataParallelController.__init__` via
`__new__` and inject only the attrs the schedulers read (`workers`, `status`,
`_active_workers`, `round_robin_counter`, `dp_budget`). Update `_make_controller`
if a scheduler starts reading another attr. `maybe_external_dp_rank_routing`
is exercised as the real method, no mock. The refresh-throttle tests inject
`load_snapshot_reader` and `_last_refresh_time` on top of those.
"""

import time
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
    _consistent_hash,
    _select_consistent_hash_rank,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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
    ctl._active_workers = list(range(dp_size))
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


class TestRefreshLoadBudgetThrottle(CustomTestCase):
    @staticmethod
    def _controller_with_reader(dp_size, snapshots):
        ctl = _make_controller(dp_size)
        ctl.load_snapshot_reader = MagicMock()
        ctl.load_snapshot_reader.read_all.return_value = snapshots
        return ctl

    def test_throttled_refresh_spreads_a_burst_across_ranks(self):
        idle = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(4)]
        ctl = self._controller_with_reader(dp_size=4, snapshots=idle)
        # A refresh stamp in the future keeps every call inside the window, so
        # the burst runs entirely on speculative counters.
        ctl._last_refresh_time = time.perf_counter() + 3600.0

        for _ in range(8):
            ctl.refresh_load_budget()
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 100))

        ctl.load_snapshot_reader.read_all.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [200, 200, 200, 200],
            "speculative increments should spread the burst evenly",
        )
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(
                worker.send_pyobj.call_count, 2, f"worker {i} should get 2 of 8 reqs"
            )

    def test_refresh_outside_window_overwrites_speculative_increments(self):
        reported = [
            _load(dp_rank=0, timestamp=2.0, num_total_tokens=10),
            _load(dp_rank=1, timestamp=2.0, num_total_tokens=20),
        ]
        ctl = self._controller_with_reader(dp_size=2, snapshots=reported)
        ctl._last_refresh_time = 0.0  # window has long passed
        ctl.dp_budget.total_tokens = [999, 999]

        ctl.refresh_load_budget()

        ctl.load_snapshot_reader.read_all.assert_called_once()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [10, 20],
            "a fresh snapshot must replace the speculative state",
        )
        self.assertGreater(ctl._last_refresh_time, 0.0)

    def test_unchanged_snapshot_does_not_reset_the_burst(self):
        frozen = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(2)]
        ctl = self._controller_with_reader(dp_size=2, snapshots=frozen)
        ctl._last_refresh_time = 0.0
        ctl.refresh_load_budget()  # adopts timestamp 1.0

        for _ in range(4):
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 50))
        after_burst = list(ctl.dp_budget.total_tokens)

        ctl._last_refresh_time = 0.0  # let the next refresh through the throttle
        ctl.refresh_load_budget()  # same timestamp -> update_budget skips it

        self.assertEqual(
            after_burst, [100, 100], "burst should have spread over both ranks"
        )
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            after_burst,
            "a stale-timestamp snapshot must not wipe the speculative state",
        )


# ---------------------------------------------------------------------------
# consistent_hash load-balance method: pin + optional spill + optional repin.
# Most tests drive the extracted pure decision `_select_consistent_hash_rank`;
# a few drive `consistent_hash_scheduler` end to end for the wiring.
# ---------------------------------------------------------------------------


def _select(key, active, loads, *, spill=False, repin=True, gap=0.30, abs_floor=8192,
            group_size=8, pin=None, cap=200000):
    return _select_consistent_hash_rank(
        key,
        active,
        loads,
        enable_spill=spill,
        enable_repin=repin,
        gap_pct=gap,
        abs_floor=abs_floor,
        group_size=group_size,
        session_pin={} if pin is None else pin,
        session_pin_cap=cap,
    )


def _key_homing_to(rank, active):
    """A session key whose consistent-hash home (over `active`) is `rank`."""
    for i in range(1_000_000):
        k = f"k{i}"
        if active[_consistent_hash(k, len(active))] == rank:
            return k
    raise AssertionError(f"no key homing to {rank}")


class TestConsistentHashPin(CustomTestCase):
    def test_method_registered(self):
        self.assertIs(
            LoadBalanceMethod.from_str("consistent_hash"),
            LoadBalanceMethod.CONSISTENT_HASH,
        )

    def test_pin_is_deterministic(self):
        active = list(range(8))
        self.assertEqual(
            _select("sess-A", active, [0] * 8),
            _select("sess-A", active, [0] * 8),
        )

    def test_pin_distributes_across_all_ranks(self):
        active = list(range(8))
        seen = {_select(f"s{i}", active, [0] * 8) for i in range(4000)}
        self.assertEqual(seen, set(range(8)))

    def test_pin_only_selects_active_ranks(self):
        active = [1, 3, 5]
        for i in range(500):
            self.assertIn(_select(f"s{i}", active, [0] * 8), active)

    def test_no_active_rank_returns_none(self):
        self.assertIsNone(_select("x", [], [0] * 8))

    def test_pin_stable_regardless_of_load_when_spill_off(self):
        active = list(range(8))
        key = _key_homing_to(0, active)
        hot = [0] * 8
        hot[0] = 10**9  # home massively loaded
        self.assertEqual(_select(key, active, hot, spill=False), 0)


class TestConsistentHashSpill(CustomTestCase):
    def test_no_spill_when_balanced(self):
        active = list(range(8))
        key = _key_homing_to(3, active)
        self.assertEqual(_select(key, active, [5000] * 8, spill=True), 3)

    def test_spill_fires_under_imbalance(self):
        active = list(range(8))
        key = _key_homing_to(2, active)
        loads = [0] * 8
        loads[2] = 10**6  # home hot, others empty
        target = _select(key, active, loads, spill=True)
        self.assertNotEqual(target, 2)
        self.assertIn(target, active)

    def test_abs_floor_prevents_trivial_spill(self):
        # home relatively bigger than min(=0) but below the abs floor -> no spill.
        active = list(range(8))
        key = _key_homing_to(0, active)
        loads = [0] * 8
        loads[0] = 100  # < abs floor 8192
        self.assertEqual(_select(key, active, loads, spill=True), 0)

    def test_gap_threshold_boundary(self):
        active = list(range(8))
        key = _key_homing_to(0, active)
        # min=1000, gap=0.30, abs=0 -> threshold = 1300.
        just_below = [1000] * 8
        just_below[0] = 1300
        self.assertEqual(
            _select(key, active, just_below, spill=True, abs_floor=0), 0
        )
        just_above = [1000] * 8
        just_above[0] = 1301
        self.assertNotEqual(
            _select(key, active, just_above, spill=True, abs_floor=0), 0
        )

    def test_anti_herding_spreads_spills(self):
        # Many distinct sessions all homing to the same hot rank should not all
        # land on one spill target (the speculative bump spreads them).
        active = list(range(8))
        pin = {}
        loads = [0] * 8
        loads[0] = 50000  # hot home; others start empty, bumped as we spill
        targets = set()
        placed = 0
        for i in range(2000):
            k = f"h{i}"
            if active[_consistent_hash(k, 8)] != 0:
                continue  # only sessions that home to the hot rank spill
            targets.add(_select(k, active, loads, spill=True))
            placed += 1
            if placed >= 50:
                break
        self.assertGreater(len(targets), 1, "spills must spread across ranks")


class TestConsistentHashSpillGroups(CustomTestCase):
    def test_spill_never_crosses_node_group(self):
        # dp16, group_size 8 -> node groups [0..7], [8..15]. A rank in group 0
        # spills only within group 0, never to the (emptier) group 1.
        active = list(range(16))
        key = _key_homing_to(0, active)  # home 0, group 0
        loads = [0] * 16
        loads[0] = 10**6  # hot home
        for r in range(1, 8):
            loads[r] = 1000  # group-0 peers lightly loaded
        # group 1 (8..15) is empty but must be ignored
        target = _select(key, active, loads, spill=True, group_size=8)
        self.assertLess(target, 8, "spill must stay in the home rank's node group")

    def test_single_group_matches_global_spill(self):
        # dp8 with default group_size 8 -> one group -> spill considers all ranks.
        active = list(range(8))
        key = _key_homing_to(4, active)
        loads = [0] * 8
        loads[4] = 10**6
        target = _select(key, active, loads, spill=True, group_size=8)
        self.assertNotEqual(target, 4)


class TestConsistentHashRepin(CustomTestCase):
    def test_repin_on_is_stateless_and_returns_home(self):
        active = list(range(8))
        key = _key_homing_to(2, active)
        pin = {}
        hot = [0] * 8
        hot[2] = 10**6
        spilled = _select(key, active, hot, spill=True, repin=True, pin=pin)
        self.assertNotEqual(spilled, 2)
        self.assertEqual(pin, {}, "repin=on keeps no per-session state")
        # once the home rank drains, the same session returns to its CH home.
        self.assertEqual(
            _select(key, active, [0] * 8, spill=True, repin=True, pin=pin), 2
        )

    def test_repin_off_sticks_to_spilled_rank(self):
        active = list(range(8))
        key = _key_homing_to(2, active)
        pin = {}
        hot = [0] * 8
        hot[2] = 10**6
        spilled = _select(key, active, hot, spill=True, repin=False, pin=pin)
        self.assertNotEqual(spilled, 2)
        self.assertEqual(pin.get(key), spilled, "sticky mode records the spilled rank")
        # even after the CH home drains, a sticky session stays on the spilled rank.
        self.assertEqual(
            _select(key, active, [0] * 8, spill=True, repin=False, pin=pin), spilled
        )

    def test_sticky_map_is_bounded(self):
        active = list(range(8))
        pin = {}
        cap = 50
        for i in range(5000):
            k = f"sticky{i}"
            if active[_consistent_hash(k, 8)] != 0:
                continue  # only hot-home sessions spill and get pinned
            hot = [0] * 8
            hot[0] = 10**6
            _select(k, active, hot, spill=True, repin=False, pin=pin, cap=cap)
        self.assertLessEqual(len(pin), cap, "sticky map must stay bounded")


class TestConsistentHashSchedulerWiring(CustomTestCase):
    """End-to-end through consistent_hash_scheduler -> sock_send(worker)."""

    def _ctl(self, dp_size, **cfg):
        ctl = _make_controller(dp_size)
        ctl.enable_dp_spill = cfg.get("spill", False)
        ctl.enable_dp_repin = cfg.get("repin", True)
        ctl.dp_spill_gap_pct = cfg.get("gap", 0.30)
        ctl.dp_spill_abs = cfg.get("abs", 8192)
        ctl.dp_spill_group_size = cfg.get("group", 8)
        ctl._session_pin = {}
        ctl._session_pin_cap = 200000
        return ctl

    def _req(self, session_id=None, rid="r", routed_dp_rank=None):
        return SimpleNamespace(
            session_id=session_id, rid=rid, routed_dp_rank=routed_dp_rank
        )

    def test_routes_session_to_its_hash_home(self):
        ctl = self._ctl(8)
        ctl.consistent_hash_scheduler(self._req(session_id="abc"))
        home = _consistent_hash("abc", 8)
        ctl.workers[home].send_pyobj.assert_called_once()
        for i in range(8):
            if i != home:
                ctl.workers[i].send_pyobj.assert_not_called()

    def test_same_session_routes_to_same_worker(self):
        ctl = self._ctl(8)
        for _ in range(3):
            ctl.consistent_hash_scheduler(self._req(session_id="s"))
        self.assertEqual(ctl.workers[_consistent_hash("s", 8)].send_pyobj.call_count, 3)

    def test_routed_dp_rank_bypasses_consistent_hash(self):
        ctl = self._ctl(8)
        ctl.consistent_hash_scheduler(self._req(session_id="s", routed_dp_rank=5))
        ctl.workers[5].send_pyobj.assert_called_once()

    def test_falls_back_to_rid_without_session_id(self):
        ctl = self._ctl(8)
        ctl.consistent_hash_scheduler(self._req(session_id=None, rid="the-rid"))
        ctl.workers[_consistent_hash("the-rid", 8)].send_pyobj.assert_called_once()


if __name__ == "__main__":
    unittest.main()
