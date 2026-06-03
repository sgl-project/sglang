"""Regression test: scheduler waiting/running-timeout aborts must be TP-rank
consistent.

`_abort_on_waiting_timeout` / `_abort_on_running_timeout` previously decided which
requests to abort using each rank's own `time.perf_counter()` and mutated the local
`waiting_queue` / `running_batch` with no cross-rank sync. With no per-iteration
barrier forcing the scheduler ranks to agree on batch composition, a request whose
wait straddles the deadline got aborted on some ranks but not others -> divergent
batch composition at the same forward step -> NCCL collective shape mismatch ->
permanent TP deadlock (observed in production with TP=8 + SGLANG_REQ_WAITING_TIMEOUT).

The fix decides the timed-out set on the dp_tp_group entry rank and broadcasts the
rid list, so every rank aborts the SAME requests. These tests pin that invariant
WITHOUT real GPUs by stubbing the scheduler and modeling broadcast_object_list.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

GROUP_SIZE = 8
TIMEOUT_S = 10.0
# Entry rank's authoritative clock. deadline = LEADER_CLOCK - TIMEOUT_S = 990.0
LEADER_CLOCK = 1000.0

# (rid, entry_time): A long-expired, B straddles the 990.0 deadline, C fresh.
ENTRIES = [
    ("A", 985.000),  # elapsed 15.0  -> expired on any nearby clock
    ("B", 989.999),  # elapsed ~10.0 -> straddles the deadline across skewed clocks
    ("C", 995.000),  # elapsed  5.0  -> never expired
]
# Clocks a BUGGY per-rank implementation would each read; B flips between rank 0
# (1000.000 -> abort B) and rank 1 (999.998 -> keep B).
NAIVE_PER_RANK_CLOCKS = [1000.000, 999.998, 1000.001, 999.997, 1000.002, 999.999]


def _make_fake_broadcast_object_list():
    """Model torch.distributed.broadcast_object_list: the src rank's object wins and
    is written in-place into every participant's list. Tests call the entry rank
    (rank_in_group == 0) first, so the first call captures the authoritative payload."""
    state = {"payload": None, "seen": False}

    def fake(object_list, src=0, group=None):
        if not state["seen"]:
            state["payload"] = object_list[0]
            state["seen"] = True
        object_list[0] = state["payload"]

    return fake


class _Req:
    """Minimal hashable request stub. SimpleNamespace is unhashable, but the abort
    path removes requests via `set(deleted_reqs)` membership on the real (hashable)
    Req objects, so the stub must be hashable too -- otherwise it raises
    `TypeError: unhashable type` on the removal step instead of exercising the fix."""

    def __init__(self, rid, time_stats):
        self.rid = rid
        self.time_stats = time_stats


def _waiting_req(rid, entry_time):
    return _Req(rid, SimpleNamespace(wait_queue_entry_time=entry_time))


def _running_req(rid, forward_entry_time):
    r = _Req(rid, SimpleNamespace(forward_entry_time=forward_entry_time))
    r.to_finish = None
    r.finished = lambda: False
    return r


def _rank_scheduler(rank_in_group, *, world_size=GROUP_SIZE):
    s = Scheduler.__new__(Scheduler)
    s.dp_tp_group = SimpleNamespace(rank_in_group=rank_in_group, first_rank=0)
    s.dp_tp_cpu_group = None
    s._dp_tp_group_world_size = lambda: world_size
    s.enable_hicache_storage = False
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    s.tree_cache = MagicMock()
    return s


class TestAbortTimeoutTPConsistency(CustomTestCase):
    def test_waiting_timeout_aborts_identical_rids_across_ranks(self):
        from sglang.srt.environ import envs

        fake = _make_fake_broadcast_object_list()
        schedulers = [_rank_scheduler(r) for r in range(GROUP_SIZE)]
        for s in schedulers:
            s.waiting_queue = [_waiting_req(rid, t) for rid, t in ENTRIES]

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(TIMEOUT_S), patch(
            "torch.distributed.broadcast_object_list", fake
        ), patch(
            "sglang.srt.managers.scheduler.time.perf_counter", return_value=LEADER_CLOCK
        ):
            for s in schedulers:  # entry rank (index 0) first
                Scheduler._abort_on_waiting_timeout(s)

        survivor_sets = [{r.rid for r in s.waiting_queue} for s in schedulers]
        for survivors in survivor_sets:
            self.assertEqual(survivors, {"C"})  # A and B aborted on every rank
        self.assertEqual(len(set(map(frozenset, survivor_sets))), 1)

    def test_running_timeout_marks_identical_rids_across_ranks(self):
        from sglang.srt.environ import envs

        fake = _make_fake_broadcast_object_list()
        schedulers = [_rank_scheduler(r) for r in range(GROUP_SIZE)]
        for s in schedulers:
            reqs = [_running_req(rid, t) for rid, t in ENTRIES]
            s.running_batch = SimpleNamespace(is_empty=lambda: False, reqs=reqs)

        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(TIMEOUT_S), patch(
            "torch.distributed.broadcast_object_list", fake
        ), patch(
            "sglang.srt.managers.scheduler.time.perf_counter", return_value=LEADER_CLOCK
        ):
            for s in schedulers:
                Scheduler._abort_on_running_timeout(s)

        aborted_sets = [
            {r.rid for r in s.running_batch.reqs if r.to_finish is not None}
            for s in schedulers
        ]
        for aborted in aborted_sets:
            self.assertEqual(aborted, {"A", "B"})
        self.assertEqual(len(set(map(frozenset, aborted_sets))), 1)

    def test_single_rank_aborts_locally_without_broadcast(self):
        from sglang.srt.environ import envs

        s = _rank_scheduler(0, world_size=1)
        s.waiting_queue = [_waiting_req(rid, t) for rid, t in ENTRIES]

        def _boom(*a, **k):
            raise AssertionError("broadcast_object_list called with world_size == 1")

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(TIMEOUT_S), patch(
            "torch.distributed.broadcast_object_list", _boom
        ), patch(
            "sglang.srt.managers.scheduler.time.perf_counter", return_value=LEADER_CLOCK
        ):
            Scheduler._abort_on_waiting_timeout(s)

        self.assertEqual({r.rid for r in s.waiting_queue}, {"C"})

    def test_negative_control_naive_per_rank_clocks_would_diverge(self):
        # Proves the chosen entries/clocks straddle the deadline: the OLD per-rank
        # decision over NAIVE_PER_RANK_CLOCKS yields different abort sets across ranks.
        def naive_abort_set(clock):
            deadline = clock - TIMEOUT_S
            return {rid for rid, t in ENTRIES if 0 < t < deadline}

        naive_sets = [naive_abort_set(c) for c in NAIVE_PER_RANK_CLOCKS]
        self.assertGreater(
            len(set(map(frozenset, naive_sets))),
            1,
            "test setup failed to straddle the deadline; the fix would be untested",
        )
        self.assertEqual(naive_sets[0], {"A", "B"})  # clock 1000.000
        self.assertEqual(naive_sets[1], {"A"})  # clock 999.998

    def test_timeout_off_is_noop_and_does_not_broadcast(self):
        # No-regression / no-perf-impact for the common case: when the timeout is not
        # set, the function returns before any work and issues NO collective.
        from sglang.srt.environ import envs

        bcast = MagicMock()
        s = _rank_scheduler(0)
        original = [_waiting_req(rid, t) for rid, t in ENTRIES]
        s.waiting_queue = list(original)
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(-1.0), patch(
            "torch.distributed.broadcast_object_list", bcast
        ):
            Scheduler._abort_on_waiting_timeout(s)
        bcast.assert_not_called()
        self.assertEqual(s.waiting_queue, original)  # nothing aborted

    def test_running_timeout_empty_batch_still_enters_broadcast(self):
        # The collective must be gated only on replicated state (timeout + group
        # size), never on per-rank running_batch.is_empty(). With an empty batch the
        # broadcast still runs so ranks cannot diverge on entering the collective.
        from sglang.srt.environ import envs

        def _inplace_noop(object_list, src=0, group=None):
            object_list[0] = object_list[0]  # empty set stays empty

        bcast = MagicMock(side_effect=_inplace_noop)
        s = _rank_scheduler(0)
        s.running_batch = SimpleNamespace(is_empty=lambda: True, reqs=[])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(TIMEOUT_S), patch(
            "torch.distributed.broadcast_object_list", bcast
        ), patch(
            "sglang.srt.managers.scheduler.time.perf_counter", return_value=LEADER_CLOCK
        ):
            Scheduler._abort_on_running_timeout(s)
        bcast.assert_called_once()  # collective reached despite empty batch


if __name__ == "__main__":
    unittest.main()
