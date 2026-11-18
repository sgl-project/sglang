"""Request-row states of a unified pool: what a table reset frees and ends, and
the row counts by state the memory-pressure predicates read."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.role_switch import (
    _release_prefix_cache_for_role_switch,
)
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.mem_cache.kv.alloc.signals import NetworkTransfer, ReceiveHold
from sglang.srt.mem_cache.kv.requests import RequestTable, RowState
from sglang.srt.models.unlimited_ocr_v2_support.retained_window import (
    RetainedWindow,
)
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_test_unified_pools,
    setup_allocator_context,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _storage(*, max_num_reqs=4, pre_alloc_size=0):
    pools = build_test_unified_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=8,
        page_size=4,
        start_layer=0,
        end_layer=1,
        full_attention_layer_ids=[0],
        total_bytes=8192,
        max_num_reqs=max_num_reqs,
        decode_pre_alloc_size=pre_alloc_size,
    )
    return pools.token_to_kv_pool_allocator.storage


def _req(rid):
    return SimpleNamespace(rid=rid, kv=ReqKvInfo())


def _active(storage, rid):
    """A request bound to a fresh row, as admission or a PD decode
    preallocation binds it."""
    req = _req(rid)
    assert storage.acquire([req]) is not None
    return req


def _parked(storage):
    """A streaming session's row between turns."""
    slot = SessionSlot(kv=_active(storage, "turn").kv)
    storage.park_row(slot)
    return slot


def _streaming_session(storage):
    return StreamingSession(
        SimpleNamespace(
            req_to_token_pool=storage.requests,
            token_to_kv_pool_allocator=storage.allocator,
            page_size=4,
            logical_page_size=4,
            unlock=lambda lock: None,
            free_kv_row=lambda kv, ranges: None,
        )
    )


def _finished_turn(storage, session_id):
    turn = SimpleNamespace(
        rid=f"{session_id}-turn",
        session=SimpleNamespace(
            session_id=session_id, streaming=True, finish_req=lambda req: None
        ),
        kv=ReqKvInfo(),
        origin_input_ids=list(range(4)),
        output_ids=[],
        finished_reason=FINISH_LENGTH(length=0),
        finished_len=0,
        swa_branching_seqlen=None,
        lock=None,
    )
    turn.detach_kv = lambda: setattr(turn, "kv", ReqKvInfo())
    assert storage.acquire([turn]) is not None
    turn.kv.kv_committed_len = turn.kv.kv_allocated_len = 4
    return turn


class TestRowStates(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def test_rows_counted_by_state(self):
        table = RequestTable(
            size=4,
            pre_alloc_size=2,
            max_context_len=8,
            state_sub_pool=None,
            tracking=None,
            device="cpu",
            enable_memory_saver=False,
        )
        self.assertEqual(table.count(RowState.FREE), 6)
        a, b, c = table.alloc_rows(3)
        (beam,) = table.alloc_rows(1, state=RowState.HELD)
        table.park(b)
        table.hold(c)
        counts = {s: table.count(s) for s in RowState}
        self.assertEqual(
            counts,
            {
                RowState.FREE: 2,
                RowState.ACTIVE: 1,
                RowState.PARKED: 1,
                RowState.HELD: 2,
            },
        )
        table.unpark(b)
        table.unpark(b)  # every chunk of the next turn says so
        self.assertIs(table.state_of(b), RowState.ACTIVE)
        table.free_rows([a, b, c, beam])
        self.assertEqual(table.count(RowState.FREE), len(table.free_slots))
        self.assertEqual(table.count(RowState.FREE), 6)

    def test_double_release_of_a_parked_row_fails_loudly(self):
        storage = _storage()
        slot = _parked(storage)
        row = slot.kv.req_pool_idx
        storage.release_row(slot)
        self.assertIs(storage.requests.state_of(row), RowState.FREE)
        # A stale holder still naming the row: it must not go back twice.
        stale = SessionSlot(kv=ReqKvInfo())
        stale.kv.req_pool_idx = row
        with self.assertRaisesRegex(AssertionError, "is FREE"):
            storage.release_row(stale)
        with self.assertRaisesRegex(AssertionError, "is FREE"):
            storage.release_rows([row])
        self.assertEqual(storage.requests.free_slots.count(row), 1)


class TestHeldRowPredicates(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def test_preallocation_rows_do_not_hide_held_rows(self):
        """Regression: the held-row predicates give the same answer for the same
        live work with or without preallocation rows on the free list."""
        for pre_alloc_size in (0, 8):
            with self.subTest(pre_alloc_size=pre_alloc_size):
                storage = _storage(pre_alloc_size=pre_alloc_size)
                first, second = _active(storage, "a"), _active(storage, "b")
                self.assertTrue(storage.others_hold_rows(None))
                self.assertTrue(storage.others_hold_rows(first))
                storage.release_row(second)
                self.assertFalse(storage.others_hold_rows(first))
                self.assertTrue(storage.others_hold_rows(None))
                _parked(storage)
                self.assertFalse(storage.only_parked_rows_held())
                storage.release_row(first)
                self.assertTrue(storage.only_parked_rows_held())

    def test_pd_decode_preallocated_rows_count_as_held(self):
        """A PD decode node's preallocated receive row comes back when its
        transfer ends, so it counts as held, beside parked rows or alone."""
        storage = _storage(max_num_reqs=2, pre_alloc_size=4)
        receiving = [_active(storage, f"recv-{i}") for i in range(3)]
        self.assertTrue(storage.others_hold_rows(None))
        slot = _parked(storage)
        self.assertFalse(storage.only_parked_rows_held())
        for req in receiving:
            storage.release_row(req)
        self.assertTrue(storage.only_parked_rows_held())
        self.assertFalse(storage.others_hold_rows(None))
        storage.release_row(slot)
        self.assertFalse(storage.only_parked_rows_held())

    def test_beam_and_retained_window_rows_count_as_held(self):
        storage = _storage()
        (beam,) = storage.take_rows(1)
        self.assertIs(storage.requests.state_of(beam), RowState.HELD)
        self.assertTrue(storage.others_hold_rows(None))
        storage.release_rows([beam])
        window = RetainedWindow()
        window.kv = _active(storage, "window").kv
        storage.hold_row(window)
        self.assertTrue(storage.others_hold_rows(None))
        # A retained window is no request: its release ends no transfer.
        storage.release_row(window)
        self.assertFalse(storage.others_hold_rows(None))
        self.assertEqual(storage.requests.count(RowState.FREE), 4)


class TestResets(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def test_flushed_parked_rows_come_back_as_active_rows(self):
        """BUG REGRESSION. A flush frees open sessions' parked rows; new
        requests taking them must count them ACTIVE, not PARKED."""
        storage = _storage()
        session = _streaming_session(storage)
        for session_id in ("s0", "s1"):
            turn = _finished_turn(storage, session_id)
            session.take(turn)
            self.assertTrue(session.try_cache_finished_req(turn))
        self.assertEqual(storage.requests.count(RowState.PARKED), 2)
        self.assertTrue(storage.only_parked_rows_held())

        # The flush: the tree reset forgets the slots, then the table clears.
        session.slots.clear()
        storage.requests.clear()
        self.assertEqual(storage.requests.count(RowState.FREE), 4)
        self.assertEqual(storage.requests.count(RowState.PARKED), 0)

        rows = {_active(storage, f"new-{i}").kv.req_pool_idx for i in range(4)}
        self.assertEqual(len(rows), 4)
        self.assertEqual(storage.requests.count(RowState.ACTIVE), 4)
        self.assertTrue(storage.others_hold_rows(None))
        self.assertFalse(storage.only_parked_rows_held())

    def test_role_switch_reset_ends_transfers_and_receive_holds(self):
        """The reset a role switch (as a flush) runs frees every row, so every
        network transfer and receive hold keyed by a request ends with it."""
        storage = _storage(pre_alloc_size=4)
        sent, received = _active(storage, "sent"), _active(storage, "received")
        transfer, hold = NetworkTransfer(), ReceiveHold()
        storage._network_transfers[id(sent)] = (sent, [transfer])
        storage._receive_holds[id(received)] = (received, hold)
        storage.park_row(SessionSlot(kv=_active(storage, "turn").kv))
        resets = []
        scheduler = SimpleNamespace(
            disable_radix_cache=False,
            tree_cache=SimpleNamespace(reset=lambda: resets.append("tree")),
            req_to_token_pool=storage.requests,
            token_to_kv_pool_allocator=storage.allocator,
        )
        _release_prefix_cache_for_role_switch(scheduler)
        self.assertEqual(resets, ["tree"])
        self.assertTrue(transfer.released)
        self.assertTrue(hold.released)
        self.assertEqual(storage._network_transfers, {})
        self.assertEqual(storage._receive_holds, {})
        self.assertEqual(storage.requests.count(RowState.FREE), 8)
        self.assertFalse(storage.others_hold_rows(None))
        self.assertFalse(storage.only_parked_rows_held())

    def test_closing_a_parked_session_frees_its_row(self):
        """A session slot is no request: releasing its parked row ends no
        network transfer, and needs no request id."""
        storage = _storage()
        session = _streaming_session(storage)
        turn = _finished_turn(storage, "s")
        session.take(turn)
        self.assertTrue(session.try_cache_finished_req(turn))
        row = session.slots["s"].kv.req_pool_idx
        session.release_session("s")
        self.assertIs(storage.requests.state_of(row), RowState.FREE)
        self.assertEqual(storage.requests.count(RowState.FREE), 4)


class TestIdleRowCheck(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def test_idle_rows_are_free_parked_or_held(self):
        storage = _storage()
        check = SchedulerInvariantChecker._row_state_violation
        self.assertIsNone(check(storage.requests, 0))
        _parked(storage)
        self.assertIsNone(check(storage.requests, 1))
        self.assertIn("session_held", check(storage.requests, 0))
        _active(storage, "left-behind")
        self.assertIn("ACTIVE rows while idle", check(storage.requests, 1))


if __name__ == "__main__":
    unittest.main()
