"""Dynamic PD gates invalidate peer-compaction credit without allocator writes."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_swa_pool,
    build_tri_pool,
    reset_context,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDynamicGateCapacity(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def assert_dynamic_views(self, allocator, state):
        members = allocator._move_gate_targets()
        immediate = [x.available_size() for x in members]
        epochs = [x._chain_capacity_epoch() for x in members]
        observed = []
        for opened in (True, False, True, False, True):
            state["open"] = opened
            self.assertEqual(allocator.verify_byte_accounting(), [])
            values = []
            with patch.object(
                allocator, "ensure_capacity", side_effect=AssertionError("query moved")
            ):
                for x, original in zip(members, immediate):
                    actual = x.schedulable_available_size()
                    expected = x._available_tokens(
                        extra_gap_bytes=x._peer_drainable_hole_bytes()
                    )
                    self.assertEqual(actual, expected)
                    self.assertEqual(x.available_size(), original)
                    values.append(actual)
            self.assertEqual(allocator.verify_byte_accounting(), [])
            self.assertEqual(epochs, [x._chain_capacity_epoch() for x in members])
            observed.append(values)
        self.assertEqual(observed[0], observed[2])
        self.assertEqual(observed[1], observed[3])
        self.assertNotEqual(observed[0], observed[1])

    def test_two_pool_public_gate_transitions(self):
        for ps in (1, 4, 16):
            for initially_open in (False, True):
                with self.subTest(page_size=ps, initially_open=initially_open):
                    a, _ = build_swa_pool(occupancy=0, lazy=True, page_size=ps)
                    slots = a.alloc(96 * ps)
                    a.free_swa(slots[: 40 * ps])
                    # Prime ungated memo, then install the public gate.
                    a.full_attn_allocator.schedulable_available_size()
                    state = {"open": initially_open}
                    a.set_disagg_move_gate(lambda: state["open"])
                    x = a.full_attn_allocator
                    self.assertEqual(
                        x.schedulable_available_size(),
                        x._available_tokens(x._peer_drainable_hole_bytes()),
                    )
                    self.assert_dynamic_views(a, state)
                    a.set_disagg_move_gate(lambda: False)
                    self.assertEqual(
                        x.schedulable_available_size(), x._available_tokens()
                    )

    def test_three_pool_end_float_and_transparent_peer(self):
        for ps in (1, 4, 16):
            for empty_float in (False, True):
                with self.subTest(page_size=ps, empty_float=empty_float):
                    b, a, _ = build_tri_pool(
                        lazy=True, page_size=ps, state_cache=False, temporal=(1, 4, 8)
                    )
                    states = a.mamba_allocator.alloc(8)
                    slots = a.alloc(80 * ps)
                    a.mamba_allocator.free(states[1:4])
                    if empty_float:
                        a.free_swa(slots)
                    else:
                        a.free(slots[10 * ps : 11 * ps])
                    state = {"open": True}
                    a.set_disagg_move_gate(lambda: state["open"])
                    self.assert_dynamic_views(a, state)

    def test_real_prefill_queue_gate(self):
        from sglang.srt.disaggregation.utils import (
            DisaggregationMode,
            unified_memory_disagg_move_gate,
        )

        a, _ = build_swa_pool(occupancy=0, lazy=True)
        slots = a.alloc(96)
        a.free_swa(slots[:40])
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_inflight_queue=[],
            disagg_prefill_pending_chunk_rids=set(),
        )
        a.set_disagg_move_gate(unified_memory_disagg_move_gate(scheduler))
        x = a.full_attn_allocator
        opened = x.schedulable_available_size()
        scheduler.disagg_prefill_pending_chunk_rids.add("request")
        closed = x.schedulable_available_size()
        self.assertLess(closed, opened)
        self.assertEqual(closed, x._available_tokens())
        scheduler.disagg_prefill_pending_chunk_rids.clear()
        self.assertEqual(x.schedulable_available_size(), opened)
        self.assertEqual(a.verify_byte_accounting(), [])

    def test_ungated_view_keeps_its_memo(self):
        a, _ = build_swa_pool(occupancy=0, lazy=True)
        slots = a.alloc(96)
        a.free_swa(slots[:40])
        x = a.full_attn_allocator
        expected = x._available_tokens(x._peer_drainable_hole_bytes())
        with patch.object(x, "_available_tokens", wraps=x._available_tokens) as compute:
            self.assertEqual(x.schedulable_available_size(), expected)
            self.assertEqual(x.schedulable_available_size(), expected)
            self.assertEqual(compute.call_count, 1)

    def test_stable_gate_reuses_memo_and_flips_refresh_it(self):
        """Stable PD gates must not force a fresh capacity calculation per query."""
        a, _ = build_swa_pool(occupancy=0, lazy=True)
        slots = a.alloc(96)
        a.free_swa(slots[:40])
        state = {"open": True}
        a.set_disagg_move_gate(lambda: state["open"])
        x = a.full_attn_allocator
        with patch.object(x, "_available_tokens", wraps=x._available_tokens) as compute:
            opened = x.schedulable_available_size()
            for _ in range(1999):
                self.assertEqual(x.schedulable_available_size(), opened)
            self.assertEqual(compute.call_count, 1)
        epoch = x._chain_capacity_epoch()
        for flip in range(100):
            state["open"] = bool(flip % 2)
            expected = x._available_tokens(x._peer_drainable_hole_bytes())
            with patch.object(
                x, "_available_tokens", wraps=x._available_tokens
            ) as compute:
                self.assertEqual(x.schedulable_available_size(), expected)
                self.assertEqual(x.schedulable_available_size(), expected)
                self.assertEqual(compute.call_count, 1)
            self.assertEqual(x._chain_capacity_epoch(), epoch)
            self.assertEqual(x._capacity_memo_violations(), [])

    def test_gated_memo_diagnostic_detects_untracked_write(self):
        """The gate-key cache must retain the missed-epoch diagnostic under PD."""
        a, _ = build_swa_pool(occupancy=0, lazy=True)
        slots = a.alloc(96)
        a.free_swa(slots[:40])
        a.set_disagg_move_gate(lambda: True)
        x = a.full_attn_allocator
        peer = a.swa_attn_allocator
        cached = x.schedulable_available_size()
        epoch = x._chain_capacity_epoch()
        # Inject a write bypassing the descriptor, the failure this check guards.
        peer.__dict__["_free_phys_pages"] = peer._free_phys_pages[:0]
        self.assertEqual(x._chain_capacity_epoch(), epoch)
        self.assertNotEqual(cached, x._available_tokens(x._peer_drainable_hole_bytes()))
        self.assertTrue(
            any(
                "stale schedulable_available_size" in e
                for e in x._capacity_memo_violations()
            )
        )

    def test_float_signature_distinguishes_asymmetric_gates(self):
        """Closing opposite sides must not alias a single combined gate boolean."""
        _, a, _ = build_tri_pool(lazy=True, state_cache=False, temporal=(1, 4, 8))
        states = a.mamba_allocator.alloc(8)
        slots = a.alloc(80)
        a.mamba_allocator.free(states[1:4])
        a.free_full(slots[10:11])
        state = {"low": True, "high": True}
        x = a.swa_attn_allocator
        x.low_peer.disagg_move_gate = lambda: state["low"]
        x.high_peer.disagg_move_gate = lambda: state["high"]
        epoch = x._chain_capacity_epoch()
        observed = {}
        for low, high in ((False, True), (True, False), (False, False), (True, True)):
            state.update(low=low, high=high)
            expected = x._available_tokens(x._peer_drainable_hole_bytes())
            observed[low, high] = x.schedulable_available_size()
            self.assertEqual(observed[low, high], expected)
            self.assertEqual(x._chain_capacity_epoch(), epoch)
            self.assertEqual(x._capacity_memo_violations(), [])
        self.assertNotEqual(observed[False, True], observed[True, False])

    def test_neighbor_memo_tracks_float_becoming_transparent(self):
        """Freeing the float must invalidate the end's cached blocking neighbor."""
        _, a, _ = build_tri_pool(lazy=True, state_cache=False, temporal=(1, 4, 8))
        states = a.mamba_allocator.alloc(8)
        slots = a.alloc(80)
        a.mamba_allocator.free(states[1:4])
        a.set_disagg_move_gate(lambda: True)
        x = a.full_attn_allocator
        self.assertIs(x._growth_side_neighbor(), a.swa_attn_allocator)
        x.schedulable_available_size()
        a.free_swa(slots)
        self.assertIs(x._growth_side_neighbor(), a.mamba_allocator)
        self.assertEqual(
            x.schedulable_available_size(),
            x._available_tokens(x._peer_drainable_hole_bytes()),
        )
        self.assertEqual(a.verify_byte_accounting(), [])

    def test_independent_pd_and_host_gates_refresh_memo(self):
        for ps in (1, 4, 16):
            with self.subTest(page_size=ps):
                a, _ = build_swa_pool(occupancy=0, lazy=True, page_size=ps)
                slots = a.alloc(96 * ps)
                a.free_swa(slots[: 40 * ps])
                state = {"pd": True, "host": True}
                a.set_disagg_move_gate(lambda: state["pd"])
                a.set_host_transfer_move_gate(lambda: state["host"])
                x = a.full_attn_allocator
                epoch = x._chain_capacity_epoch()
                opened = x.schedulable_available_size()
                for pd, host in (
                    (True, False),
                    (False, False),
                    (False, True),
                    (True, True),
                ):
                    state.update(pd=pd, host=host)
                    expected = x._available_tokens(x._peer_drainable_hole_bytes())
                    self.assertEqual(x.schedulable_available_size(), expected)
                    self.assertEqual(expected == opened, pd and host)
                    with patch.object(
                        x,
                        "_available_tokens",
                        side_effect=AssertionError("stable gate recomputed"),
                    ):
                        self.assertEqual(x.schedulable_available_size(), expected)
                    self.assertEqual(x._chain_capacity_epoch(), epoch)
                    self.assertEqual(x._capacity_memo_violations(), [])

    def test_float_opposite_gate_owners_do_not_alias(self):
        _, a, _ = build_tri_pool(lazy=True, state_cache=False, temporal=(1, 4, 8))
        states = a.mamba_allocator.alloc(8)
        slots = a.alloc(80)
        a.mamba_allocator.free(states[1:4])
        a.free_full(slots[10:11])
        x = a.swa_attn_allocator
        state = {"low": True, "high": True}
        x.low_peer.host_transfer_move_gate = lambda: state["low"]
        x.high_peer.disagg_move_gate = lambda: state["high"]
        epoch = x._chain_capacity_epoch()
        observed = {}
        for low, high in ((False, True), (True, False), (False, False), (True, True)):
            state.update(low=low, high=high)
            expected = x._available_tokens(x._peer_drainable_hole_bytes())
            observed[low, high] = x.schedulable_available_size()
            self.assertEqual(observed[low, high], expected)
            self.assertEqual(x._chain_capacity_epoch(), epoch)
            self.assertEqual(x._capacity_memo_violations(), [])
        self.assertNotEqual(observed[False, True], observed[True, False])

    def test_float_neighbor_walk_without_epoch_is_repeatable(self):
        """A non-cacheable stub chain must not fail on its second neighbor lookup."""
        _, a, _ = build_tri_pool(lazy=True, state_cache=False, temporal=(1, 4, 8))
        x = a.swa_attn_allocator
        with patch.object(
            x, "_chain_capacity_epoch", side_effect=AttributeError("stub")
        ):
            for side, expected in (("low", x.low_peer), ("high", x.high_peer)):
                for _ in range(2):
                    self.assertIs(x._side_capacity_neighbor(side), expected)


if __name__ == "__main__":
    unittest.main()
