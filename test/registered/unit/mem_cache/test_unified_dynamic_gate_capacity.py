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


if __name__ == "__main__":
    unittest.main()
