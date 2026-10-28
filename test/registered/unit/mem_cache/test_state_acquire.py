"""Unit tests for binding a request's row and state slots together."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.kv.state_storage import StateSlotsExhausted, StateStorage
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.test.test_utils import CustomTestCase


def _table(rows: int) -> HybridReqToTokenPool:
    """The request table's rows and slot columns, with two tracking slots per
    request (the extra buffer, not lazy)."""
    table = object.__new__(HybridReqToTokenPool)
    ReqToTokenPool.__init__(
        table, size=rows, max_context_len=8, device="cpu", enable_memory_saver=False
    )
    table.enable_mamba_extra_buffer = True
    table.enable_mamba_extra_buffer_lazy = False
    table.mamba_ping_pong_track_buffer_size = 2
    table.mamba_initial_tracking_slots = 2
    table.req_index_to_mamba_index_mapping = torch.zeros(rows + 1, dtype=torch.int32)
    table.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
        (rows + 1, 2), dtype=torch.int64
    )
    return table


def _state(slots: int) -> StateStorage:
    return StateStorage(
        sub_pool="mamba",
        pool=SimpleNamespace(
            replayssm_spec_write_pos=None, replayssm_write_pos=None, size=slots
        ),
        slots=MambaSlotAllocator(slots, "cpu"),
        layer_map={0: 0},
        short_conv_pool=None,
        ngram_pool=None,
    )


class TestStateAcquire(CustomTestCase):
    def test_running_out_of_state_slots_gives_back_rows_and_slots(self):
        """Each request takes a state slot and two tracking slots. With slots
        for one request and a half, the second runs out after the first got
        all three: the rows bound for both and the slots handed out go back,
        so a failed admission leaves the pools as it found them."""
        table, state = _table(rows=4), _state(slots=5)
        reqs = [SimpleNamespace(kv=ReqKvInfo()) for _ in range(2)]

        with self.assertRaises(StateSlotsExhausted):
            acquire_rows(table, state, reqs)

        self.assertEqual(table.available_size(), 4)
        self.assertEqual(state.slots.available_size(), 5)
        for req in reqs:
            self.assertIsNone(req.kv.req_pool_idx)
            self.assertIsNone(req.kv.mamba_pool_idx)
            self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)

    def test_a_request_keeps_the_state_slot_it_holds(self):
        """A radix hit arrives holding its state slot: acquire binds a row and
        tracking slots, writes the held slot into the table, and takes no new
        state slot for it."""
        table, state = _table(rows=4), _state(slots=5)
        held = state.slots.alloc(1)
        req = SimpleNamespace(kv=ReqKvInfo(mamba_pool_idx=held[0]))

        (row,) = acquire_rows(table, state, [req])

        self.assertEqual(int(table.req_index_to_mamba_index_mapping[row]), int(held))
        self.assertEqual(state.slots.available_size(), 2)


if __name__ == "__main__":
    unittest.main()
