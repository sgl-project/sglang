"""The unified request table hands out rows as the static request pool does."""

import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.requests import RequestTable, StateTracking
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _table(*, size=8, pre_alloc_size=0, state=False):
    return RequestTable(
        size=size,
        pre_alloc_size=pre_alloc_size,
        max_context_len=16,
        state_sub_pool="mamba" if state else None,
        tracking=(
            StateTracking(extra_buffer=True, lazy=False, track_slots=2, initial_slots=2)
            if state
            else None
        ),
        device="cpu",
        enable_memory_saver=False,
    )


class _Kv:
    def __init__(self):
        self.req_pool_idx = None
        self.kv_allocated_len = 1

    @property
    def holds_kv(self):
        return self.req_pool_idx is not None


def _req():
    return SimpleNamespace(kv=_Kv())


class TestRequestTable(CustomTestCase):
    def test_rows_follow_the_static_pool(self):
        """The same alloc/free/clear sequence takes the same rows with the same
        generations from both tables."""
        rng = random.Random(0)
        table = _table()
        static = ReqToTokenPool(
            size=8, max_context_len=16, device="cpu", enable_memory_saver=False
        )
        for _ in range(400):
            op = rng.random()
            if op < 0.5:
                n = rng.randint(0, 3)
                self.assertEqual(table.alloc_rows(n), static.alloc_rows(n))
            elif op < 0.95:
                taken = sorted(set(range(1, 9)) - set(static.free_slots))
                if taken:
                    rows = rng.sample(taken, rng.randint(1, len(taken)))
                    table.free_rows(rows)
                    static.free_rows(rows)
            else:
                table.clear()
                static.clear()
            self.assertEqual(table.free_slots, static.free_slots)
            self.assertTrue(torch.equal(table.req_generation, static.req_generation))

    def test_alloc_binds_fresh_requests_only(self):
        table = _table()
        first, later = _req(), _req()
        rows = table.alloc([first])
        self.assertEqual(rows, [first.kv.req_pool_idx])
        rows = table.alloc([first, later])
        self.assertEqual(rows[0], first.kv.req_pool_idx)
        self.assertNotEqual(rows[1], rows[0])
        table.free(later)
        self.assertIsNone(later.kv.req_pool_idx)
        self.assertEqual(table.available_size(), 7)

    def test_preallocated_rows_extend_the_table_not_its_size(self):
        table = _table(size=4, pre_alloc_size=3)
        self.assertEqual(table.size, 4)
        self.assertEqual(table.available_size(), 7)
        self.assertEqual(table.req_to_token.shape, (8, 16))

    def test_state_columns_come_with_a_state_sub_pool(self):
        plain = _table()
        self.assertFalse(plain.holds_state_slots)
        with_state = _table(state=True)
        self.assertTrue(with_state.holds_state_slots)
        self.assertEqual(with_state.req_index_to_mamba_index_mapping.shape, (9,))
        self.assertEqual(
            with_state.req_index_to_mamba_ping_pong_track_buffer_mapping.shape, (9, 2)
        )
        with_state.req_index_to_mamba_index_mapping[3] = 5
        with_state.clear()
        self.assertEqual(int(with_state.req_index_to_mamba_index_mapping.sum()), 0)

    def test_donated_tracking_slot_leaves_the_row(self):
        table = _table(state=True)
        req = _req()
        table.alloc([req])
        req.kv.mamba_ping_pong_track_buffer = torch.tensor([4, 6])
        req.kv.mamba_ping_pong_empty = [False, False]
        req.kv.mamba_last_track_idx = 1
        donated = table.donate_mamba_ping_pong_slot(req, torch.tensor([9]))
        self.assertEqual(donated.tolist(), [6])
        row = table.req_index_to_mamba_ping_pong_track_buffer_mapping[
            req.kv.req_pool_idx
        ]
        self.assertEqual(row.tolist(), [4, 9])


if __name__ == "__main__":
    unittest.main()
