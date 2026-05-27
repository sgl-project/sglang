"""Unit tests for the prefix/extend split + cp_full_out_cache_loc helper
(DESIGN_kv_reshard.md §6, Part 4c).

Verifies that ``cp_alloc_forward_transient`` separately tracks prefix-range
and extend-range scatter coordinates, and that
``cp_build_full_out_cache_loc`` returns the union of owned-permanent and
non-owned-transient row IDs for the current step's new tokens in canonical
batch order.

Pure-Python / CPU tests using a fake allocator.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

import unittest
from types import SimpleNamespace
from typing import List, Optional

import torch

from sglang.srt.layers.utils.cp_transient import (
    CpTransientState,
    cp_alloc_forward_transient,
    cp_build_full_out_cache_loc,
    cp_free_forward_transient,
)
from sglang.test.test_utils import CustomTestCase


class _FakeAllocator:
    def __init__(self, start: int = 100, capacity: int = 1024):
        self._free: List[int] = list(range(start, start + capacity))

    @property
    def available(self) -> int:
        return len(self._free)

    def available_size(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> Optional[torch.Tensor]:
        if n > len(self._free):
            return None
        out = torch.tensor(self._free[:n], dtype=torch.int64)
        self._free = self._free[n:]
        return out

    def free(self, indices: torch.Tensor) -> None:
        self._free.extend(int(x) for x in indices.tolist())


def _make_forward_batch(
    cp_owner_per_pages,
    extend_prefix_lens_cpu,
    extend_seq_lens_cpu,
    req_pool_indices,
    num_req_rows: int = 8,
    max_context: int = 64,
    preset_owned_slots=None,
):
    """Tiny ForwardBatch stub. ``preset_owned_slots`` is an optional
    mapping ``{(req_pool_idx, position): slot_id}`` for pre-populating
    req_to_token at owned positions (simulating the scheduler's
    allocator alloc that runs before init_forward_metadata)."""
    req_to_token = torch.zeros((num_req_rows, max_context), dtype=torch.int64)
    if preset_owned_slots:
        for (r, p), s in preset_owned_slots.items():
            req_to_token[r, p] = s
    pool = SimpleNamespace(req_to_token=req_to_token)
    return SimpleNamespace(
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64),
        req_to_token_pool=pool,
        cp_transient=CpTransientState(owner_per_pages=cp_owner_per_pages),
    )


class TestPrefixExtendSplit(CustomTestCase):
    def test_split_fields_populated_when_prefix_present(self):
        # cp_size=2, cp_rank=0, page_size=2.
        # Request s=0: 4 pages, owners [0,1,0,1]; prefix=4 (pages 0..1, positions 0..3),
        # extend=4 (pages 2..3, positions 4..7).
        # Owned by self (rank 0): pages 0 (pos 0..1) and 2 (pos 4..5).
        # Non-owned: page 1 (pos 2..3) and page 3 (pos 6..7).
        #   -> prefix non-owned: pos 2..3 (2 rows)
        #   -> extend non-owned: pos 6..7 (2 rows)
        owner = torch.tensor([0, 1, 0, 1], dtype=torch.int8)
        fb = _make_forward_batch(
            cp_owner_per_pages=[owner],
            extend_prefix_lens_cpu=[4],
            extend_seq_lens_cpu=[4],
            req_pool_indices=[3],
        )
        allocator = _FakeAllocator(start=500, capacity=32)

        dropped = cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
        self.assertEqual(dropped, [])

        # Prefix subset: 2 rows at positions 2, 3.
        self.assertIsNotNone(fb.cp_transient.prefix_rows)
        self.assertEqual(fb.cp_transient.prefix_position_indices.tolist(), [2, 3])
        self.assertEqual(fb.cp_transient.prefix_req_indices.tolist(), [3, 3])
        self.assertEqual(fb.cp_transient.prefix_rows.numel(), 2)

        # Union: 4 rows covering prefix (positions 2,3) and extend (positions 6,7).
        self.assertEqual(fb.cp_transient.rows.numel(), 4)
        self.assertEqual(fb.cp_transient.position_indices.tolist(), [2, 3, 6, 7])
        self.assertEqual(fb.cp_transient.req_indices.tolist(), [3, 3, 3, 3])

        # Verify scatter into req_to_token at all 4 coords.
        rtt = fb.req_to_token_pool.req_to_token
        for pos in (2, 3, 6, 7):
            self.assertNotEqual(rtt[3, pos].item(), 0)

    def test_only_extend_when_prefix_empty(self):
        # prefix=0, extend=4 -> all non-owned positions are extend-range; the
        # prefix subset stays None and the union covers everything.
        owner = torch.tensor([0, 1], dtype=torch.int8)
        fb = _make_forward_batch(
            cp_owner_per_pages=[owner],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[4],
            req_pool_indices=[5],
        )
        allocator = _FakeAllocator(start=200, capacity=8)

        cp_alloc_forward_transient(fb, allocator, cp_rank=1, page_size=2)
        self.assertIsNone(fb.cp_transient.prefix_rows)
        self.assertIsNotNone(fb.cp_transient.rows)
        self.assertEqual(fb.cp_transient.position_indices.tolist(), [0, 1])

    def test_only_prefix_when_extend_zero(self):
        # extend=0 -> no positions to scan for extend; prefix=8 with non-owned positions.
        owner = torch.tensor([0, 1, 0, 1], dtype=torch.int8)
        fb = _make_forward_batch(
            cp_owner_per_pages=[owner],
            extend_prefix_lens_cpu=[8],
            extend_seq_lens_cpu=[0],
            req_pool_indices=[2],
        )
        allocator = _FakeAllocator(start=400, capacity=16)

        cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
        # All non-owned positions (2,3,6,7) fall in prefix range -> prefix subset has 4 rows.
        self.assertIsNotNone(fb.cp_transient.prefix_rows)
        self.assertEqual(fb.cp_transient.prefix_position_indices.tolist(), [2, 3, 6, 7])
        # Union equals the prefix subset when there are no extend positions.
        self.assertEqual(fb.cp_transient.rows.numel(), 4)


class TestBuildFullOutCacheLoc(CustomTestCase):
    def test_reads_union_from_req_to_token(self):
        # cp_size=2, cp_rank=0, page_size=2.
        # Request s=0: owners [0,1,0,1]; prefix=4, extend=4.
        # Pre-populate owned-permanent slot IDs at positions 0..1 (page 0) and 4..5 (page 2):
        #   req_to_token[3, 0..1] = [10, 11]
        #   req_to_token[3, 4..5] = [12, 13]
        # After cp_alloc_forward_transient, transient slot IDs land at
        # non-owned positions 2..3 (prefix) and 6..7 (extend).
        owner = torch.tensor([0, 1, 0, 1], dtype=torch.int8)
        fb = _make_forward_batch(
            cp_owner_per_pages=[owner],
            extend_prefix_lens_cpu=[4],
            extend_seq_lens_cpu=[4],
            req_pool_indices=[3],
            preset_owned_slots={
                (3, 0): 10,
                (3, 1): 11,
                (3, 4): 12,
                (3, 5): 13,
            },
        )
        allocator = _FakeAllocator(start=600, capacity=32)
        cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)

        # cp_full_out_cache_loc should be the slice [prefix_len, prefix_len+extend_len)
        # = positions 4..7 -> [12, 13, <transient_6>, <transient_7>].
        loc = cp_build_full_out_cache_loc(fb)
        self.assertIsNotNone(loc)
        self.assertEqual(loc.numel(), 4)
        # First two are the pre-populated owned slot IDs.
        self.assertEqual(loc[0].item(), 12)
        self.assertEqual(loc[1].item(), 13)
        # Last two are the transient rows scattered into req_to_token at positions 6, 7.
        rtt = fb.req_to_token_pool.req_to_token
        self.assertEqual(loc[2].item(), rtt[3, 6].item())
        self.assertEqual(loc[3].item(), rtt[3, 7].item())

    def test_returns_none_when_no_extend(self):
        fb = _make_forward_batch(
            cp_owner_per_pages=[None],
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=None,
            req_pool_indices=[0],
        )
        self.assertIsNone(cp_build_full_out_cache_loc(fb))

    def test_multi_request_concatenates_in_batch_order(self):
        # Two requests, both contributing extend positions.
        # Req 0 (pool_idx=4): owner=[0,1], prefix=0, extend=4 -> positions 0..3.
        # Req 1 (pool_idx=5): owner=[1,0], prefix=0, extend=4 -> positions 0..3.
        # Pre-populate owned slots; transient fills the rest.
        fb = _make_forward_batch(
            cp_owner_per_pages=[
                torch.tensor([0, 1], dtype=torch.int8),
                torch.tensor([1, 0], dtype=torch.int8),
            ],
            extend_prefix_lens_cpu=[0, 0],
            extend_seq_lens_cpu=[4, 4],
            req_pool_indices=[4, 5],
            preset_owned_slots={
                # Req 0: page 0 owned by rank 0 -> positions 0,1.
                (4, 0): 20,
                (4, 1): 21,
                # Req 1: page 1 owned by rank 0 -> positions 2,3.
                (5, 2): 30,
                (5, 3): 31,
            },
        )
        allocator = _FakeAllocator(start=700, capacity=32)
        cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)

        loc = cp_build_full_out_cache_loc(fb)
        self.assertIsNotNone(loc)
        # 4 (req 0) + 4 (req 1) = 8 rows in canonical batch order.
        self.assertEqual(loc.numel(), 8)
        # Req 0 contribution: [20, 21, <transient_2>, <transient_3>].
        self.assertEqual(loc[0].item(), 20)
        self.assertEqual(loc[1].item(), 21)
        # Req 1 contribution: [<transient_0>, <transient_1>, 30, 31].
        self.assertEqual(loc[6].item(), 30)
        self.assertEqual(loc[7].item(), 31)


class TestFreeClearsAllSplits(CustomTestCase):
    def test_free_resets_split_and_full_fields(self):
        owner = torch.tensor([0, 1, 0, 1], dtype=torch.int8)
        fb = _make_forward_batch(
            cp_owner_per_pages=[owner],
            extend_prefix_lens_cpu=[4],
            extend_seq_lens_cpu=[4],
            req_pool_indices=[3],
        )
        allocator = _FakeAllocator(start=800, capacity=32)
        cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
        fb.cp_transient.full_out_cache_loc = cp_build_full_out_cache_loc(fb)

        # All fields populated.
        self.assertIsNotNone(fb.cp_transient.rows)
        self.assertIsNotNone(fb.cp_transient.prefix_rows)
        self.assertIsNotNone(fb.cp_transient.full_out_cache_loc)

        cp_free_forward_transient(fb, allocator)

        # All fields cleared.
        self.assertIsNone(fb.cp_transient.rows)
        self.assertIsNone(fb.cp_transient.req_indices)
        self.assertIsNone(fb.cp_transient.position_indices)
        self.assertIsNone(fb.cp_transient.prefix_rows)
        self.assertIsNone(fb.cp_transient.prefix_req_indices)
        self.assertIsNone(fb.cp_transient.prefix_position_indices)
        self.assertIsNone(fb.cp_transient.full_out_cache_loc)


if __name__ == "__main__":
    unittest.main()
