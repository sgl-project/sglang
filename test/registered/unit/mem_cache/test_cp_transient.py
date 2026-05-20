"""Unit tests for the CP-resharding transient-row helpers (Chunk A,
DESIGN_kv_reshard.md §6 dynamic variant).

Pure-Python / CPU tests using a fake allocator that mirrors
``TokenToKVPoolAllocator.alloc`` / ``free`` semantics — no GPU, no NCCL.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

import unittest
from typing import List, Optional

import torch

from sglang.srt.layers.utils.cp_transient import (
    compute_cp_non_owned_positions,
    cp_alloc_transient_rows,
    cp_free_transient_rows,
)
from sglang.test.test_utils import CustomTestCase


class _FakeAllocator:
    """Mirrors ``TokenToKVPoolAllocator.alloc`` (returns ``None`` on
    over-request) and ``free`` (accepts an int64 tensor of indices)."""

    def __init__(self, start: int = 1, capacity: int = 1024):
        self._free: List[int] = list(range(start, start + capacity))
        self.alloc_log: List[torch.Tensor] = []
        self.free_log: List[torch.Tensor] = []

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
        self.alloc_log.append(out.clone())
        return out

    def free(self, indices: torch.Tensor) -> None:
        self._free.extend(int(x) for x in indices.tolist())
        self.free_log.append(indices.clone())


def _make_req_to_token(num_reqs: int, max_context: int) -> torch.Tensor:
    """Initial req_to_token: all positions point to slot-0 sentinel."""
    return torch.zeros((num_reqs, max_context), dtype=torch.int64)


class TestComputeNonOwnedPositions(CustomTestCase):
    def test_balanced_owner_one_request(self):
        # 4 pages, owners [0, 1, 2, 3], page_size=2, cp_rank=0.
        # Positions 0..7. Owned by self: 0, 1 (page 0). Non-owned: 2..7.
        owner = torch.tensor([0, 1, 2, 3], dtype=torch.int8)
        req_idxs, positions = compute_cp_non_owned_positions(
            cp_owner_per_pages=[owner],
            prefix_lens=[0],
            extend_lens=[8],
            page_size=2,
            cp_rank=0,
            req_pool_indices=[7],
        )
        self.assertEqual(positions.tolist(), [2, 3, 4, 5, 6, 7])
        self.assertEqual(req_idxs.tolist(), [7, 7, 7, 7, 7, 7])

    def test_all_owned_returns_empty(self):
        owner = torch.tensor([0, 0, 0, 0], dtype=torch.int8)
        req_idxs, positions = compute_cp_non_owned_positions(
            cp_owner_per_pages=[owner],
            prefix_lens=[0],
            extend_lens=[8],
            page_size=2,
            cp_rank=0,
            req_pool_indices=[3],
        )
        self.assertEqual(positions.numel(), 0)
        self.assertEqual(req_idxs.numel(), 0)

    def test_partial_range_via_prefix_len(self):
        # 8 pages; cp_rank=1 owns pages 1, 3, 5, 7 (page % 2 == 1).
        owner = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int8)
        # Only inspect positions [4, 12): pages 2, 3, 4, 5.
        # cp_rank=1 owns pages 3 and 5. Non-owned positions: 4, 5, 8, 9.
        req_idxs, positions = compute_cp_non_owned_positions(
            cp_owner_per_pages=[owner],
            prefix_lens=[4],
            extend_lens=[8],
            page_size=2,
            cp_rank=1,
            req_pool_indices=[42],
        )
        self.assertEqual(positions.tolist(), [4, 5, 8, 9])
        self.assertEqual(req_idxs.tolist(), [42, 42, 42, 42])

    def test_multi_request_concatenation(self):
        # Two requests with different owners and different ranges.
        owner_a = torch.tensor([0, 1], dtype=torch.int8)  # 2 pages
        owner_b = torch.tensor([1, 0, 1], dtype=torch.int8)  # 3 pages
        req_idxs, positions = compute_cp_non_owned_positions(
            cp_owner_per_pages=[owner_a, owner_b],
            prefix_lens=[0, 0],
            extend_lens=[4, 6],
            page_size=2,
            cp_rank=0,
            req_pool_indices=[5, 9],
        )
        # Req 5 (owner_a, cp_rank=0): page 0 owned, page 1 not -> positions [2, 3]
        # Req 9 (owner_b, cp_rank=0): page 1 owned, pages 0,2 not -> positions [0,1,4,5]
        self.assertEqual(req_idxs.tolist(), [5, 5, 9, 9, 9, 9])
        self.assertEqual(positions.tolist(), [2, 3, 0, 1, 4, 5])

    def test_zero_extend_skips_request(self):
        owner = torch.tensor([0, 1], dtype=torch.int8)
        req_idxs, positions = compute_cp_non_owned_positions(
            cp_owner_per_pages=[owner],
            prefix_lens=[0],
            extend_lens=[0],
            page_size=2,
            cp_rank=0,
            req_pool_indices=[1],
        )
        self.assertEqual(positions.numel(), 0)

    def test_position_beyond_owner_raises(self):
        owner = torch.tensor([0, 1], dtype=torch.int8)
        with self.assertRaises(IndexError):
            compute_cp_non_owned_positions(
                cp_owner_per_pages=[owner],
                prefix_lens=[0],
                extend_lens=[8],  # needs 4 pages, owner only has 2
                page_size=2,
                cp_rank=0,
                req_pool_indices=[0],
            )

    def test_length_mismatch_raises(self):
        owner = torch.tensor([0, 1], dtype=torch.int8)
        with self.assertRaises(ValueError):
            compute_cp_non_owned_positions(
                cp_owner_per_pages=[owner, owner],
                prefix_lens=[0],
                extend_lens=[2],
                page_size=2,
                cp_rank=0,
                req_pool_indices=[0],
            )


class TestAllocTransientRows(CustomTestCase):
    def test_alloc_scatters_into_req_to_token(self):
        req_to_token = _make_req_to_token(num_reqs=3, max_context=16)
        allocator = _FakeAllocator(start=100, capacity=64)
        req_idxs = torch.tensor([1, 1, 2], dtype=torch.int64)
        positions = torch.tensor([5, 6, 4], dtype=torch.int64)

        rows = cp_alloc_transient_rows(req_to_token, allocator, req_idxs, positions)

        self.assertIsNotNone(rows)
        self.assertEqual(rows.tolist(), [100, 101, 102])
        # Slots updated:
        self.assertEqual(req_to_token[1, 5].item(), 100)
        self.assertEqual(req_to_token[1, 6].item(), 101)
        self.assertEqual(req_to_token[2, 4].item(), 102)
        # Untouched slots remain sentinel:
        self.assertEqual(req_to_token[0, 0].item(), 0)
        self.assertEqual(req_to_token[1, 0].item(), 0)

    def test_alloc_empty_returns_empty_tensor_no_allocator_call(self):
        req_to_token = _make_req_to_token(2, 8)
        allocator = _FakeAllocator()
        empty = torch.empty((0,), dtype=torch.int64)
        rows = cp_alloc_transient_rows(req_to_token, allocator, empty, empty)
        self.assertIsNotNone(rows)
        self.assertEqual(rows.numel(), 0)
        self.assertEqual(len(allocator.alloc_log), 0)

    def test_alloc_oom_returns_none_and_does_not_scatter(self):
        req_to_token = _make_req_to_token(1, 8)
        allocator = _FakeAllocator(start=1, capacity=2)  # only 2 free rows
        req_idxs = torch.tensor([0, 0, 0], dtype=torch.int64)
        positions = torch.tensor([3, 4, 5], dtype=torch.int64)

        rows = cp_alloc_transient_rows(req_to_token, allocator, req_idxs, positions)
        self.assertIsNone(rows)
        # No scatter happened:
        self.assertEqual(req_to_token.sum().item(), 0)

    def test_pair_length_mismatch_raises(self):
        req_to_token = _make_req_to_token(1, 4)
        allocator = _FakeAllocator()
        with self.assertRaises(ValueError):
            cp_alloc_transient_rows(
                req_to_token,
                allocator,
                torch.tensor([0, 0], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            )


class TestFreeTransientRows(CustomTestCase):
    def test_free_restores_allocator_and_rewrites_sentinels(self):
        req_to_token = _make_req_to_token(2, 8)
        allocator = _FakeAllocator(start=50, capacity=16)
        req_idxs = torch.tensor([0, 1], dtype=torch.int64)
        positions = torch.tensor([2, 3], dtype=torch.int64)

        rows = cp_alloc_transient_rows(req_to_token, allocator, req_idxs, positions)
        self.assertEqual(allocator.available, 14)

        cp_free_transient_rows(req_to_token, allocator, rows, req_idxs, positions)
        self.assertEqual(allocator.available, 16)
        self.assertEqual(req_to_token[0, 2].item(), 0)
        self.assertEqual(req_to_token[1, 3].item(), 0)

    def test_free_none_is_noop(self):
        req_to_token = _make_req_to_token(1, 4)
        allocator = _FakeAllocator()
        empty = torch.empty((0,), dtype=torch.int64)
        cp_free_transient_rows(req_to_token, allocator, None, empty, empty)
        self.assertEqual(len(allocator.free_log), 0)

    def test_free_empty_is_noop(self):
        req_to_token = _make_req_to_token(1, 4)
        allocator = _FakeAllocator()
        empty = torch.empty((0,), dtype=torch.int64)
        cp_free_transient_rows(req_to_token, allocator, empty, empty, empty)
        self.assertEqual(len(allocator.free_log), 0)

    def test_free_length_mismatch_raises(self):
        req_to_token = _make_req_to_token(1, 4)
        allocator = _FakeAllocator()
        with self.assertRaises(ValueError):
            cp_free_transient_rows(
                req_to_token,
                allocator,
                torch.tensor([10, 11], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
            )


class TestRoundTrip(CustomTestCase):
    def test_alloc_free_round_trip_preserves_pool_and_sentinels(self):
        req_to_token = _make_req_to_token(num_reqs=4, max_context=32)
        allocator = _FakeAllocator(start=200, capacity=128)
        baseline_free = allocator.available

        # cp_rank=2, page_size=4, 4 ranks
        owners = [
            torch.tensor([0, 1, 2, 3], dtype=torch.int8),  # 16 tokens
            torch.tensor([2, 0, 1, 3, 2], dtype=torch.int8),  # 20 tokens
            torch.tensor([2, 2, 2, 2], dtype=torch.int8),  # all owned
        ]
        prefix_lens = [0, 0, 0]
        extend_lens = [16, 20, 16]
        req_pool_indices = [0, 1, 2]

        req_idxs, positions = compute_cp_non_owned_positions(
            cp_owner_per_pages=owners,
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            page_size=4,
            cp_rank=2,
            req_pool_indices=req_pool_indices,
        )

        # Req 0 (owners [0,1,2,3]): page 2 owned -> non-owned positions 0-7, 12-15 (12 positions)
        # Req 1 (owners [2,0,1,3,2]): pages 0,4 owned -> non-owned positions 4-15 (12 positions)
        # Req 2 (all 2): all owned -> 0 non-owned
        self.assertEqual(positions.numel(), 24)

        rows = cp_alloc_transient_rows(req_to_token, allocator, req_idxs, positions)
        self.assertIsNotNone(rows)
        self.assertEqual(rows.numel(), 24)
        self.assertEqual(allocator.available, baseline_free - 24)

        # Every (req_idx, pos) in the output should now hold a non-sentinel value.
        for r, p, row in zip(req_idxs.tolist(), positions.tolist(), rows.tolist()):
            self.assertEqual(req_to_token[r, p].item(), row)

        cp_free_transient_rows(req_to_token, allocator, rows, req_idxs, positions)
        self.assertEqual(allocator.available, baseline_free)
        # All non-owned slots restored to sentinel:
        for r, p in zip(req_idxs.tolist(), positions.tolist()):
            self.assertEqual(req_to_token[r, p].item(), 0)


from types import SimpleNamespace

from sglang.srt.layers.utils.cp_transient import (
    cp_alloc_forward_transient,
    cp_alloc_req_transient,
    cp_free_forward_transient,
)


class _StubTreeCache:
    """Mimics ``RadixCache.evict``: frees ``num_tokens`` fresh ids into
    the underlying allocator. ``capacity`` caps how many tokens can be
    freed cumulatively (simulates the realistic case where all evictable
    nodes are exhausted)."""

    def __init__(self, allocator, capacity: int = 10**9):
        self._allocator = allocator
        self._next_row = 9000
        self._budget = capacity
        self.evict_calls: List[int] = []

    def evict(self, params):
        from sglang.srt.mem_cache.base_prefix_cache import EvictResult

        n = min(params.num_tokens, self._budget)
        self.evict_calls.append(params.num_tokens)
        if n > 0:
            rows = torch.arange(self._next_row, self._next_row + n, dtype=torch.int64)
            self._next_row += n
            self._budget -= n
            self._allocator.free(rows)
        return EvictResult(num_tokens_evicted=n)


def _make_forward_batch(
    cp_owner_per_pages,
    extend_prefix_lens_cpu,
    extend_seq_lens_cpu,
    req_pool_indices,
    num_req_rows: int = 8,
    max_context: int = 32,
):
    """Tiny stand-in for ``ForwardBatch`` exposing just the fields the
    wrappers read/write. ``req_to_token_pool.req_to_token`` is an int64
    tensor on CPU."""
    req_to_token = _make_req_to_token(num_req_rows, max_context)
    pool = SimpleNamespace(req_to_token=req_to_token)
    return SimpleNamespace(
        cp_owner_per_pages=cp_owner_per_pages,
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64),
        req_to_token_pool=pool,
        cp_transient_rows=None,
        cp_transient_req_indices=None,
        cp_transient_position_indices=None,
    )


class TestForwardBatchWrappers(CustomTestCase):
    def test_alloc_no_cp_admitted_short_circuits(self):
        # All requests have cp_owner_per_page=None -> nothing to do.
        fb = _make_forward_batch(
            cp_owner_per_pages=[None, None],
            extend_prefix_lens_cpu=[0, 0],
            extend_seq_lens_cpu=[4, 4],
            req_pool_indices=[0, 1],
        )
        allocator = _FakeAllocator()
        baseline = allocator.available
        dropped = cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
        self.assertEqual(dropped, [])
        self.assertIsNone(fb.cp_transient_rows)
        self.assertEqual(allocator.available, baseline)

    def test_alloc_decode_mode_no_extend(self):
        # extend_seq_lens_cpu None -> decode/idle, nothing to write.
        fb = _make_forward_batch(
            cp_owner_per_pages=[torch.tensor([0, 1], dtype=torch.int8)],
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=None,
            req_pool_indices=[0],
        )
        allocator = _FakeAllocator()
        dropped = cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
        self.assertEqual(dropped, [])
        self.assertIsNone(fb.cp_transient_rows)

    def test_alloc_all_owned_no_rows_no_drops(self):
        # cp_rank=0, every page owned by 0 -> no non-owned.
        fb = _make_forward_batch(
            cp_owner_per_pages=[torch.tensor([0, 0], dtype=torch.int8)],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[4],
            req_pool_indices=[2],
        )
        allocator = _FakeAllocator()
        baseline = allocator.available
        dropped = cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
        self.assertEqual(dropped, [])
        self.assertIsNone(fb.cp_transient_rows)
        self.assertEqual(allocator.available, baseline)

    def test_alloc_happy_path_populates_forward_batch(self):
        # cp_rank=1, page 0 owned by 0 (non-owned), page 1 owned by 1 (owned).
        fb = _make_forward_batch(
            cp_owner_per_pages=[torch.tensor([0, 1], dtype=torch.int8)],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[4],
            req_pool_indices=[3],
        )
        allocator = _FakeAllocator(start=100, capacity=16)

        dropped = cp_alloc_forward_transient(fb, allocator, cp_rank=1, page_size=2)
        self.assertEqual(dropped, [])
        # 2 non-owned positions -> 2 transient rows.
        self.assertIsNotNone(fb.cp_transient_rows)
        self.assertEqual(fb.cp_transient_rows.tolist(), [100, 101])
        self.assertEqual(fb.cp_transient_position_indices.tolist(), [0, 1])
        self.assertEqual(fb.cp_transient_req_indices.tolist(), [3, 3])
        # Scattered into req_to_token:
        rtt = fb.req_to_token_pool.req_to_token
        self.assertEqual(rtt[3, 0].item(), 100)
        self.assertEqual(rtt[3, 1].item(), 101)

    def test_alloc_mixed_admission(self):
        # First req is CP-admitted, second is not.
        fb = _make_forward_batch(
            cp_owner_per_pages=[torch.tensor([0, 1], dtype=torch.int8), None],
            extend_prefix_lens_cpu=[0, 0],
            extend_seq_lens_cpu=[4, 4],
            req_pool_indices=[5, 6],
        )
        allocator = _FakeAllocator(start=200, capacity=8)
        dropped = cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
        self.assertEqual(dropped, [])
        # Only req 5's non-owned positions (page 1 = positions 2,3) get rows.
        self.assertEqual(fb.cp_transient_req_indices.tolist(), [5, 5])
        self.assertEqual(fb.cp_transient_position_indices.tolist(), [2, 3])

    def test_alloc_drops_only_starved_req(self):
        # Two CP-admitted reqs. The first succeeds; the second hits OOM
        # and is dropped, but req 0's transient stays allocated.
        fb = _make_forward_batch(
            cp_owner_per_pages=[
                torch.tensor([0, 1], dtype=torch.int8),  # req 0: 2 non-owned positions
                torch.tensor(
                    [0, 1, 2, 3], dtype=torch.int8
                ),  # req 1: 6 non-owned positions
            ],
            extend_prefix_lens_cpu=[0, 0],
            extend_seq_lens_cpu=[4, 8],
            req_pool_indices=[5, 6],
        )
        # Pool has room for req 0 but not req 1 (need 6, only 3 left after).
        allocator = _FakeAllocator(start=300, capacity=5)

        dropped = cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
        self.assertEqual(dropped, [1])
        # Req 0's allocation survives:
        self.assertIsNotNone(fb.cp_transient_rows)
        self.assertEqual(fb.cp_transient_req_indices.tolist(), [5, 5])
        # Pool reflects only req 0's 2-row alloc.
        self.assertEqual(allocator.available, 5 - 2)

    def test_alloc_drops_oom_when_no_tree_cache(self):
        fb = _make_forward_batch(
            cp_owner_per_pages=[torch.tensor([0, 1, 2, 3], dtype=torch.int8)],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[8],
            req_pool_indices=[0],
        )
        # Only 2 rows free, but 6 non-owned positions needed.
        allocator = _FakeAllocator(start=1, capacity=2)
        dropped = cp_alloc_forward_transient(fb, allocator, cp_rank=0, page_size=2)
        self.assertEqual(dropped, [0])
        # No partial scatter: req_to_token stays at sentinel.
        self.assertIsNone(fb.cp_transient_rows)
        self.assertEqual(fb.req_to_token_pool.req_to_token.sum().item(), 0)

    def test_alloc_recovers_via_eviction(self):
        # Allocator is too small to fit on its own, but the tree_cache
        # can free enough rows to make the alloc succeed on retry.
        fb = _make_forward_batch(
            cp_owner_per_pages=[torch.tensor([0, 1, 2, 3], dtype=torch.int8)],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[8],
            req_pool_indices=[0],
        )
        allocator = _FakeAllocator(start=1, capacity=2)
        cache = _StubTreeCache(allocator, capacity=100)

        dropped = cp_alloc_forward_transient(
            fb, allocator, cp_rank=0, page_size=2, tree_cache=cache
        )
        self.assertEqual(dropped, [])
        self.assertIsNotNone(fb.cp_transient_rows)
        self.assertEqual(fb.cp_transient_rows.numel(), 6)
        # Tree cache was driven for the deficit (6 - 2 = 4).
        self.assertEqual(cache.evict_calls, [4])

    def test_alloc_drops_when_eviction_insufficient(self):
        # tree_cache exists but can only free a tiny amount; still drops.
        fb = _make_forward_batch(
            cp_owner_per_pages=[torch.tensor([0, 1, 2, 3], dtype=torch.int8)],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[8],
            req_pool_indices=[0],
        )
        allocator = _FakeAllocator(start=1, capacity=2)
        # Only 1 token of evictable capacity -- not enough to cover deficit 4.
        cache = _StubTreeCache(allocator, capacity=1)

        dropped = cp_alloc_forward_transient(
            fb, allocator, cp_rank=0, page_size=2, tree_cache=cache
        )
        self.assertEqual(dropped, [0])
        self.assertIsNone(fb.cp_transient_rows)

    def test_alloc_free_round_trip(self):
        fb = _make_forward_batch(
            cp_owner_per_pages=[torch.tensor([0, 1, 0, 1], dtype=torch.int8)],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[8],
            req_pool_indices=[2],
        )
        allocator = _FakeAllocator(start=50, capacity=32)
        baseline = allocator.available

        dropped = cp_alloc_forward_transient(fb, allocator, cp_rank=1, page_size=2)
        self.assertEqual(dropped, [])
        self.assertIsNotNone(fb.cp_transient_rows)
        non_owned_count = fb.cp_transient_rows.numel()
        self.assertEqual(allocator.available, baseline - non_owned_count)

        cp_free_forward_transient(fb, allocator)
        self.assertIsNone(fb.cp_transient_rows)
        self.assertIsNone(fb.cp_transient_req_indices)
        self.assertIsNone(fb.cp_transient_position_indices)
        self.assertEqual(allocator.available, baseline)
        # Sentinels restored.
        rtt = fb.req_to_token_pool.req_to_token
        self.assertEqual(rtt.sum().item(), 0)

    def test_free_is_noop_when_nothing_allocated(self):
        fb = _make_forward_batch(
            cp_owner_per_pages=[None],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[4],
            req_pool_indices=[0],
        )
        allocator = _FakeAllocator()
        cp_free_forward_transient(fb, allocator)  # no raise
        self.assertEqual(len(allocator.free_log), 0)


class TestAllocReqTransient(CustomTestCase):
    def test_returns_empty_rows_when_all_owned(self):
        req_to_token = _make_req_to_token(num_reqs=2, max_context=16)
        allocator = _FakeAllocator(start=10, capacity=8)
        owner = torch.tensor([3, 3], dtype=torch.int8)  # cp_rank=3 owns everything

        rows, req_idxs, positions = cp_alloc_req_transient(
            req_to_token=req_to_token,
            allocator=allocator,
            cp_owner_per_page=owner,
            prefix_len=0,
            extend_len=4,
            req_pool_idx=1,
            cp_rank=3,
            page_size=2,
        )
        self.assertIsNotNone(rows)
        self.assertEqual(rows.numel(), 0)
        self.assertEqual(req_idxs.numel(), 0)
        self.assertEqual(positions.numel(), 0)
        # No alloc was actually taken.
        self.assertEqual(len(allocator.alloc_log), 0)

    def test_happy_path_scatters_into_req_to_token(self):
        req_to_token = _make_req_to_token(num_reqs=3, max_context=16)
        allocator = _FakeAllocator(start=42, capacity=8)
        owner = torch.tensor([0, 1], dtype=torch.int8)

        rows, req_idxs, positions = cp_alloc_req_transient(
            req_to_token=req_to_token,
            allocator=allocator,
            cp_owner_per_page=owner,
            prefix_len=0,
            extend_len=4,
            req_pool_idx=2,
            cp_rank=1,
            page_size=2,
        )
        self.assertIsNotNone(rows)
        # cp_rank=1: page 0 (owner 0) non-owned -> positions 0,1.
        self.assertEqual(positions.tolist(), [0, 1])
        self.assertEqual(req_idxs.tolist(), [2, 2])
        self.assertEqual(rows.tolist(), [42, 43])
        self.assertEqual(req_to_token[2, 0].item(), 42)
        self.assertEqual(req_to_token[2, 1].item(), 43)

    def test_returns_none_on_oom_without_tree_cache(self):
        req_to_token = _make_req_to_token(num_reqs=1, max_context=16)
        allocator = _FakeAllocator(start=1, capacity=1)  # 1 row free
        owner = torch.tensor([0, 1, 2, 3], dtype=torch.int8)

        rows, _, _ = cp_alloc_req_transient(
            req_to_token=req_to_token,
            allocator=allocator,
            cp_owner_per_page=owner,
            prefix_len=0,
            extend_len=8,  # 6 non-owned, only 1 row free -> OOM
            req_pool_idx=0,
            cp_rank=0,
            page_size=2,
        )
        self.assertIsNone(rows)
        # No partial scatter happened.
        self.assertEqual(req_to_token.sum().item(), 0)

    def test_retries_after_eviction(self):
        req_to_token = _make_req_to_token(num_reqs=1, max_context=16)
        allocator = _FakeAllocator(start=1, capacity=2)
        cache = _StubTreeCache(allocator, capacity=100)
        owner = torch.tensor([0, 1, 2, 3], dtype=torch.int8)

        rows, req_idxs, positions = cp_alloc_req_transient(
            req_to_token=req_to_token,
            allocator=allocator,
            cp_owner_per_page=owner,
            prefix_len=0,
            extend_len=8,
            req_pool_idx=0,
            cp_rank=0,
            page_size=2,
            tree_cache=cache,
        )
        self.assertIsNotNone(rows)
        # need 6, available 2 -> deficit 4 evicted, then alloc succeeds.
        self.assertEqual(cache.evict_calls, [4])
        self.assertEqual(rows.numel(), 6)


if __name__ == "__main__":
    unittest.main()
