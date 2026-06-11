"""Unit test for dLLM FDFO in-place KV reuse in alloc_for_extend.

Directly asserts the lifecycle ClawSeven asked about: an unresolved FDFO block
(retained req_pool_idx + stashed dllm_incomplete_ids) reuses the KV slots already
mapped in req_to_token instead of being freed and reallocated, while a fresh
request in the same batch still draws new slots from the allocator.

Pure CPU test: exercises _alloc_extend_loc_with_kv_reuse (page_size==1 path) with
a real ReqToTokenPool and a minimal fake tree_cache/allocator.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.common import _alloc_extend_loc_with_kv_reuse
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


class _FakeAllocator:
    """Hands out a deterministic, easily-recognizable fresh slot range."""

    def __init__(self, base=1000):
        self.base = base
        self.calls = []

    def available_size(self):
        return 1 << 30

    def alloc(self, need_size):
        self.calls.append(need_size)
        return torch.arange(self.base, self.base + need_size, dtype=torch.int64)


class _FakeTreeCache:
    def __init__(self, allocator):
        self.page_size = 1
        self.token_to_kv_pool_allocator = allocator

    def is_chunk_cache(self):
        # Short-circuits evict_from_tree_cache so the fake allocator is used as-is.
        return True


def _make_batch(pool, allocator):
    return SimpleNamespace(
        device="cpu",
        req_to_token_pool=pool,
        tree_cache=_FakeTreeCache(allocator),
    )


class TestDllmFdfoKvReuse(unittest.TestCase):
    def setUp(self):
        self.block_size = 4
        self.pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )

    def test_reuse_req_keeps_block_slots_fresh_req_allocates(self):
        """Mixed batch: reuse req reuses its mapped slots; fresh req gets new ones."""
        b = self.block_size
        prefix_len = b

        # Reuse req lives in pool row 1 with a known block already mapped.
        retained = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
        self.pool.req_to_token[1, prefix_len : prefix_len + b] = retained

        allocator = _FakeAllocator(base=200)
        batch = _make_batch(self.pool, allocator)

        # Batch order: [reuse req (row 1), fresh req (row 2)].
        reuse_kv = [True, False]
        req_pool_indices_cpu = torch.tensor([1, 2], dtype=torch.int64)
        prefix_lens_cpu = torch.tensor([prefix_len, prefix_len], dtype=torch.int64)
        extend_lens_cpu = torch.tensor([b, b], dtype=torch.int64)

        out = _alloc_extend_loc_with_kv_reuse(
            batch, reuse_kv, req_pool_indices_cpu, prefix_lens_cpu, extend_lens_cpu
        )

        # Allocator was asked only for the fresh request's tokens (not the reuse one).
        self.assertEqual(allocator.calls, [b])
        # Reuse segment == the previously-mapped slots (no realloc).
        self.assertEqual(out[:b].tolist(), retained.tolist())
        # Fresh segment == allocator output.
        self.assertEqual(out[b : 2 * b].tolist(), [200, 201, 202, 203])
        self.assertEqual(out.numel(), 2 * b)

    def test_all_reuse_batch_allocates_nothing(self):
        """When every req reuses, the allocator must not be called at all."""
        b = self.block_size
        prefix_len = b
        retained = torch.tensor([300, 301, 302, 303], dtype=torch.int32)
        self.pool.req_to_token[3, prefix_len : prefix_len + b] = retained

        allocator = _FakeAllocator(base=900)
        batch = _make_batch(self.pool, allocator)

        out = _alloc_extend_loc_with_kv_reuse(
            batch,
            [True],
            torch.tensor([3], dtype=torch.int64),
            torch.tensor([prefix_len], dtype=torch.int64),
            torch.tensor([b], dtype=torch.int64),
        )

        self.assertEqual(allocator.calls, [])
        self.assertEqual(out.tolist(), retained.tolist())


if __name__ == "__main__":
    unittest.main()
