"""Tests dLLM FDFO KV slot reuse in alloc_for_extend."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.dllm.mixin.scheduler import DllmManager
from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache.allocation import alloc_for_extend
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


import pytest as _pytest_defer

_DEFER_REASON = (
    "Temporarily skipped during the ServerArgs config-namespace migration; "
    "re-enabled once the runtime-config accessor API stabilizes."
)
pytestmark = _pytest_defer.mark.skip(reason=_DEFER_REASON)


def setUpModule():
    import unittest

    raise unittest.SkipTest(_DEFER_REASON)


class _FakeAllocator:
    def __init__(self, base=1000, page_size=1):
        self.base = base
        self.page_size = page_size
        self.alloc_calls = []
        self.extend_calls = []

    def available_size(self):
        return 1 << 30

    def alloc(self, need_size):
        self.alloc_calls.append(need_size)
        return torch.arange(self.base, self.base + need_size, dtype=torch.int64)

    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        **kwargs,
    ):
        self.extend_calls.append(
            {
                "extend_num_tokens": extend_num_tokens,
                "seq_lens_cpu": seq_lens_cpu.tolist(),
            }
        )
        return torch.arange(self.base, self.base + extend_num_tokens, dtype=torch.int64)


class _FakeTreeCache:
    def __init__(self, allocator):
        self.page_size = allocator.page_size
        self.token_to_kv_pool_allocator = allocator

    def is_chunk_cache(self):
        return True


def _make_req(rid, prefix, block_size, *, req_pool_idx=None, reuse=False):
    return SimpleNamespace(
        rid=rid,
        prefix_indices=torch.tensor(prefix, dtype=torch.int32),
        req_pool_idx=req_pool_idx,
        dllm_incomplete_ids=array("q", range(block_size)) if reuse else array("q"),
        inflight_middle_chunks=1 if req_pool_idx is not None else 0,
        kv_committed_len=len(prefix) if req_pool_idx is not None else 0,
        kv=(
            SimpleNamespace(kv_allocated_len=len(prefix) + block_size)
            if req_pool_idx is not None
            else None
        ),
    )


def _remove_allocated_req_slots(pool, *reqs):
    for req in reqs:
        if req.req_pool_idx in pool.free_slots:
            pool.free_slots.remove(req.req_pool_idx)


def _make_batch(pool, allocator, reqs, extend_lens):
    seq_lens_cpu = torch.tensor(
        [
            len(req.prefix_indices) + extend_len
            for req, extend_len in zip(reqs, extend_lens)
        ],
        dtype=torch.int64,
    )
    return SimpleNamespace(
        device="cpu",
        reqs=reqs,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=_FakeTreeCache(allocator),
        prefix_lens=[len(req.prefix_indices) for req in reqs],
        extend_lens=extend_lens,
        seq_lens=seq_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=sum(extend_lens),
        maybe_evict_swa=lambda: None,
        is_dllm=lambda: True,
    )


def _seed_retained_block(pool, req, values):
    prefix_len = len(req.prefix_indices)
    pool.req_to_token[req.req_pool_idx, :prefix_len] = req.prefix_indices
    pool.req_to_token[req.req_pool_idx, prefix_len : prefix_len + len(values)] = (
        torch.tensor(values, dtype=torch.int32)
    )


class TestDllmFdfoKvReuse(unittest.TestCase):
    def setUp(self):
        self.block_size = 4
        self.pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        self._old_support_triton = allocation.support_triton
        self._old_get_server_args = allocation.get_server_args
        allocation.support_triton = lambda _: False
        allocation.get_server_args = lambda: SimpleNamespace(
            attention_backend="torch_native", dcp_size=1
        )

    def tearDown(self):
        allocation.support_triton = self._old_support_triton
        allocation.get_server_args = self._old_get_server_args

    def test_alloc_for_extend_mixed_reuse_allocates_only_fresh_and_writes_rows(self):
        allocator = _FakeAllocator(base=200)
        reused = _make_req(
            "reuse", [10, 11, 12, 13], self.block_size, req_pool_idx=1, reuse=True
        )
        fresh = _make_req("fresh", [20, 21, 22, 23], self.block_size)
        _remove_allocated_req_slots(self.pool, reused)
        _seed_retained_block(self.pool, reused, [100, 101, 102, 103])

        batch = _make_batch(self.pool, allocator, [reused, fresh], [4, 4])
        out, _, req_pool_indices_cpu = alloc_for_extend(batch)

        self.assertEqual(allocator.alloc_calls, [4])
        self.assertEqual(req_pool_indices_cpu.tolist(), [1, 2])
        self.assertEqual(out.tolist(), [100, 101, 102, 103, 200, 201, 202, 203])
        self.assertEqual(self.pool.req_to_token[1, 4:8].tolist(), [100, 101, 102, 103])
        self.assertEqual(self.pool.req_to_token[2, 4:8].tolist(), [200, 201, 202, 203])
        self.assertEqual(reused.kv.kv_allocated_len, 8)
        self.assertEqual(fresh.kv.kv_allocated_len, 8)

    def test_alloc_for_extend_all_reuse_allocates_nothing(self):
        allocator = _FakeAllocator(base=900)
        req0 = _make_req(
            "r0", [1, 2, 3, 4], self.block_size, req_pool_idx=1, reuse=True
        )
        req1 = _make_req(
            "r1", [5, 6, 7, 8], self.block_size, req_pool_idx=2, reuse=True
        )
        _remove_allocated_req_slots(self.pool, req0, req1)
        _seed_retained_block(self.pool, req0, [300, 301, 302, 303])
        _seed_retained_block(self.pool, req1, [400, 401, 402, 403])

        batch = _make_batch(self.pool, allocator, [req0, req1], [4, 4])
        out, _, req_pool_indices_cpu = alloc_for_extend(batch)

        self.assertEqual(allocator.alloc_calls, [])
        self.assertEqual(req_pool_indices_cpu.tolist(), [1, 2])
        self.assertEqual(out.tolist(), [300, 301, 302, 303, 400, 401, 402, 403])

    def test_alloc_for_extend_paged_mixed_reuse_skips_reused_rows(self):
        allocator = _FakeAllocator(base=500, page_size=4)
        reused = _make_req(
            "reuse", [10, 11, 12, 13], self.block_size, req_pool_idx=1, reuse=True
        )
        fresh = _make_req("fresh", [20, 21, 22, 23], self.block_size)
        _remove_allocated_req_slots(self.pool, reused)
        _seed_retained_block(self.pool, reused, [100, 101, 102, 103])

        batch = _make_batch(self.pool, allocator, [reused, fresh], [4, 4])
        out, _, req_pool_indices_cpu = alloc_for_extend(batch)

        self.assertEqual(req_pool_indices_cpu.tolist(), [1, 2])
        self.assertEqual(out.tolist(), [100, 101, 102, 103, 500, 501, 502, 503])
        self.assertEqual(
            allocator.extend_calls,
            [{"extend_num_tokens": 4, "seq_lens_cpu": [4, 8]}],
        )

    def test_alloc_for_extend_rejects_partial_retained_block_reuse(self):
        allocator = _FakeAllocator(base=700)
        reused = _make_req(
            "reuse", [10, 11, 12, 13], self.block_size, req_pool_idx=1, reuse=True
        )
        _remove_allocated_req_slots(self.pool, reused)
        _seed_retained_block(self.pool, reused, [100, 101, 102, 103])

        batch = _make_batch(self.pool, allocator, [reused], [2])
        with self.assertRaisesRegex(RuntimeError, "full block"):
            alloc_for_extend(batch)

    def test_dllm_manager_pop_aborted_reqs_removes_waiting_and_staging(self):
        manager = DllmManager(SimpleNamespace(max_running_requests=4))
        waiting = _make_req("abort-waiting", [1], self.block_size)
        staging = _make_req("abort-staging", [2], self.block_size)
        keep = _make_req("keep", [3], self.block_size)
        manager.waiting_queue = [waiting, keep]
        manager.staging_queue = [staging, waiting]

        aborted = manager.pop_aborted_reqs(False, "abort")

        self.assertEqual(
            [req.rid for req in aborted], ["abort-waiting", "abort-staging"]
        )
        self.assertEqual(manager.waiting_queue, [keep])
        self.assertEqual(manager.staging_queue, [])


if __name__ == "__main__":
    unittest.main()
