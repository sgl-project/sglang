"""Tests dLLM FDFO KV slot reuse in alloc_for_extend."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.dllm.manager import DllmManager
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocation import alloc_for_extend
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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

    def supports_prefix_sharing(self):
        return False

    def maybe_hand_to_session(self, req):
        pass

    def prefix_device_indices(self, req):
        return req.tree_prefix


def _make_req(rid, prefix, block_size, *, req_pool_idx=None, reuse=False):
    return SimpleNamespace(
        rid=rid,
        tree_prefix=torch.tensor(prefix, dtype=torch.int32),
        prefix_len=len(prefix),
        dllm=SimpleNamespace(
            incomplete_ids=array("q", range(block_size)) if reuse else array("q")
        ),
        inflight_middle_chunks=1 if req_pool_idx is not None else 0,
        kv=ReqKvInfo(
            req_pool_idx=req_pool_idx,
            kv_committed_len=len(prefix) if req_pool_idx is not None else 0,
            kv_allocated_len=(
                len(prefix) + block_size if req_pool_idx is not None else 0
            ),
        ),
    )


def _remove_allocated_req_slots(pool, *reqs):
    for req in reqs:
        if req.kv.req_pool_idx in pool.free_slots:
            pool.free_slots.remove(req.kv.req_pool_idx)


def _make_batch(pool, allocator, reqs, extend_lens):
    seq_lens_cpu = torch.tensor(
        [req.prefix_len + extend_len for req, extend_len in zip(reqs, extend_lens)],
        dtype=torch.int64,
    )
    return SimpleNamespace(
        device="cpu",
        reqs=reqs,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=_FakeTreeCache(allocator),
        hisparse_coordinator=None,
        prefix_lens=[req.prefix_len for req in reqs],
        extend_lens=extend_lens,
        seq_lens=seq_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=sum(extend_lens),
        maybe_evict_swa=lambda: None,
        kv_reuse_lens=[
            len(req.dllm.incomplete_ids) if req.kv.holds_kv else 0 for req in reqs
        ],
    )


def _seed_retained_block(pool, req, values):
    prefix_len = req.prefix_len
    pool.req_to_token[req.kv.req_pool_idx, :prefix_len] = req.tree_prefix
    pool.req_to_token[req.kv.req_pool_idx, prefix_len : prefix_len + len(values)] = (
        torch.tensor(values, dtype=torch.int32)
    )


class TestDllmFdfoKvReuse(unittest.TestCase):
    def setUp(self):
        self.block_size = 4
        self.pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        override = get_context().override_server_args(
            attention_backend="torch_native", dcp_size=1
        )
        override.install()
        self.addCleanup(override.restore)

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
        # Allocation order is not semantically meaningful (ReqToTokenPool.alloc
        # picks whichever free slot is cheapest to pop), so only pin the
        # reused row's index and that the fresh row got a different, real slot.
        self.assertEqual(req_pool_indices_cpu[0].item(), 1)
        fresh_idx = req_pool_indices_cpu[1].item()
        self.assertNotEqual(fresh_idx, 1)
        self.assertEqual(out.tolist(), [100, 101, 102, 103, 200, 201, 202, 203])
        self.assertEqual(self.pool.req_to_token[1, 4:8].tolist(), [100, 101, 102, 103])
        self.assertEqual(
            self.pool.req_to_token[fresh_idx, 4:8].tolist(), [200, 201, 202, 203]
        )
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

        # See test_alloc_for_extend_mixed_reuse_allocates_only_fresh_and_writes_rows:
        # allocation order is not semantically meaningful.
        self.assertEqual(req_pool_indices_cpu[0].item(), 1)
        self.assertNotEqual(req_pool_indices_cpu[1].item(), 1)
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


class TestDllmFdfoResolvedBlockKeepsRow(unittest.TestCase):
    def test_resolved_block_keeps_row_until_next_block(self):
        """A resolved FDFO block used to hand its row back to the pool while the
        request kept running. An abort before the next block then skipped
        release_kv_cache (it only runs for row holders), leaking the request's
        tree lock and any KV the tree does not own."""
        pool = ReqToTokenPool(
            size=4, max_context_len=16, device="cpu", enable_memory_saver=False
        )
        req = SimpleNamespace(
            dllm=SimpleNamespace(incomplete_ids=array("q"), is_prefill=lambda: False),
            kv=ReqKvInfo(kv_allocated_len=8, kv_committed_len=8),
        )
        pool.alloc([req])
        row = req.kv.req_pool_idx
        manager = SimpleNamespace(
            dllm_config=SimpleNamespace(
                first_done_first_out_mode=True,
                requires_separate_context_encoding=False,
            ),
            tree_cache=object(),
        )

        with patch("sglang.srt.dllm.manager.checkpoint_kv_cache") as checkpoint:
            DllmManager.finish_dllm_forward(manager, req)

        checkpoint.assert_called_once_with(req, manager.tree_cache)
        self.assertTrue(req.kv.holds_kv)
        self.assertEqual(req.kv.req_pool_idx, row)
        self.assertNotIn(row, pool.free_slots)


if __name__ == "__main__":
    unittest.main()
