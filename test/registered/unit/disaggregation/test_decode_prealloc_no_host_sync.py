"""Decode-side KV preallocation allocates one batch per `pop_preallocated`
and stages its allocator arguments through pinned memory. Per-request
`torch.tensor(..., device=cuda)` copies were a cudaStreamSynchronize each on
the scheduler stream; behind the WAR barrier that parks the host for the
whole in-flight forward.

    python -m pytest test/registered/unit/disaggregation/test_decode_prealloc_no_host_sync.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.base import pinned_int64_pair
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 4


def _sync_error(fn):
    """The RuntimeError torch raises if `fn` synchronizes, or None."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    except RuntimeError as exc:
        return exc
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    return None


class _RowPool:
    """req_to_token stand-in: hands out rows and records the writes."""

    def __init__(self):
        self.next_row = 0
        self.writes = []

    def alloc(self, reqs):
        for req in reqs:
            req.kv.req_pool_idx = self.next_row
            self.next_row += 1
        return [r.kv.req_pool_idx for r in reqs]

    def write(self, indices, values):
        self.writes.append((indices, values))


def _req(rid, num_tokens):
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=list(range(num_tokens)),
        output_ids=[],
        kv=ReqKvInfo(),
    )
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", SimpleNamespace(start=start, end=end)
    )
    return req


def _queue(allocator):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.req_to_token_pool = _RowPool()
    queue.token_to_kv_pool_allocator = allocator
    queue.tree_cache = MagicMock()
    queue.scheduler = SimpleNamespace(enable_hisparse=False)
    queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
    return queue


class TestDecodePreallocBatch(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_cpu_device_needs_no_pinned_staging(self):
        # pin_memory requires an accelerator; on a CPU allocator the pair is
        # the one host tensor.
        host, dev = pinned_int64_pair([3], "cpu")
        self.assertIs(host, dev)
        self.assertEqual(host.tolist(), [3])

    def test_one_allocation_serves_every_plan_in_order(self):
        allocator = TokenToKVPoolAllocator(
            size=64, dtype=torch.float16, device="cpu", kvcache=None, need_sort=False
        )
        queue = _queue(allocator)
        lens = [5, 3, 8]
        plans = [queue._plan_prealloc(_req(f"r{i}", n)) for i, n in enumerate(lens)]
        before = allocator.available_size()

        with unittest.mock.patch.object(
            allocator, "alloc", wraps=allocator.alloc
        ) as alloc:
            queue._alloc_planned(plans)
        alloc.assert_called_once_with(sum(lens))

        self.assertEqual(allocator.available_size(), before - sum(lens))
        all_slots = torch.cat([plan.kv_loc for plan in plans])
        self.assertEqual(len(torch.unique(all_slots)), sum(lens))
        for plan, n, (indices, values) in zip(
            plans, lens, queue.req_to_token_pool.writes
        ):
            self.assertEqual(plan.kv_loc.numel(), n)
            self.assertEqual(indices, (plan.req.kv.req_pool_idx, slice(0, n)))
            self.assertTrue(torch.equal(values, plan.kv_loc))
            self.assertEqual(plan.req.kv.kv_allocated_len, n)
            self.assertEqual(plan.req.kv.kv_committed_len, n)

    @unittest.skipUnless(torch.cuda.is_available(), "sync detection needs CUDA")
    def test_batched_prealloc_does_not_synchronize(self):
        allocator = PagedTokenToKVPoolAllocator(
            size=64 * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.float16,
            device="cuda",
            kvcache=None,
            need_sort=False,
        )
        queue = _queue(allocator)

        def prealloc(lens):
            plans = [queue._plan_prealloc(_req(f"r{n}", n)) for n in lens]
            queue._alloc_planned(plans)

        # Warm up outside the window: a first-time cudaMalloc / pinned block
        # can synchronize on its own, which the detector would blame on the call.
        prealloc([2 * PAGE_SIZE, PAGE_SIZE + 1])

        # Gate on the pre-fix form: a detector blind to pageable H2D copies would
        # pass the assert below no matter how the arguments are staged.
        if (
            _sync_error(lambda: torch.tensor([1], dtype=torch.int64, device="cuda"))
            is None
        ):
            self.skipTest("sync debug mode does not flag a pageable H2D copy here")

        self.assertIsNone(
            _sync_error(lambda: prealloc([3 * PAGE_SIZE - 1, 2, PAGE_SIZE]))
        )


if __name__ == "__main__":
    unittest.main()
