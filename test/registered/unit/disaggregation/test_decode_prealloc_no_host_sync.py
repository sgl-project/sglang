"""Decode-side KV preallocation stages its allocator arguments through pinned
memory. A pageable `torch.tensor(..., device=cuda)` copy is a
cudaStreamSynchronize on the scheduler stream; behind the WAR barrier that
parks the host for the whole in-flight forward.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode import (
    alloc_for_decode_prealloc,
    alloc_for_decode_prealloc_hisparse,
)
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

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


def _req():
    return MagicMock(kv=ReqKvInfo())


def _paged_prealloc():
    # With a device-resident prefix: the alloc_extend branch's lengths.
    allocator = PagedTokenToKVPoolAllocator(
        size=64 * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cuda",
        kvcache=None,
        need_sort=False,
    )
    prefix_indices = torch.arange(
        PAGE_SIZE, 2 * PAGE_SIZE, dtype=torch.int64, device="cuda"
    )
    fill_len = 3 * PAGE_SIZE - 1
    return lambda: alloc_for_decode_prealloc(
        allocator,
        req=_req(),
        fill_len=fill_len,
        delta_len=fill_len - PAGE_SIZE,
        prefix_len=PAGE_SIZE,
        total_prefix_len=PAGE_SIZE,
        prefix_indices=prefix_indices,
        uses_swa_tail=False,
        swa_tail_len=fill_len,
    )


def _swa_tail_prealloc():
    # No prefix: the -1 last_loc, and alloc_extend_swa_tail's own lengths.
    kvcache = MagicMock(spec=BaseSWAKVPool)
    kvcache.swa_req_ring_size = None
    allocator = SWATokenToKVPoolAllocator(
        size=64 * PAGE_SIZE,
        size_swa=32 * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cuda",
        kvcache=kvcache,
        need_sort=False,
    )
    fill_len = 5 * PAGE_SIZE
    return lambda: alloc_for_decode_prealloc(
        allocator,
        req=_req(),
        fill_len=fill_len,
        delta_len=fill_len,
        prefix_len=0,
        total_prefix_len=0,
        prefix_indices=None,
        uses_swa_tail=True,
        swa_tail_len=2 * PAGE_SIZE,
    )


def _hisparse_prealloc():
    # The HiSparse allocator needs a full coordinator; only the argument
    # staging is this function's own work.
    allocator = MagicMock(device="cuda", page_size=PAGE_SIZE)
    return lambda: alloc_for_decode_prealloc_hisparse(
        allocator,
        req=_req(),
        fill_len=3 * PAGE_SIZE,
        uses_swa_tail=False,
        swa_tail_len=3 * PAGE_SIZE,
    )


@unittest.skipUnless(torch.cuda.is_available(), "sync detection needs CUDA")
class TestDecodePreallocNoHostSync(CustomTestCase):
    def test_prealloc_paths_do_not_synchronize(self):
        # Gate on the pre-fix form: a detector blind to pageable H2D copies
        # would pass every assert below no matter how the arguments are staged.
        if (
            _sync_error(lambda: torch.tensor([1], dtype=torch.int64, device="cuda"))
            is None
        ):
            self.skipTest("sync debug mode does not flag a pageable H2D copy here")

        for name, prealloc in (
            ("paged", _paged_prealloc()),
            ("swa_tail", _swa_tail_prealloc()),
            ("hisparse", _hisparse_prealloc()),
        ):
            with self.subTest(name):
                # Warm up outside the window: a first-time cudaMalloc / pinned
                # block can synchronize on its own.
                prealloc()
                self.assertIsNone(_sync_error(prealloc))


if __name__ == "__main__":
    unittest.main()
