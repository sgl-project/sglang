"""CPU test: c128 page refcount retain/release must be exactly symmetric.

The C128 prefix-cache pages are shared (refcounted). `retain_c128_pages` and
`release_c128_pages` must normalize their inputs identically, otherwise the pair
is unbalanced: a page appearing twice in one `retain` call (or the page-0
sentinel, which `release` drops via `> 0`) inflates the refcount, the matching
`release` can never bring it to 0, and the page leaks out of the free list.

This is a pure-CPU test of the refcount arithmetic (no NPU): the methods only
touch `self.c128_page_refcount` and `self.c128_attn_allocator`, both stubbed.
"""

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
    DSV4NPUTokenToKVPoolAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _FakePagedAllocator:
    page_size = 16

    def __init__(self):
        self.freed_slots = []

    def free(self, flat_slots):
        self.freed_slots.append(flat_slots.clone())


class _Harness:
    retain_c128_pages = DSV4NPUTokenToKVPoolAllocator.retain_c128_pages
    release_c128_pages = DSV4NPUTokenToKVPoolAllocator.release_c128_pages

    def __init__(self, n_pages=64):
        self.c128_page_refcount = torch.zeros(n_pages, dtype=torch.int32)
        self.c128_attn_allocator = _FakePagedAllocator()


def test_retain_release_are_exactly_symmetric_with_duplicates():
    h = _Harness()
    # A page referenced by several groups appears multiple times in the tensor.
    h.retain_c128_pages(torch.tensor([5, 5, 5, 7]))
    assert int(h.c128_page_refcount[5]) == 1, "retain must de-duplicate"
    assert int(h.c128_page_refcount[7]) == 1

    h.release_c128_pages(torch.tensor([5, 5, 5, 7]))
    assert int(h.c128_page_refcount[5]) == 0, "release must undo retain exactly"
    assert int(h.c128_page_refcount[7]) == 0
    # Pages returned to the free list (page_size * page id).
    freed = torch.cat(h.c128_attn_allocator.freed_slots) if h.c128_attn_allocator.freed_slots else torch.tensor([])
    assert set(freed.tolist()) == {5 * 16, 7 * 16}


def test_retain_drops_page0_sentinel_like_release():
    h = _Harness()
    h.retain_c128_pages(torch.tensor([0, 3]))
    assert int(h.c128_page_refcount[0]) == 0, "page-0 sentinel must be dropped"
    assert int(h.c128_page_refcount[3]) == 1


def test_repeated_retain_release_cycles_do_not_leak():
    h = _Harness()
    for _ in range(3):
        h.retain_c128_pages(torch.tensor([11, 11, 13]))
        h.release_c128_pages(torch.tensor([11, 11, 13]))
    assert int(h.c128_page_refcount[11]) == 0
    assert int(h.c128_page_refcount[13]) == 0


if __name__ == "__main__":
    test_retain_release_are_exactly_symmetric_with_duplicates()
    test_retain_drops_page0_sentinel_like_release()
    test_repeated_retain_release_cycles_do_not_leak()
    print("OK")
