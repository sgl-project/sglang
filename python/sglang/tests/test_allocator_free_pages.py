import os

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator


class _DummyKVCache:
    # Minimal stub; allocator will not use it in these tests
    pass


def _make_allocator(debug: bool = False):
    if debug:
        os.environ["SGLANG_DEBUG_MEMORY_POOL"] = "1"
    else:
        os.environ.pop("SGLANG_DEBUG_MEMORY_POOL", None)
    # size must be multiple of page_size
    size = 1024
    page_size = 64
    return PagedTokenToKVPoolAllocator(
        size=size,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=_DummyKVCache(),
    )


def test_free_pages_append_only_no_debug():
    alloc = _make_allocator(debug=False)
    # Start with empty release list
    alloc.release_pages = torch.empty((0,), dtype=torch.int64, device=alloc.device)
    pages = torch.tensor([2, 5, 7], dtype=torch.int64, device=alloc.device)
    alloc.free_pages(pages)
    assert torch.equal(alloc.release_pages, pages)


def test_free_shim_collapses_tokens_to_single_page():
    alloc = _make_allocator(debug=False)
    alloc.release_pages = torch.empty((0,), dtype=torch.int64, device=alloc.device)
    page_size = alloc.page_size
    page_id = torch.tensor(3, dtype=torch.int64)
    # Tokens from the same page (1-based page ids, token indices are 0-based)
    tokens = page_id * page_size + torch.arange(0, 10, dtype=torch.int64)
    alloc.free(tokens.to(device=alloc.device))
    assert alloc.release_pages.numel() == 1
    assert alloc.release_pages.item() == page_id.item()


def test_merge_and_sort_free_no_debug_is_append_only():
    alloc = _make_allocator(debug=False)
    # Override lists to controlled values
    alloc.free_pages = torch.tensor([5, 3], dtype=torch.int64, device=alloc.device)
    alloc.release_pages = torch.tensor([4, 1], dtype=torch.int64, device=alloc.device)
    alloc.merge_and_sort_free()
    assert torch.equal(
        alloc.free_pages,
        torch.tensor([5, 3, 4, 1], dtype=torch.int64, device=alloc.device),
    )
    assert alloc.release_pages.numel() == 0


def test_merge_and_sort_free_debug_sorts():
    alloc = _make_allocator(debug=True)
    # Do not call free()/free_pages() in debug to avoid free-guard; set buffers directly
    alloc.free_pages = torch.tensor([5, 3], dtype=torch.int64, device=alloc.device)
    alloc.release_pages = torch.tensor([4, 1], dtype=torch.int64, device=alloc.device)
    alloc.merge_and_sort_free()
    assert torch.equal(
        alloc.free_pages,
        torch.tensor([1, 3, 4, 5], dtype=torch.int64, device=alloc.device),
    )
    assert alloc.release_pages.numel() == 0
