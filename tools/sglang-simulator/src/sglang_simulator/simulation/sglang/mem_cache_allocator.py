import types

import torch
from sglang_simulator.hook import BaseHook


def _alloc_extend_cpu(
    self,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    num_new_pages: int = None,
):
    """CPU implementation using SGLang's native paged-allocation helper."""
    from sglang.srt.mem_cache.allocator import alloc_extend_naive
    from sglang.srt.utils import get_num_new_pages

    if num_new_pages is None:
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
    if self.need_sort and num_new_pages > len(self.free_pages):
        self.merge_and_sort_free()
    if num_new_pages > len(self.free_pages):
        return None

    out_indices = torch.empty(
        (extend_num_tokens,),
        dtype=self.free_pages.dtype,
        device=self.device,
    )
    alloc_extend_naive(
        prefix_lens,
        seq_lens,
        last_loc,
        self.free_pages,
        out_indices,
        self.page_size,
        self.device,
    )
    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices


def _alloc_decode_cpu(
    self,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
):
    """CPU decode allocation through the allocator's public method contract."""
    from sglang.srt.utils import get_num_new_pages

    num_new_pages = get_num_new_pages(
        seq_lens=seq_lens_cpu,
        page_size=self.page_size,
        decode=True,
    )
    if self.need_sort and num_new_pages > len(self.free_pages):
        self.merge_and_sort_free()
    if num_new_pages > len(self.free_pages):
        return None

    out_indices = (last_loc + 1).to(dtype=self.free_pages.dtype)
    need_new_page = seq_lens % self.page_size == 1
    if num_new_pages:
        out_indices = out_indices.clone()
        out_indices[need_new_page] = self.free_pages[:num_new_pages] * self.page_size

    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices


def alloc_extend_cpu(*args, **kwargs):
    """Compatibility entry plus the native allocator-method implementation."""
    if args and isinstance(args[0], torch.Tensor):
        from sglang.srt.mem_cache.allocator import alloc_extend_naive

        prefix_lens, seq_lens, last_loc, free_pages, out_indices = args[:5]
        alloc_extend_naive(
            prefix_lens,
            seq_lens,
            last_loc,
            free_pages,
            out_indices,
            kwargs["page_size"],
            prefix_lens.device,
        )
        return None
    return _alloc_extend_cpu(*args, **kwargs)


def alloc_decode_cpu(*args, **kwargs):
    """Compatibility entry plus the native allocator-method implementation."""
    if args and isinstance(args[0], torch.Tensor):
        seq_lens, last_loc, free_pages, out_indices = args[:4]
        page_size = kwargs["page_size"]
        need_new_page = seq_lens % page_size == 1
        result = last_loc + 1
        result[need_new_page] = (
            free_pages[: int(need_new_page.sum().item())] * page_size
        )
        out_indices.copy_(result)
        return None
    return _alloc_decode_cpu(*args, **kwargs)


class C_PagedTokenToKVPoolAllocatorHook(BaseHook):
    HOOK_CLASS_NAME = "PagedTokenToKVPoolAllocator"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.allocator(?:\.paged)?$"
    REGEX = True

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if self.device == "cpu":
                self.alloc_extend = types.MethodType(_alloc_extend_cpu, self)
                self.alloc_decode = types.MethodType(_alloc_decode_cpu, self)

        target.__init__ = wrapped_init
