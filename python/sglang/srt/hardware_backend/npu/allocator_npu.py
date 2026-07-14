from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.utils import get_num_new_pages, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import KVCache


def alloc_extend_naive(
    prefix_lens,
    seq_lens,
    last_loc,
    free_pages,
    out_indices,
    page_size,
    device,
):
    extend_lens = seq_lens - prefix_lens
    end_pos = torch.cumsum(extend_lens, 0)
    start_pos = end_pos - extend_lens
    num_new_pages = (seq_lens + page_size - 1) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    num_full_new_pages = (seq_lens) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    need_page = num_new_pages - num_full_new_pages
    end_new_pages = torch.cumsum(num_new_pages, 0)
    start_new_pages = end_new_pages - num_new_pages
    pos_in_page = torch.arange(page_size, device=device, dtype=torch.int32)
    for i in range(len(prefix_lens)):
        num1 = (
            min(
                seq_lens[i],
                (prefix_lens[i] + page_size - 1) // page_size * page_size,
            )
            - prefix_lens[i]
        )
        if num1:
            out_indices[start_pos[i] : start_pos[i] + num1] = (
                last_loc[i] + 1 + pos_in_page[:num1].view(-1)
            )

        if prefix_lens[i] + num1 == seq_lens[i]:
            continue

        num2 = (
            seq_lens[i] // page_size - (prefix_lens[i] + page_size - 1) // page_size
        ) * page_size
        if num2:
            pages = (
                free_pages[start_new_pages[i] : end_new_pages[i] - need_page[i]]
                * page_size
            )
            out_indices[start_pos[i] + num1 : start_pos[i] + num1 + num2] = (
                pages.view(-1, 1) + pos_in_page.view(1, -1)
            ).view(-1)

        if prefix_lens[i] + num1 + num2 == seq_lens[i]:
            continue

        num3 = seq_lens[i] - seq_lens[i] // page_size * page_size
        if num3:
            out_indices[end_pos[i] - num3 : end_pos[i]] = (
                free_pages[end_new_pages[i] - 1] * page_size + pos_in_page[:num3]
            ).view(-1)


def alloc_extend_swa_tail_npu(
    allocator: "SWATokenToKVPoolAllocator",
    *,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    swa_tail_len: int,
    swa_tail_end: int,
) -> Optional[torch.Tensor]:
    assert allocator.page_size > 1
    assert len(seq_lens_cpu) == 1, "SWA tail allocation currently supports bs=1"
    assert len(prefix_lens_cpu) == 1
    assert 0 <= swa_tail_len <= swa_tail_end <= extend_num_tokens
    win_start: int = swa_tail_end - swa_tail_len
    assert win_start % allocator.page_size == 0

    num_full_pages = get_num_new_pages(
        seq_lens=seq_lens_cpu,
        page_size=allocator.page_size,
        prefix_lens=prefix_lens_cpu,
    )
    num_swa_pages = (swa_tail_len + allocator.page_size - 1) // allocator.page_size
    if not allocator.new_pages_available(num_full_pages, num_swa_pages):
        return None

    alloc_full_indices = allocator.full_attn_allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        num_new_pages=num_full_pages,
    )
    assert alloc_full_indices is not None

    if swa_tail_len == 0:
        return alloc_full_indices

    device = allocator.device
    swa_prefix_lens = torch.zeros((1,), dtype=torch.int64, device=device)
    swa_prefix_lens_cpu = torch.zeros((1,), dtype=torch.int64)
    swa_seq_lens = torch.tensor([swa_tail_len], dtype=torch.int64, device=device)
    swa_seq_lens_cpu = torch.tensor([swa_tail_len], dtype=torch.int64)
    swa_last_loc = torch.tensor([-1], dtype=torch.int64, device=device)

    alloc_swa_indices = allocator.swa_attn_allocator.alloc_extend(
        swa_prefix_lens,
        swa_prefix_lens_cpu,
        swa_seq_lens,
        swa_seq_lens_cpu,
        swa_last_loc,
        swa_tail_len,
        num_new_pages=num_swa_pages,
    )
    assert alloc_swa_indices is not None

    allocator.set_full_to_swa_mapping(
        alloc_full_indices[win_start:swa_tail_end], alloc_swa_indices
    )
    if win_start > 0:
        allocator.full_to_swa_index_mapping[
            alloc_full_indices[:win_start].to(torch.int64)
        ] = 0
    if swa_tail_end < extend_num_tokens:
        allocator.full_to_swa_index_mapping[
            alloc_full_indices[swa_tail_end:].to(torch.int64)
        ] = 0
    return alloc_full_indices


class NPUPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.roundup = page_size - 1

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        if num_new_pages is None:
            num_new_pages_tensor = (
                (seq_lens + self.roundup) // self.page_size
                - (prefix_lens + self.roundup) // self.page_size
            ).sum()
            num_new_pages_item = num_new_pages_tensor.item()
        else:
            num_new_pages_item = num_new_pages
        if self.need_sort and num_new_pages_item > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages_item > len(self.free_pages):
            return None

        if num_new_pages_item < 200:
            from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel

            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            max_num_extend_tokens = next_power_of_2(extend_num_tokens)
            bs = prefix_lens.shape[0]
            alloc_extend_kernel[(bs,)](
                prefix_lens,
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                next_power_of_2(bs),
                self.page_size,
                max_num_extend_tokens,
            )

        else:
            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int32,
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

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages_item:]
        return out_indices.int()

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )

        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages > len(self.free_pages):
            return None

        need_new_pages = (seq_lens % self.page_size == 1).int()
        end_new_pages = torch.cumsum(need_new_pages, 0)
        start_new_pages = end_new_pages - need_new_pages
        if num_new_pages == 0:
            out_indices = last_loc + 1
        else:
            out_indices = (last_loc + 1) * (1 - need_new_pages) + self.free_pages[
                start_new_pages
            ] * self.page_size * need_new_pages

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices.int()

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            device = free_index.device
            free_page_indices = torch.unique(free_index.cpu() // self.page_size)
            free_page_indices = free_page_indices.to(device)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)
