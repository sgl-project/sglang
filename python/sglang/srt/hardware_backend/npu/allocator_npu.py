import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.common import (
    available_and_evictable_str,
    evict_from_tree_cache,
)
from sglang.srt.utils import get_num_new_pages, is_npu, next_power_of_2

_is_npu = is_npu()

if _is_npu:
    import torch_npu

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
        DSV4NPUTokenToKVPoolAllocator,
    )
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import DSV4StateLens

logger = logging.getLogger(__name__)


def resolve_dsv4_npu_allocator(
    allocator: BaseTokenToKVPoolAllocator,
) -> Optional["DSV4NPUTokenToKVPoolAllocator"]:
    from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
        DSV4NPUTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.allocator.hisparse import (
        DeepSeekV4HiSparseTokenToKVPoolAllocator,
    )

    if isinstance(allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator):
        raise RuntimeError("DeepSeek V4 HiSparse is not supported on NPU")
    if not isinstance(allocator, DSV4NPUTokenToKVPoolAllocator):
        return None
    if allocator.page_size == 1:
        raise RuntimeError("DeepSeek V4 NPU allocation requires page_size > 1")
    return allocator


def alloc_for_extend_npu(
    tree_cache: "BasePrefixCache",
    *,
    prefix_tensors: list[torch.Tensor],
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_num_tokens: int,
    req_pool_indices: torch.Tensor,
    dsv4_state_lens: Optional["DSV4StateLens"],
    dsv4_allocator: Optional["DSV4NPUTokenToKVPoolAllocator"],
    batch: "ScheduleBatch",
) -> torch.Tensor:
    allocator = tree_cache.token_to_kv_pool_allocator
    if allocator.page_size == 1:
        return _alloc_token_slots_npu(
            tree_cache=tree_cache,
            num_tokens=extend_num_tokens,
            operation="Prefill",
        )

    last_locations = [
        (
            prefix_tensor[-1:]
            if prefix_tensor.numel() > 0
            else torch.tensor([-1], dtype=torch.int64, device=batch.device)
        )
        for prefix_tensor in prefix_tensors
    ]
    last_loc = (
        torch.cat(last_locations)
        if last_locations
        else torch.empty((0,), dtype=torch.int64, device=batch.device)
    )
    return _alloc_paged_token_slots_extend_npu(
        tree_cache=tree_cache,
        prefix_lens=prefix_lens,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        last_loc=last_loc,
        extend_num_tokens=extend_num_tokens,
        req_pool_indices=req_pool_indices,
        dsv4_state_lens=dsv4_state_lens,
        dsv4_allocator=dsv4_allocator,
        batch=batch,
    )


def alloc_for_decode_npu(
    batch: "ScheduleBatch",
    *,
    current_combined_lens: torch.Tensor,
    next_combined_lens: torch.Tensor,
    next_combined_lens_cpu: torch.Tensor,
    token_per_req: int,
    dsv4_state_lens: Optional["DSV4StateLens"],
    dsv4_allocator: Optional["DSV4NPUTokenToKVPoolAllocator"],
) -> torch.Tensor:
    allocator = batch.tree_cache.token_to_kv_pool_allocator
    if allocator.page_size == 1:
        return _alloc_token_slots_npu(
            tree_cache=batch.tree_cache,
            num_tokens=len(next_combined_lens) * token_per_req,
            operation="Decode",
        )

    last_loc = get_last_loc(
        batch.req_to_token_pool.req_to_token,
        batch.req_pool_indices,
        current_combined_lens,
    )
    return _alloc_paged_token_slots_decode_npu(
        tree_cache=batch.tree_cache,
        seq_lens=next_combined_lens,
        seq_lens_cpu=next_combined_lens_cpu,
        last_loc=last_loc,
        token_per_req=token_per_req,
        req_pool_indices=batch.req_pool_indices,
        dsv4_state_lens=dsv4_state_lens,
        dsv4_allocator=dsv4_allocator,
        batch=batch,
    )


def alloc_for_spec_decode_npu(
    tree_cache: "BasePrefixCache",
    *,
    req_to_token_pool: "ReqToTokenPool",
    req_pool_indices: torch.Tensor,
    decoder_current_lens_cpu: torch.Tensor,
    decoder_next_lens_cpu: torch.Tensor,
    combined_current_lens: torch.Tensor,
    combined_current_lens_cpu: torch.Tensor,
    combined_next_lens: torch.Tensor,
    combined_next_lens_cpu: torch.Tensor,
    num_needed_tokens: int,
    dsv4_allocator: Optional["DSV4NPUTokenToKVPoolAllocator"],
    batch: "ScheduleBatch",
) -> torch.Tensor:
    allocator = tree_cache.token_to_kv_pool_allocator
    if allocator.page_size == 1:
        return _alloc_token_slots_npu(
            tree_cache=tree_cache,
            num_tokens=num_needed_tokens,
            operation="Speculative decode",
        )

    last_loc = get_last_loc(
        req_to_token_pool.req_to_token,
        req_pool_indices,
        combined_current_lens,
    )
    dsv4_state_lens = (
        dsv4_allocator.compute_dsv4_state_lens_reserve(
            batch.reqs,
            decoder_current_lens_cpu,
            decoder_next_lens_cpu,
        )
        if dsv4_allocator is not None
        else None
    )
    return _alloc_paged_token_slots_extend_npu(
        tree_cache=tree_cache,
        prefix_lens=combined_current_lens,
        prefix_lens_cpu=combined_current_lens_cpu,
        seq_lens=combined_next_lens,
        seq_lens_cpu=combined_next_lens_cpu,
        last_loc=last_loc,
        extend_num_tokens=num_needed_tokens,
        req_pool_indices=req_pool_indices,
        dsv4_state_lens=dsv4_state_lens,
        dsv4_allocator=dsv4_allocator,
        batch=batch,
    )


def alloc_for_decode_prealloc_npu(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: "Req",
    fill_len: int,
    delta_len: int,
    prefix_len: int,
    total_prefix_len: int,
    prefix_indices: Optional[torch.Tensor],
    uses_swa_tail: bool,
    swa_tail_len: int,
) -> Optional[torch.Tensor]:
    dsv4_allocator = resolve_dsv4_npu_allocator(allocator)
    if dsv4_allocator is not None:
        raise RuntimeError("DeepSeek V4 NPU decode disaggregation is not supported")

    from sglang.srt.managers.schedule_batch import ReqKvInfo

    if req.kv is None:
        req.kv = ReqKvInfo(kv_allocated_len=fill_len, swa_evicted_seqlen=0)
    else:
        req.kv.kv_allocated_len = fill_len

    if allocator.page_size == 1:
        return allocator.alloc(delta_len)

    if uses_swa_tail:
        output_locations = allocator.alloc_extend_swa_tail(
            extend_num_tokens=fill_len,
            swa_tail_len=swa_tail_len,
            swa_tail_end=fill_len,
        )
        req.kv.swa_evicted_seqlen = fill_len - swa_tail_len
        return output_locations

    device = allocator.device
    if prefix_len > 0:
        assert prefix_indices is not None
        last_loc = prefix_indices[-1:].to(dtype=torch.int64, device=device)
    else:
        last_loc = torch.tensor([-1], dtype=torch.int64, device=device)
    prefix_lens_cpu = torch.tensor([total_prefix_len], dtype=torch.int64)
    seq_lens_cpu = torch.tensor([fill_len], dtype=torch.int64)
    return allocator.alloc_extend(
        prefix_lens=prefix_lens_cpu.to(device=device),
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=seq_lens_cpu.to(device=device),
        seq_lens_cpu=seq_lens_cpu,
        last_loc=last_loc,
        extend_num_tokens=fill_len - total_prefix_len,
    )


def _alloc_token_slots_npu(
    *,
    tree_cache: "BasePrefixCache",
    num_tokens: int,
    operation: str,
) -> torch.Tensor:
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)
    output_locations = allocator.alloc(num_tokens)
    if output_locations is not None:
        return output_locations

    error_message = (
        f"{operation} out of memory. Try to lower your batch size.\n"
        f"Try to allocate {num_tokens} tokens.\n"
        f"{available_and_evictable_str(tree_cache)}"
    )
    logger.error(error_message)
    if tree_cache is not None:
        tree_cache.pretty_print()
    raise RuntimeError(error_message)


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


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    """Slot id of each req's last already-allocated token, or -1 when
    ``prefix_lens[i] == 0`` (fresh req).

    Looks up ``req_to_token[req, prefix_lens - 1]`` to anchor the paged
    allocator's ``alloc_extend`` on the real previous tail slot, preserving the
    intra-page slot continuity the kernel's ``cmp_block_table`` relies on (the
    allocator debug-asserts ``(last_loc + 1) % page_size == prefix_lens %
    page_size``). Result dtype matches ``prefix_lens``.
    """
    req_pool_indices = req_pool_indices.to(torch.int64)
    safe_idx = (prefix_lens.to(torch.int64) - 1).clamp(min=0)
    looked_up = req_to_token[req_pool_indices, safe_idx].to(prefix_lens.dtype)
    return torch.where(
        prefix_lens > 0,
        looked_up,
        torch.full_like(prefix_lens, -1),
    )


def _alloc_paged_token_slots_extend_npu(
    tree_cache: "BasePrefixCache",
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional["DSV4StateLens"] = None,
    dsv4_allocator: Optional["DSV4NPUTokenToKVPoolAllocator"] = None,
    batch: Optional["ScheduleBatch"] = None,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = allocator.backup_state() if backup_state else None
    if dsv4_allocator is not None:
        from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc

        assert allocator is dsv4_allocator
        assert req_pool_indices is not None
        assert batch is not None
        batch.out_cache_loc_dsv4 = None
        output = dsv4_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            req_pool_indices=req_pool_indices,
            req_to_token_pool=batch.req_to_token_pool,
            dsv4_state_lens=dsv4_state_lens,
        )
        bundle = output
        if bundle is not None and not isinstance(bundle, DSV4OutCacheLoc):
            raise TypeError("DSV4 NPU extend allocation must return DSV4OutCacheLoc")
        output_locations = None if bundle is None else bundle.out_full_loc
        batch.out_cache_loc_dsv4 = bundle
    else:
        output_locations = allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    if output_locations is None:
        error_message = (
            "Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_message)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_message)
    return (output_locations, state) if backup_state else output_locations


def _alloc_paged_token_slots_decode_npu(
    tree_cache: "BasePrefixCache",
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional["DSV4StateLens"] = None,
    dsv4_allocator: Optional["DSV4NPUTokenToKVPoolAllocator"] = None,
    batch: Optional["ScheduleBatch"] = None,
) -> torch.Tensor:
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, len(seq_lens) * allocator.page_size)

    if dsv4_allocator is not None:
        from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc

        assert allocator is dsv4_allocator
        assert req_pool_indices is not None
        assert batch is not None
        batch.out_cache_loc_dsv4 = None
        output = dsv4_allocator.alloc_decode(
            seq_lens,
            seq_lens_cpu,
            last_loc,
            req_pool_indices=req_pool_indices,
            req_to_token_pool=batch.req_to_token_pool,
            dsv4_state_lens=dsv4_state_lens,
        )
        bundle = output
        if bundle is not None and not isinstance(bundle, DSV4OutCacheLoc):
            raise TypeError("DSV4 NPU decode allocation must return DSV4OutCacheLoc")
        output_locations = None if bundle is None else bundle.out_full_loc
        batch.out_cache_loc_dsv4 = bundle
    else:
        output_locations = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    if output_locations is None:
        error_message = (
            "Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_message)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_message)
    return output_locations


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


class NPUSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        assert self.page_size > 1

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if not self.new_pages_available(num_new_pages, num_new_pages):
            return None

        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            swa_last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.set_full_to_swa_mapping(alloc_full_indices, alloc_swa_indices)

        return alloc_full_indices

    def alloc_extend_swa_tail(
        self,
        *,
        extend_num_tokens: int,
        swa_tail_len: int,
        swa_tail_end: int,
    ) -> Optional[torch.Tensor]:
        allocator = self
        device = allocator.device
        prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
        seq_lens_cpu = torch.tensor([extend_num_tokens], dtype=torch.int64)
        prefix_lens = torch.tensor([0], dtype=torch.int64, device=device)
        seq_lens = torch.tensor([extend_num_tokens], dtype=torch.int64, device=device)
        last_loc = torch.tensor([-1], dtype=torch.int64, device=device)
        assert allocator.page_size > 1
        assert len(seq_lens_cpu) == 1
        assert len(prefix_lens_cpu) == 1
        assert 0 <= swa_tail_len <= swa_tail_end <= extend_num_tokens
        win_start = swa_tail_end - swa_tail_len
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

        swa_prefix_lens = torch.zeros((1,), dtype=torch.int64, device=allocator.device)
        swa_prefix_lens_cpu = torch.zeros((1,), dtype=torch.int64)
        swa_seq_lens = torch.tensor(
            [swa_tail_len], dtype=torch.int64, device=allocator.device
        )
        swa_seq_lens_cpu = torch.tensor([swa_tail_len], dtype=torch.int64)
        swa_last_loc = torch.tensor([-1], dtype=torch.int64, device=allocator.device)
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

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        assert self.page_size > 1
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, swa_last_loc
        )

        if alloc_full_indices is None or alloc_swa_indices is None:
            return None

        if _is_npu:
            indices_2d = alloc_full_indices.to(torch.int64).unsqueeze(-1)
            torch_npu.npu_scatter_nd_update_(
                self.full_to_swa_index_mapping,
                indices_2d,
                alloc_swa_indices.to(torch.int64),
            )
        else:
            self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices

        return alloc_full_indices

    def _get_paged_allocator_class(
        self,
    ) -> type[PagedTokenToKVPoolAllocator]:
        return NPUPagedTokenToKVPoolAllocator
