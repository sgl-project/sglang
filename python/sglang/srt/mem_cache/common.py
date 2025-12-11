from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.allocator import (
    HIERARCHICAL_NSA_DECODE_MAX_TOKENS,
    SWATokenToKVPoolAllocator,
    is_enable_hierarchical_nsa,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import support_triton
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
    if support_triton(get_global_server_args().attention_backend):
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            device=req_to_token_pool.device,
            dtype=torch.uint64,
        )
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i],
            )
            req_to_token_pool.write(
                (req_idx, slice(prefix_len, seq_len)),
                out_cache_loc[pt : pt + extend_len],
            )
            pt += extend_len


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    if (
        get_global_server_args().attention_backend != "ascend"
        and get_global_server_args().attention_backend != "torch_native"
    ):
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


@triton.jit
def get_last_loc_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    token_mask = prefix_lens > 0
    token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)

    tl.store(result + offset, tokens, mask=mask)


def get_last_loc_triton(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    BLOCK_SIZE = 256
    num_tokens = prefix_lens_tensor.shape[0]
    result = torch.empty_like(prefix_lens_tensor)
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    get_last_loc_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE,
    )
    return result


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if isinstance(tree_cache, (SWAChunkCache, ChunkCache)):
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    # Check if this is a hybrid allocator
    if hasattr(allocator, "full_available_size"):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(full_num_tokens, swa_num_tokens)
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(num_tokens)


def truncate_kv_cache_after_prefill(req: "Req", req_to_token_pool, tree_cache):
    """Truncate KV cache to HIERARCHICAL_NSA_DECODE_MAX_TOKENS after prefill completes."""
    if not is_enable_hierarchical_nsa(tree_cache.token_to_kv_pool_allocator):
        return

    if req.is_chunked > 0:
        return

    current_len = len(req.origin_input_ids)
    page_size = tree_cache.page_size
    kv_keep_len = ceil_align(HIERARCHICAL_NSA_DECODE_MAX_TOKENS, page_size)
    if current_len > kv_keep_len:
        old_prefix_len = len(req.prefix_indices)

        free_indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, kv_keep_len:current_len
        ]
        tree_cache.token_to_kv_pool_allocator.kv_allocator.free(free_indices)
        req.kv_committed_len = kv_keep_len
        req.kv_allocated_len = kv_keep_len
        req.prefix_indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_keep_len
        ].to(dtype=torch.int64, copy=True)
        tree_cache.protected_size_ -= old_prefix_len - kv_keep_len


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
    index_k_last_loc: Optional[torch.Tensor] = None,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    if is_enable_hierarchical_nsa(allocator):
        result = allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            index_k_last_loc,
        )
    else:
        result = allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    if result is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    if backup_state:
        # result is tuple (kv_indices, index_k_indices) for NSA or just kv_indices
        return (result, state) if isinstance(result, tuple) else ((result, None), state)
    return result


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    num_reqs: int,
    reqs: list[Req] | None,
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool."""
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        mamba_available_size = req_to_token_pool.mamba_pool.available_size()
        if mamba_available_size < num_reqs:
            if tree_cache is not None and isinstance(tree_cache, MambaRadixCache):
                mamba_num = max(0, num_reqs - mamba_available_size)
                tree_cache.evict_mamba(mamba_num)
        req_pool_indices = req_to_token_pool.alloc(num_reqs, reqs)
    else:
        req_pool_indices = req_to_token_pool.alloc(num_reqs)

    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, "
            f"{num_reqs=}, "
        )
    return req_pool_indices


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, list[int], Optional[torch.Tensor]]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated KV cache locations
        req_pool_indices_device: request pool indices as device tensor
        req_pool_indices: request pool indices as list
        out_index_cache_loc: allocated index_k locations (None if not NSA)
    """
    # free out-of-window swa tokens
    if isinstance(batch.tree_cache, SWAChunkCache):
        for req, pre_len in zip(batch.reqs, batch.prefix_lens):
            batch.tree_cache.evict_swa(
                req, pre_len, batch.model_config.attention_chunk_size
            )

    bs = len(batch.reqs)
    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, bs, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if batch.tree_cache.page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
        out_index_cache_loc = None
    else:
        # Get KV last_loc
        kv_last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        kv_last_loc = torch.cat(kv_last_loc)

        # Get index_k last_loc (NSA only)
        index_k_last_loc = _get_index_k_last_loc_for_extend(batch)
        alloc_result = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=kv_last_loc,
            extend_num_tokens=batch.extend_num_tokens,
            index_k_last_loc=index_k_last_loc,
        )

        # Unpack result
        if is_enable_hierarchical_nsa(batch.tree_cache.token_to_kv_pool_allocator):
            out_cache_loc, out_index_cache_loc = alloc_result
        else:
            out_cache_loc = alloc_result
            out_index_cache_loc = None

    # Write to req_to_token_pool (KV cache)
    write_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )

    if out_index_cache_loc is not None:
        _write_index_k_indices(
            batch,
            req_pool_indices,
            prefix_lens_cpu,
            extend_lens_cpu,
            out_index_cache_loc,
        )

    return out_cache_loc, req_pool_indices_device, req_pool_indices, out_index_cache_loc


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    out_cache_loc = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> tuple:
    """
    Unified entry point for decode allocation.

    Returns:
        tuple: (out_cache_loc, out_index_k_cache_loc or None)
    """
    if is_enable_hierarchical_nsa(batch.tree_cache.token_to_kv_pool_allocator):
        return _alloc_decode_nsa(batch, token_per_req)
    return _alloc_decode_standard(batch, token_per_req)


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    tree_cache.cache_finished_req(req, is_insert=is_insert)
    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    if spec_algo is None:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p >= end_p:
        return

    # Dispatch to appropriate release function
    if is_enable_hierarchical_nsa(tree_cache.token_to_kv_pool_allocator):
        _release_nsa(req, tree_cache, start_p, end_p)
    else:
        _release_standard(req, tree_cache, start_p, end_p)


def available_and_evictable_str(tree_cache) -> str:
    token_to_kv_pool_allocator = tree_cache.token_to_kv_pool_allocator
    if isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
        full_available_size = token_to_kv_pool_allocator.full_available_size()
        swa_available_size = token_to_kv_pool_allocator.swa_available_size()
        full_evictable_size = tree_cache.full_evictable_size()
        swa_evictable_size = tree_cache.swa_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"
            f"Full LRU list evictable size: {tree_cache.full_lru_list_evictable_size()}\n"
            f"SWA LRU list evictable size: {tree_cache.swa_lru_list_evictable_size()}\n"
        )
    else:
        available_size = token_to_kv_pool_allocator.available_size()
        evictable_size = tree_cache.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"


def _release_standard(req: Req, tree_cache: BasePrefixCache, start_p: int, end_p: int):
    """Standard KV cache release."""
    kv_indices = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
        start_p:end_p
    ]
    tree_cache.token_to_kv_pool_allocator.free(kv_indices)


def _release_nsa(req: Req, tree_cache: BasePrefixCache, start_p: int, end_p: int):
    """NSA hierarchical KV cache release."""
    kv_indices = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
        start_p:end_p
    ]
    index_k_indices = tree_cache.req_to_token_pool.req_to_nsa_index_k[req.req_pool_idx][
        start_p:end_p
    ]

    tree_cache.token_to_kv_pool_allocator.free((kv_indices, index_k_indices))


def _get_index_k_last_loc_for_extend(
    batch: ScheduleBatch,
) -> Optional[torch.Tensor]:
    """Get index_k last locations for NSA extend allocation."""
    if not is_enable_hierarchical_nsa(batch.tree_cache.token_to_kv_pool_allocator):
        return None

    bs = len(batch.reqs)
    index_k_last_loc = [
        (
            batch.reqs[i].index_k_prefix_indices[-1:]
            if batch.reqs[i].index_k_prefix_indices is not None
            and len(batch.reqs[i].index_k_prefix_indices) > 0
            else torch.tensor([-1], device=batch.device, dtype=torch.int32)
        )
        for i in range(bs)
    ]
    return torch.cat(index_k_last_loc)


def _write_index_k_indices(
    batch: ScheduleBatch,
    req_pool_indices: list[int],
    prefix_lens_cpu: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    out_index_cache_loc: torch.Tensor,
):
    pt = 0
    bs = len(batch.reqs)
    for i in range(bs):
        req_idx = req_pool_indices[i]
        prefix_len = prefix_lens_cpu[i].item()
        seq_len = batch.seq_lens_cpu[i].item()
        extend_len = extend_lens_cpu[i].item()
        req = batch.reqs[i]

        if prefix_len > 0 and req.index_k_prefix_indices is not None:
            batch.req_to_token_pool.write_index_token(
                (req_idx, slice(0, prefix_len)),
                req.index_k_prefix_indices[:prefix_len],
            )

        batch.req_to_token_pool.write_index_token(
            (req_idx, slice(prefix_len, seq_len)),
            out_index_cache_loc[pt : pt + extend_len].to(torch.int32),
        )
        pt += extend_len


def _alloc_decode_standard(batch: ScheduleBatch, token_per_req: int) -> tuple:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.
    """
    if isinstance(batch.tree_cache, SWAChunkCache):
        for req in batch.reqs:
            batch.tree_cache.evict_swa(
                req, req.seqlen - 1, batch.model_config.attention_chunk_size
            )

    bs = batch.seq_lens.shape[0]

    if batch.tree_cache.page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
        seq_lens_next = batch.seq_lens + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + batch.seq_lens
    else:
        locs = batch.seq_lens.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    return (out_cache_loc, None)


def _alloc_decode_nsa(batch: ScheduleBatch, token_per_req: int) -> tuple:
    """
    NSA hierarchical decode allocation.

    Returns:
        tuple: (out_cache_loc, out_index_cache_loc)
    """
    bs = batch.seq_lens.shape[0]
    seq_lens_next = batch.seq_lens + token_per_req
    allocator = batch.tree_cache.token_to_kv_pool_allocator
    kv_avail_before = allocator.kv_allocator.available_size()

    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + batch.seq_lens
    else:
        locs = batch.seq_lens.clone()

    # Find truncated and non-truncated requests
    kv_prompt_lens = torch.tensor(
        [len(req.origin_input_ids) for req in batch.reqs],
        dtype=torch.int32,
        device=batch.device,
    )
    truncated_mask = kv_prompt_lens >= HIERARCHICAL_NSA_DECODE_MAX_TOKENS
    out_cache_loc = torch.empty(bs, dtype=torch.int32, device=batch.device)

    num_truncated = truncated_mask.sum().item()
    num_non_truncated = bs - num_truncated

    # Handle truncated requests: use fixed position
    if truncated_mask.any():
        truncated_indices = truncated_mask.nonzero(as_tuple=True)[0]
        out_cache_loc[truncated_indices] = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices[truncated_indices],
            HIERARCHICAL_NSA_DECODE_MAX_TOKENS - 1,
        ]

    # Handle non-truncated requests: allocate normally
    if (~truncated_mask).any():
        non_truncated_indices = (~truncated_mask).nonzero(as_tuple=True)[0]
        if batch.tree_cache.page_size == 1:
            non_truncated_out = alloc_token_slots(
                batch.tree_cache, len(non_truncated_indices) * token_per_req
            )
        else:
            non_truncated_last_loc = batch.req_to_token_pool.req_to_token[
                batch.req_pool_indices[non_truncated_indices],
                batch.seq_lens[non_truncated_indices] - 1,
            ]
            non_truncated_out = alloc_paged_token_slots_decode(
                tree_cache=batch.tree_cache,
                seq_lens=seq_lens_next[non_truncated_indices],
                seq_lens_cpu=batch.seq_lens_cpu[non_truncated_indices.cpu()]
                + token_per_req,
                last_loc=non_truncated_last_loc,
                token_per_req=token_per_req,
            )

        out_cache_loc[non_truncated_indices] = non_truncated_out.to(torch.int32)
        batch.req_to_token_pool.write(
            (
                batch.req_pool_indices[non_truncated_indices],
                locs[non_truncated_indices],
            ),
            out_cache_loc[non_truncated_indices],
        )

    kv_avail_after = allocator.kv_allocator.available_size()
    if kv_avail_before != kv_avail_after:
        logger.info(
            f"[KV_ALLOC_DECODE] bs={bs}, truncated={num_truncated}, non_truncated={num_non_truncated}, "
            f"kv_avail={kv_avail_before}->{kv_avail_after}"
        )

    # Allocate index_k for all requests
    index_k_last_loc = batch.req_to_token_pool.req_to_nsa_index_k[
        batch.req_pool_indices, batch.seq_lens - 1
    ]
    out_index_cache_loc = (
        batch.tree_cache.token_to_kv_pool_allocator.alloc_index_k_only_decode(
            seq_lens_next, batch.seq_lens_cpu + token_per_req, index_k_last_loc
        )
    )

    if out_index_cache_loc is None:
        error_msg = (
            f"Decode out of memory for index_k. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens_next) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(batch.tree_cache)}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Write index_k indices
    batch.req_to_token_pool.write_index_token(
        (batch.req_pool_indices, locs), out_index_cache_loc.to(torch.int32)
    )

    return (out_cache_loc, out_index_cache_loc)
