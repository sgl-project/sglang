from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.hardware_backend.npu.hybrid_swa_c4_c128_memory_pool import (
    SWAC4C128TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.chunk_cache import SWAC4C128ChunkCache
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var, support_triton
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

from sglang.srt.model_executor.forward_batch_info import (
    ExtendNumTokens,
    KvLen,
    LastLoc,
    OutCacheLoc,
)

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
MAMBA_STATE_PER_REQ_NO_CACHE = 1

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


@triton.jit
def write_req_to_token_pool_only_alloc_size_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    alloc_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    alloc_len = tl.load(alloc_lens + pid)
    # if alloc_len > 0, this batch has kv to save
    # otherwise, alloc_offset = seq_len, so that num_loop=0
    alloc_offset = seq_len - alloc_len
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
        cumsum_start += tl.load(alloc_lens + i)

    num_loop = tl.cdiv(alloc_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < alloc_len
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + alloc_offset,
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


def write_multi_cache_indices(
    out_cache_loc: OutCacheLoc,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: KvLen,
    seq_lens_cpu: KvLen,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
    prefix_pointers = torch.tensor(
        [t.data_ptr() for t in prefix_tensors],
        device=req_to_token_pool.device,
        dtype=torch.uint64,
    )
    # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
    write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
        req_to_token_pool.req_to_token,
        req_pool_indices_tensor,
        prefix_pointers,  # prefix value
        prefix_lens_tensor,  # TODO: diff prefix_pointers value
        seq_lens_tensor.full_kv_len,
        seq_lens_tensor.full_kv_len,
        out_cache_loc.out_full_loc,
        req_to_token_pool.req_to_token.shape[1],
    )

    write_req_to_token_pool_only_alloc_size_triton[(req_pool_indices_tensor.shape[0],)](
        req_to_token_pool.req_to_token_swa,
        req_pool_indices_tensor,
        prefix_pointers,
        prefix_lens_tensor,
        seq_lens_tensor.full_kv_len,
        seq_lens_tensor.swa_kv_len,
        out_cache_loc.out_swa_loc,
        req_to_token_pool.req_to_token_swa.shape[1],
    )
    write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
        req_to_token_pool.req_to_token_c4,
        req_pool_indices_tensor,
        prefix_pointers,
        prefix_lens_tensor,
        seq_lens_tensor.c4_kv_len,
        seq_lens_tensor.c4_kv_len,
        out_cache_loc.out_c4_loc,
        req_to_token_pool.req_to_token_c4.shape[1],
    )

    write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
        req_to_token_pool.req_to_token_c128,
        req_pool_indices_tensor,
        prefix_pointers,
        prefix_lens_tensor,
        seq_lens_tensor.c128_kv_len,
        seq_lens_tensor.c128_kv_len,
        out_cache_loc.out_c128_loc,
        req_to_token_pool.req_to_token_c128.shape[1],
    )

    write_req_to_token_pool_only_alloc_size_triton[(req_pool_indices_tensor.shape[0],)](
        req_to_token_pool.req_to_token_c4_state,
        req_pool_indices_tensor,
        prefix_pointers,
        prefix_lens_tensor,
        seq_lens_tensor.full_kv_len,
        seq_lens_tensor.c4_state_kv_len,
        out_cache_loc.out_c4_state_loc,
        req_to_token_pool.req_to_token_c4_state.shape[1],
    )

    write_req_to_token_pool_only_alloc_size_triton[(req_pool_indices_tensor.shape[0],)](
        req_to_token_pool.req_to_token_c128_state,
        req_pool_indices_tensor,
        prefix_pointers,
        prefix_lens_tensor,
        seq_lens_tensor.full_kv_len,
        seq_lens_tensor.c128_state_kv_len,
        out_cache_loc.out_c128_state_loc,
        req_to_token_pool.req_to_token_c128_state.shape[1],
    )


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    if get_global_server_args().attention_backend != "torch_native":
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
    MAX_PREFIX_LEN: tl.constexpr,  # 新增：最大前缀长度
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    valid_prefix_len = tl.minimum(prefix_lens, MAX_PREFIX_LEN)
    valid_prefix_len = tl.maximum(valid_prefix_len, 0)

    # 只有 prefix_len > 0 时才计算有效 token 位置
    token_mask = valid_prefix_len > 0
    # last token index = valid_prefix_len - 1, but at least 0
    col_idx = tl.where(token_mask, valid_prefix_len - 1, 0)

    token_index = req_pool_indices * req_to_token_stride + col_idx

    tokens = tl.load(req_to_token + token_index, mask=mask, other=-1)
    tl.store(result + offset, tokens, mask=mask)


def get_last_loc_triton(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    assert req_to_token.dim() == 2
    assert req_pool_indices_tensor.shape == prefix_lens_tensor.shape
    assert req_pool_indices_tensor.dtype == torch.int64
    assert prefix_lens_tensor.dtype == torch.int64

    BLOCK_SIZE = 4
    num_tokens = prefix_lens_tensor.shape[0]
    result = torch.empty_like(prefix_lens_tensor, dtype=torch.int32)
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    get_last_loc_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result,
        num_tokens,
        req_to_token.stride(0),
        MAX_PREFIX_LEN=req_to_token.size(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return result


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    extend_lens: list[int],
    backup_state: bool = False,
    is_prefill: bool = True,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    if isinstance(allocator, SWAC4C128TokenToKVPoolAllocator):
        out_cache_loc = allocator.alloc(num_tokens, extend_lens, is_prefill)
    else:
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

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: KvLen,
    prefix_lens_cpu: KvLen,
    seq_lens: KvLen,
    seq_lens_cpu: KvLen,
    last_loc: LastLoc,
    extend_num_tokens: ExtendNumTokens,
    backup_state: bool = False,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = (
        extend_num_tokens.full_extend_num_tokens
        + len(seq_lens_cpu.full_kv_len) * allocator.page_size
    )
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    )

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool."""
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        mamba_available_size = req_to_token_pool.mamba_pool.available_size()
        factor = (
            MAMBA_STATE_PER_REQ_PREFIX_CACHE
            if tree_cache.supports_mamba()
            else MAMBA_STATE_PER_REQ_NO_CACHE
        )
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    req_pool_indices = req_to_token_pool.alloc(reqs)

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
        req_pool_indices_device: request pool indices at a device tensor
        req_pool_indices: request pool indices as list
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # TODO: prefixcache
    prefix_lens_kv_cpu = KvLen.from_data(prefix_lens_cpu.shape[0], 0)
    prefix_lens_kv = prefix_lens_kv_cpu.to(batch.device)
    # Allocate req slots
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if batch.tree_cache.page_size == 1:
        out_cache_loc = alloc_token_slots(
            batch.tree_cache,
            batch.extend_num_tokens,
            batch.seq_lens_cpu,
            is_prefill=True,
        )
    else:
        # Paged allocation - build last_loc
        def _get_last_loc(prefix_tensors):
            return [
                (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
                for t in prefix_tensors
            ]

        if get_bool_env_var("IS_DEEPSEEK_V4", "False"):
            # prefix_tensors[i] = [dummy_kv, swa_kv, c4_kv, c128_kv, c4_state_kv, c128_state_kv]
            # last_loc = [torch.cat(i) for i in map(get_last_loc, map(list, zip(*prefix_tensors)))]
            last_loc = _get_last_loc(prefix_tensors)
            last_loc = torch.cat(last_loc)
            last_loc = [last_loc for _ in range(6)]
            last_loc = LastLoc(*last_loc)

            out_cache_loc = alloc_paged_token_slots_extend(
                tree_cache=batch.tree_cache,
                prefix_lens=prefix_lens_kv,
                prefix_lens_cpu=prefix_lens_kv_cpu,
                seq_lens=batch.kv_seq_lens,
                seq_lens_cpu=batch.kv_seq_lens_cpu,
                last_loc=last_loc,
                extend_num_tokens=batch.extend_num_kv,
            )
        else:
            last_loc = _get_last_loc(prefix_tensors)
            last_loc = torch.cat(last_loc)

            out_cache_loc = alloc_paged_token_slots_extend(
                tree_cache=batch.tree_cache,
                prefix_lens=prefix_lens_device,
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=batch.seq_lens_cpu,
                last_loc=torch.cat(last_loc),
                extend_num_tokens=batch.extend_num_tokens,
            )

    # Write to req_to_token_pool
    write_multi_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.kv_seq_lens,
        batch.kv_seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )

    return out_cache_loc, req_pool_indices_device, req_pool_indices


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    prefix_lens: KvLen,
    prefix_lens_cpu: KvLen,
    seq_lens: KvLen,
    seq_lens_cpu: KvLen,
    last_loc: LastLoc,
    extend_num_tokens: ExtendNumTokens,
    token_per_req: int = 1,
) -> OutCacheLoc:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens.full_kv_len) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    out_cache_loc = allocator.alloc_decode(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    )

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


def alloc_for_decode(
    batch: ScheduleBatch, c4_extend_num_kv, c128_extend_num_kv, token_per_req: int
) -> OutCacheLoc:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """

    batch.maybe_evict_swa()

    bs = batch.seq_lens.shape[0]

    if batch.tree_cache.page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(
            batch.tree_cache,
            bs * token_per_req,
            batch.seq_lens_cpu + token_per_req,
            is_prefill=False,
        )
    else:
        c4_extend_num_kv_cpu = torch.tensor(
            c4_extend_num_kv, dtype=torch.int32, device="cpu"
        )
        c4_extend_num_kv_device = c4_extend_num_kv_cpu.pin_memory().to(
            device=batch.device, non_blocking=True
        )
        c128_extend_num_kv_cpu = torch.tensor(
            c128_extend_num_kv, dtype=torch.int32, device="cpu"
        )
        c128_extend_num_kv_device = c128_extend_num_kv_cpu.pin_memory().to(
            device=batch.device, non_blocking=True
        )
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
        swa_last_loc = batch.req_to_token_pool.req_to_token_swa[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
        # TODO: prefill seqlen < 4
        c4_last_loc = get_last_loc(
            batch.req_to_token_pool.req_to_token_c4,
            batch.req_pool_indices,
            batch.kv_seq_lens.c4_kv_len,
        )

        c128_last_loc = get_last_loc(
            batch.req_to_token_pool.req_to_token_c128,
            batch.req_pool_indices,
            batch.kv_seq_lens.c128_kv_len,
        )

        c4_state_last_loc = batch.req_to_token_pool.req_to_token_c4_state[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
        c128_state_last_loc = batch.req_to_token_pool.req_to_token_c128_state[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
        cur_last_loc = LastLoc(
            last_loc=last_loc,
            last_swa_loc=swa_last_loc,
            last_c4_loc=c4_last_loc,
            last_c128_loc=c128_last_loc,
            last_c4_state_loc=c4_state_last_loc,
            last_c128_state_loc=c128_state_last_loc,
        )
        seq_lens_next = KvLen(
            full_kv_len=batch.kv_seq_lens.full_kv_len + token_per_req,
            swa_kv_len=batch.kv_seq_lens.swa_kv_len + token_per_req,
            c4_kv_len=batch.kv_seq_lens.c4_kv_len + c4_extend_num_kv_device,
            c128_kv_len=batch.kv_seq_lens.c128_kv_len + c128_extend_num_kv_device,
            c4_state_kv_len=batch.kv_seq_lens.c4_state_kv_len + token_per_req,
            c128_state_kv_len=batch.kv_seq_lens.c128_state_kv_len + token_per_req,
        )
        seq_lens_next_cpu = KvLen(
            full_kv_len=batch.kv_seq_lens_cpu.full_kv_len + token_per_req,
            swa_kv_len=batch.kv_seq_lens_cpu.swa_kv_len + token_per_req,
            c4_kv_len=batch.kv_seq_lens_cpu.c4_kv_len + c4_extend_num_kv_cpu,
            c128_kv_len=batch.kv_seq_lens_cpu.c128_kv_len + c128_extend_num_kv_cpu,
            c4_state_kv_len=batch.kv_seq_lens_cpu.c4_state_kv_len + token_per_req,
            c128_state_kv_len=batch.kv_seq_lens_cpu.c128_state_kv_len + token_per_req,
        )
        extend_num_tokens = ExtendNumTokens(
            full_extend_num_tokens=bs,
            swa_extend_num_tokens=bs,
            c4_extend_num_tokens=sum(c4_extend_num_kv),
            c128_extend_num_tokens=sum(c128_extend_num_kv),
            c4_state_extend_num_tokens=bs,
            c128_state_extend_num_tokens=bs,
        )
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            prefix_lens=batch.kv_seq_lens,
            prefix_lens_cpu=batch.kv_seq_lens_cpu,
            seq_lens=seq_lens_next,
            seq_lens_cpu=seq_lens_next_cpu,
            last_loc=cur_last_loc,
            extend_num_tokens=extend_num_tokens,
            token_per_req=token_per_req,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + batch.seq_lens
    else:
        locs = batch.seq_lens.clone()
        c4_locs = batch.kv_seq_lens.c4_kv_len.clone()
        c128_locs = batch.kv_seq_lens.c128_kv_len.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.out_full_loc.to(torch.int32)
    )
    batch.req_to_token_pool.write_swa(
        (batch.req_pool_indices, locs), out_cache_loc.out_swa_loc.to(torch.int32)
    )

    if sum(c4_extend_num_kv) > 0:
        assert sum(c4_extend_num_kv) == out_cache_loc.out_c4_loc.numel()
        c4_should_compress = torch.tensor(c4_extend_num_kv) > 0
        batch.req_to_token_pool.write_c4(
            (batch.req_pool_indices[c4_should_compress], c4_locs[c4_should_compress]),
            out_cache_loc.out_c4_loc.to(torch.int32),
        )
    if sum(c128_extend_num_kv) > 0:
        assert sum(c128_extend_num_kv) == out_cache_loc.out_c128_loc.numel()
        c128_should_compress = torch.tensor(c128_extend_num_kv) > 0
        batch.req_to_token_pool.write_c128(
            (
                batch.req_pool_indices[c128_should_compress],
                c128_locs[c128_should_compress],
            ),
            out_cache_loc.out_c128_loc.to(torch.int32),
        )
    batch.req_to_token_pool.write_c4_state(
        (batch.req_pool_indices, locs),
        out_cache_loc.out_c4_state_loc.to(torch.int32),
    )
    batch.req_to_token_pool.write_c128_state(
        (batch.req_pool_indices, locs),
        out_cache_loc.out_c128_state_loc.to(torch.int32),
    )
    return out_cache_loc


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_pool.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    tree_cache.cache_finished_req(req, is_insert=is_insert)

    # FIXME: SessionAwareCache.cache_finished_req sets req_pool_idx = None to
    # transfer KV ownership to the SessionSlot, so we skip the remaining
    # cleanup (overalloc free + pool slot free). This means over-allocated
    # tokens from speculative decoding are NOT freed between turns.
    if req.req_pool_idx is None:
        return

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    if not isinstance(tree_cache, SWAC4C128ChunkCache):
        start_p, end_p = req.pop_overallocated_kv_cache()
        if spec_algo is None:
            assert (
                start_p == end_p
            ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"
    else:
        (
            start_p,
            end_p,
            c4_start_p,
            c4_end_p,
            c128_start_p,
            c128_end_p,
        ) = req.pop_overallocated_kv_cache_swa_c4_c128()
        if spec_algo is None:
            assert (
                start_p == end_p or c4_start_p == c4_end_p or c128_start_p == c128_end_p
            ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}\
                {req.c4_kv_committed_len=}, {req.c4_kv_allocated_len=}, \
                {req.c128_kv_committed_len=}, {req.c128_kv_allocated_len=}, "

    if not isinstance(tree_cache, SWAC4C128ChunkCache):
        if page_size > 1:
            start_p = ceil_align(start_p, page_size)

        if start_p < end_p:  # next page
            indices_to_free = tree_cache.req_to_token_pool.req_to_token[
                req.req_pool_idx
            ][start_p:end_p]
            tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
        # If the prefix cache doesn't manage mamba states, we must free them here.
        if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
            not tree_cache.supports_mamba()
        ):
            assert (
                req.mamba_pool_idx is not None
            ), "mamba state is freed while the tree cache does not manage mamba states"
            tree_cache.req_to_token_pool.free_mamba_cache(req)
    else:
        if page_size > 1:
            start_p_align_up = ceil_align(start_p, page_size)
            c4_start_p_align_up = ceil_align(c4_start_p, page_size)
            c128_start_p_align_up = ceil_align(c128_start_p, page_size)

        if start_p_align_up < end_p:
            indices_to_free = tree_cache.req_to_token_pool.req_to_token[
                req.req_pool_idx
            ][start_p_align_up:end_p]
            tree_cache.token_to_kv_pool_allocator.dummy_attn_allocator.free(
                indices_to_free
            )
            indices_to_free = tree_cache.req_to_token_pool.req_to_token_swa[
                req.req_pool_idx
            ][start_p_align_up:end_p]
            c4_state_indices_to_free = (
                tree_cache.req_to_token_pool.req_to_token_c4_state[req.req_pool_idx][
                    start_p_align_up:end_p
                ]
            )
            tree_cache.token_to_kv_pool_allocator.free_compress_state(
                c4_state_indices_to_free, "c4"
            )

            c128_state_indices_to_free = (
                tree_cache.req_to_token_pool.req_to_token_c128_state[req.req_pool_idx][
                    start_p_align_up:end_p
                ]
            )
            tree_cache.token_to_kv_pool_allocator.free_compress_state(
                c128_state_indices_to_free, "c128"
            )
            tree_cache.token_to_kv_pool_allocator.free_swa(indices_to_free)
        if c4_start_p_align_up < c4_end_p:
            c4_indices_to_free = tree_cache.req_to_token_pool.req_to_token_c4[
                req.req_pool_idx
            ][c4_start_p_align_up:c4_end_p]
            tree_cache.token_to_kv_pool_allocator.free_compress(
                c4_indices_to_free, "c4"
            )
        if c128_start_p_align_up < c128_end_p:
            c128_indices_to_free = tree_cache.req_to_token_pool.req_to_token_c128[
                req.req_pool_idx
            ][c128_start_p_align_up:c128_end_p]
            tree_cache.token_to_kv_pool_allocator.free_compress(
                c128_indices_to_free, "c128"
            )

    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
