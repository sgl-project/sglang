from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_hip, support_triton
from sglang.srt.utils.common import ceil_align

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


def _safe_cache_stat(tree_cache: BasePrefixCache | None, stat_name: str):
    if tree_cache is None:
        return None
    stat_fn = getattr(tree_cache, stat_name, None)
    if stat_fn is None:
        return None
    try:
        return stat_fn()
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def _short_repr(value, limit: int = 256) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_call(obj, method_name: str):
    fn = getattr(obj, method_name, None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def _safe_sub(*values):
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
        return int(values[0] - sum(values[1:]))
    return None


def _tensor_sum_int(value):
    try:
        return int(value.sum().item())
    except Exception:
        return None


def _kv_pressure_snapshot(tree_cache: BasePrefixCache | None) -> dict[str, object]:
    if tree_cache is None:
        return {}

    allocator = tree_cache.token_to_kv_pool_allocator
    total_size = _safe_call(tree_cache, "total_size")
    if isinstance(total_size, tuple) and len(total_size) == 2:
        full_tree_total, swa_tree_total = total_size
    else:
        full_tree_total, swa_tree_total = total_size, None

    fields: dict[str, object] = {
        "allocator": type(allocator).__name__,
        "tree_total": total_size,
        "full_tree_evictable": _safe_cache_stat(tree_cache, "full_evictable_size"),
        "swa_tree_evictable": _safe_cache_stat(tree_cache, "swa_evictable_size"),
        "full_tree_protected": _safe_cache_stat(tree_cache, "full_protected_size"),
        "swa_tree_protected": _safe_cache_stat(tree_cache, "swa_protected_size"),
    }

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        full_capacity = getattr(allocator, "size_full", None)
        full_capacity = full_capacity() if callable(full_capacity) else full_capacity
        swa_capacity = getattr(allocator, "size_swa", None)
        swa_capacity = swa_capacity() if callable(swa_capacity) else swa_capacity
        full_available = _safe_call(allocator, "full_available_size")
        swa_available = _safe_call(allocator, "swa_available_size")

        fields.update(
            {
                "full_capacity": full_capacity,
                "full_available": full_available,
                "full_used": _safe_sub(full_capacity, full_available),
                "full_tree_total": full_tree_total,
                "full_non_tree_live": _safe_sub(
                    full_capacity, full_available, full_tree_total
                ),
                "swa_capacity": swa_capacity,
                "swa_available": swa_available,
                "swa_used": _safe_sub(swa_capacity, swa_available),
                "swa_tree_total": swa_tree_total,
                "swa_non_tree_live": _safe_sub(
                    swa_capacity, swa_available, swa_tree_total
                ),
            }
        )
    else:
        capacity = getattr(allocator, "size", None)
        capacity = capacity() if callable(capacity) else capacity
        available = _safe_call(allocator, "available_size")
        fields.update(
            {
                "capacity": capacity,
                "available": available,
                "used": _safe_sub(capacity, available),
                "non_tree_live": _safe_sub(capacity, available, full_tree_total),
            }
        )

    return fields


def _log_kv_alloc_debug(
    event: str,
    tree_cache: BasePrefixCache | None,
    **fields,
) -> None:
    if not envs.SGLANG_DEBUG_KV_ALLOC.get():
        return
    merged = {"event": event, **fields, **_kv_pressure_snapshot(tree_cache)}
    logger.info(
        "KV_ALLOC_DEBUG %s",
        " ".join(f"{key}={_short_repr(value)}" for key, value in merged.items()),
    )


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # The page is guaranteed to be full except the last page.
    if page_size == 1:
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    return (length // page_size) * page_size


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)


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
    attn_backend = get_global_server_args().attention_backend
    uses_triton_dispatch = attn_backend not in ("ascend", "torch_native")

    if _is_hip and uses_triton_dispatch:
        # HIP-only: the legacy get_last_loc_triton kernel emits a
        # mixed-width int32->int64 store that Triton mis-compiles on HIP,
        # producing out-of-range last_loc values under EAGLE +
        # page_size>1 (e.g. with aiter unified attention or the triton
        # attention backend). The bug is in the Triton HIP codegen, not
        # in any particular attention backend, so route every HIP path
        # that would otherwise use get_last_loc_triton through the
        # int32-safe variant. Non-HIP hardware keeps the original
        # dispatcher below.
        return get_last_loc_triton_safe(
            req_to_token, req_pool_indices_tensor, prefix_lens_tensor
        )

    if uses_triton_dispatch:
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
def _get_last_loc_safe_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result_i32,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
    PREFIX_DTYPE_IS_I64: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    if PREFIX_DTYPE_IS_I64:
        prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
        req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)
        token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    else:
        prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
        req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)
        token_index = req_pool_indices.to(tl.int64) * req_to_token_stride + (
            prefix_lens.to(tl.int64) - 1
        )

    token_mask = mask & (prefix_lens > 0)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)
    # Result stays int32 (req_to_token dtype); caller promotes after return.
    tl.store(result_i32 + offset, tokens, mask=mask)


def get_last_loc_triton_safe(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    """Fused `last_loc` Triton kernel whose in-kernel result buffer is int32
    (the dtype of req_to_token). The consumer-dtype promotion happens in
    torch after the kernel returns, so Triton never issues a mixed-width
    store — avoiding the HIP int32->int64 store bug hit by the legacy kernel.
    """
    num_tokens = prefix_lens_tensor.shape[0]
    BLOCK_SIZE = 256
    result_i32 = torch.empty(
        num_tokens, dtype=torch.int32, device=prefix_lens_tensor.device
    )
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)
    _get_last_loc_safe_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result_i32,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        PREFIX_DTYPE_IS_I64=(prefix_lens_tensor.dtype == torch.int64),
    )
    return result_i32.to(prefix_lens_tensor.dtype)


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
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    _log_kv_alloc_debug(
        "extend_before_evict",
        tree_cache,
        extend_num_tokens=extend_num_tokens,
        padded_num_tokens=num_tokens,
        batch_size=len(seq_lens_cpu),
        page_size=allocator.page_size,
        prefix_tokens=_tensor_sum_int(prefix_lens_cpu),
        seq_tokens=_tensor_sum_int(seq_lens_cpu),
    )
    evict_from_tree_cache(tree_cache, num_tokens)
    _log_kv_alloc_debug(
        "extend_after_evict",
        tree_cache,
        extend_num_tokens=extend_num_tokens,
        padded_num_tokens=num_tokens,
        batch_size=len(seq_lens_cpu),
        page_size=allocator.page_size,
    )

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
        _log_kv_alloc_debug(
            "extend_oom",
            tree_cache,
            extend_num_tokens=extend_num_tokens,
            padded_num_tokens=num_tokens,
            batch_size=len(seq_lens_cpu),
            page_size=allocator.page_size,
        )
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    _log_kv_alloc_debug(
        "extend_after_alloc",
        tree_cache,
        extend_num_tokens=extend_num_tokens,
        padded_num_tokens=num_tokens,
        batch_size=len(seq_lens_cpu),
        page_size=allocator.page_size,
    )
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
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
        req_pool_indices_device: request pool indices at a device tensor
        req_pool_indices: request pool indices as list
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()
    _log_kv_alloc_debug(
        "alloc_for_extend_batch",
        batch.tree_cache,
        batch_size=len(batch.reqs),
        extend_num_tokens=batch.extend_num_tokens,
        prefix_tokens=sum(batch.prefix_lens),
        seq_tokens=_tensor_sum_int(batch.seq_lens_cpu),
    )

    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if batch.tree_cache.page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
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

    return out_cache_loc, req_pool_indices_device, req_pool_indices


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


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """

    batch.maybe_evict_swa()

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

    return out_cache_loc


def release_kv_cache(
    req: Req,
    tree_cache: BasePrefixCache,
    is_insert: bool = True,
    debug_context: Optional[str] = None,
):
    log_cache_debug = (
        debug_context is not None and envs.SGLANG_DEBUG_DISAGG_PREFILL_CACHE.get()
    )
    if log_cache_debug:
        req_pool_idx_before = req.req_pool_idx
        kv_committed_len_before = req.kv_committed_len
        kv_allocated_len_before = req.kv_allocated_len
        skip_radix_cache_insert = getattr(req, "skip_radix_cache_insert", False)
        effective_is_insert = is_insert and not skip_radix_cache_insert
        cache_disable = getattr(tree_cache, "disable", None)
        cache_disable_finished_insert = getattr(
            tree_cache, "disable_finished_insert", None
        )
        cache_total_before = _safe_cache_stat(tree_cache, "total_size")
        cache_evictable_before = _safe_cache_stat(tree_cache, "evictable_size")
        cache_protected_before = _safe_cache_stat(tree_cache, "protected_size")

    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        if log_cache_debug:
            logger.info(
                "release_kv_cache_debug context=%s rid=%s req_pool_idx=None "
                "kv_committed_len=%s kv_allocated_len=%s is_insert=%s "
                "effective_is_insert=%s skip_radix_cache_insert=%s "
                "cache_disable=%s cache_disable_finished_insert=%s "
                "bootstrap_host=%s bootstrap_room=%s extra_key=%s "
                "tree_total_before=%s tree_evictable_before=%s "
                "tree_protected_before=%s",
                debug_context,
                getattr(req, "rid", None),
                kv_committed_len_before,
                kv_allocated_len_before,
                is_insert,
                effective_is_insert,
                skip_radix_cache_insert,
                cache_disable,
                cache_disable_finished_insert,
                getattr(req, "bootstrap_host", None),
                getattr(req, "bootstrap_room", None),
                _short_repr(getattr(req, "extra_key", None)),
                cache_total_before,
                cache_evictable_before,
                cache_protected_before,
            )
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

    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
    )

    if log_cache_debug:
        cache_total_after = _safe_cache_stat(tree_cache, "total_size")
        cache_evictable_after = _safe_cache_stat(tree_cache, "evictable_size")
        cache_protected_after = _safe_cache_stat(tree_cache, "protected_size")
        logger.info(
            "release_kv_cache_debug context=%s rid=%s req_pool_idx_before=%s "
            "req_pool_idx_after=%s kv_committed_len_before=%s "
            "kv_committed_len_after=%s kv_allocated_len_before=%s "
            "kv_allocated_len_after=%s is_insert=%s effective_is_insert=%s "
            "skip_radix_cache_insert=%s cache_disable=%s "
            "cache_disable_finished_insert=%s bootstrap_host=%s bootstrap_port=%s "
            "bootstrap_room=%s routed_dp_rank=%s disagg_prefill_dp_rank=%s "
            "origin_input_len=%s output_len=%s fill_len=%s cached_tokens=%s "
            "cache_protected_len=%s prefix_indices_len=%s extra_key=%s "
            "tree_total_before=%s tree_total_after=%s "
            "tree_evictable_before=%s tree_evictable_after=%s "
            "tree_protected_before=%s tree_protected_after=%s",
            debug_context,
            getattr(req, "rid", None),
            req_pool_idx_before,
            req.req_pool_idx,
            kv_committed_len_before,
            req.kv_committed_len,
            kv_allocated_len_before,
            req.kv_allocated_len,
            is_insert,
            effective_is_insert,
            skip_radix_cache_insert,
            cache_disable,
            cache_disable_finished_insert,
            getattr(req, "bootstrap_host", None),
            getattr(req, "bootstrap_port", None),
            getattr(req, "bootstrap_room", None),
            getattr(req, "routed_dp_rank", None),
            getattr(req, "disagg_prefill_dp_rank", None),
            len(getattr(req, "origin_input_ids", [])),
            len(getattr(req, "output_ids", [])),
            len(getattr(req, "fill_ids", [])),
            getattr(req, "cached_tokens", None),
            getattr(req, "cache_protected_len", None),
            len(getattr(req, "prefix_indices", [])),
            _short_repr(getattr(req, "extra_key", None)),
            cache_total_before,
            cache_total_after,
            cache_evictable_before,
            cache_evictable_after,
            cache_protected_before,
            cache_protected_after,
        )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # and bookkeeping flag sync internally, then sets req_pool_idx = None.
    if req.req_pool_idx is None:
        return

    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not global_server_args.strip_thinking_cache:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
