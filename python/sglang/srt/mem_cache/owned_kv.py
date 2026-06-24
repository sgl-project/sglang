from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_write_dsv4_decode,
    maybe_write_dsv4_extend,
)
from sglang.srt.mem_cache.kv_cache_utils import write_cache_indices
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.req_state import ReqKvInfo
from sglang.srt.utils import is_npu

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.free_space import FreeSpaceProvider
    from sglang.srt.model_executor.forward_batch_info import DSV4StateLens

logger = logging.getLogger(__name__)


def free_swa_out_of_window_slots(
    *,
    req_pool_idx: int,
    kv: ReqKvInfo,
    owned_start: int,
    pre_len: int,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    on_swa_evicted: Callable[[int], None],
    drop_page_margin: bool = False,
) -> None:
    from sglang.srt.environ import envs

    # For swa radix cache, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert owned_start % page_size == 0, "cache_protected_len must be page aligned"
    kv.swa_evicted_seqlen = max(kv.swa_evicted_seqlen, owned_start)

    # Subtract an extra page_size so the eviction frontier never reaches the
    # radix tree insert boundary (page_floor(seq_len)). This keeps at least one
    # page of non-evicted SWA KV for the tree to store as a non-tombstone node,
    # preserving cache reuse in multi-turn scenarios. Without this, leaf nodes
    # may become tombstoned, causing SWA memory leak.
    # See also: _insert_helper case 3 in swa_radix_cache.py (defensive counterpart).
    if drop_page_margin or envs.SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN.get():
        evict_threshold = pre_len - sliding_window_size
    else:
        evict_threshold = pre_len - sliding_window_size - page_size
    new_swa_evicted_seqlen = max(
        kv.swa_evicted_seqlen,
        evict_threshold,
    )

    if page_size > 1:
        new_swa_evicted_seqlen = (new_swa_evicted_seqlen // page_size) * page_size

    if new_swa_evicted_seqlen > kv.swa_evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req_pool_idx, kv.swa_evicted_seqlen : new_swa_evicted_seqlen
        ]
        token_to_kv_pool_allocator.free_swa(free_slots)
        on_swa_evicted(new_swa_evicted_seqlen)
        kv.swa_evicted_seqlen = new_swa_evicted_seqlen


def alloc_token_slots(
    allocator: BaseTokenToKVPoolAllocator,
    num_tokens: int,
    *,
    space: FreeSpaceProvider,
    backup_state: bool = False,
):
    space.ensure_free(num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{space.describe_for_oom()}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def _compute_dsv4_state_lens(batch, *, is_decode: bool):
    """Per-req c{4,128}_state pool alloc lens (a ``DSV4StateLens``) for this
    alloc step. The DSV4-NPU allocator owns the computation (it also mutates the
    per-req cumulative state on each ``Req``); we just trigger it here, right
    before the paged alloc that consumes the result.

    None on CUDA / non-V4 paths (allocator has no ``compute_dsv4_state_lens_*``)
    so the ``alloc_paged_token_slots_*`` forwarding stays a no-op.
    """
    allocator = batch.token_to_kv_pool_allocator
    if not hasattr(allocator, "compute_dsv4_state_lens_extend"):
        return None
    if is_decode:
        return allocator.compute_dsv4_state_lens_decode(batch.reqs)
    return allocator.compute_dsv4_state_lens_extend(
        batch.reqs, batch.seq_lens_cpu.tolist()
    )


def alloc_paged_token_slots_extend(
    allocator: BaseTokenToKVPoolAllocator,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    *,
    space: FreeSpaceProvider,
    backup_state: bool = False,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional[DSV4StateLens] = None,
    batch=None,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    space.ensure_free(num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    is_dsv4 = req_pool_indices is not None and hasattr(allocator, "c4_attn_allocator")
    extra_alloc_kwargs = {}
    if is_dsv4:
        extra_alloc_kwargs["req_pool_indices"] = req_pool_indices
        # Pass the per-req tables in per call for the c-pool / state last_loc
        # lookup; the allocator holds no reference to the pool.
        if batch is not None:
            extra_alloc_kwargs["req_to_token_pool"] = batch.req_to_token_pool
        if dsv4_state_lens is not None:
            extra_alloc_kwargs["dsv4_state_lens"] = dsv4_state_lens

    out = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        **extra_alloc_kwargs,
    )

    if is_dsv4:
        bundle = out
        out_cache_loc = None if bundle is None else bundle.out_full_loc
        if batch is not None:
            batch.out_cache_loc_dsv4 = bundle
    else:
        out_cache_loc = out

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{space.describe_for_oom()}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    reserve_req_state_slots: Callable[[int], None],
) -> list[int]:
    """Allocate request slots from the pool."""
    num_reqs = len(reqs)
    reserve_req_state_slots(num_reqs)
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
    *,
    space: FreeSpaceProvider,
    reserve_req_state_slots: Callable[[int], None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
        req_pool_indices_device: request pool indices as a device tensor
        req_pool_indices_cpu: request pool indices as a CPU tensor (host mirror)
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, reserve_req_state_slots
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if batch.token_to_kv_pool_allocator.page_size == 1:
        out_cache_loc = alloc_token_slots(
            batch.token_to_kv_pool_allocator,
            batch.extend_num_tokens,
            space=space,
        )
    else:
        # Paged allocation - build last_loc
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        out_cache_loc = alloc_paged_token_slots_extend(
            allocator=batch.token_to_kv_pool_allocator,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=batch.extend_num_tokens,
            space=space,
            req_pool_indices=req_pool_indices_device,
            dsv4_state_lens=_compute_dsv4_state_lens(batch, is_decode=False),
            batch=batch,
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

    # DSV4-NPU hook: write c4/c128/swa per-req tables from the stashed bundle.
    # No-op on non-DSV4 paths (out_cache_loc_dsv4 stays None there).
    if _is_npu:
        maybe_write_dsv4_extend(
            batch,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            batch.seq_lens_cpu,
        )

    for req, seq_len in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
        if req.kv is None:
            req.kv = ReqKvInfo(kv_allocated_len=seq_len, swa_evicted_seqlen=0)
        else:
            req.kv.kv_allocated_len = seq_len

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def alloc_paged_token_slots_decode(
    allocator: BaseTokenToKVPoolAllocator,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    *,
    space: FreeSpaceProvider,
    token_per_req: int = 1,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional[DSV4StateLens] = None,
    batch=None,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    space.ensure_free(num_tokens)

    # DSV4-NPU allocator also needs req_pool_indices + per-req state lens and
    # returns a DSV4OutCacheLoc bundle; hasattr-gated so others stay unchanged.
    is_dsv4 = req_pool_indices is not None and hasattr(allocator, "c4_attn_allocator")
    extra_alloc_kwargs = {}
    if is_dsv4:
        extra_alloc_kwargs["req_pool_indices"] = req_pool_indices
        # Per-call per-req tables for the last_loc lookup; the allocator holds
        # no reference to the pool.
        if batch is not None:
            extra_alloc_kwargs["req_to_token_pool"] = batch.req_to_token_pool
        if dsv4_state_lens is not None:
            extra_alloc_kwargs["dsv4_state_lens"] = dsv4_state_lens

    out = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc, **extra_alloc_kwargs)

    if is_dsv4:
        bundle = out
        out_cache_loc = None if bundle is None else bundle.out_full_loc
        if batch is not None:
            batch.out_cache_loc_dsv4 = bundle
    else:
        out_cache_loc = out

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{space.describe_for_oom()}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(
    batch: ScheduleBatch,
    token_per_req: int,
    *,
    space: FreeSpaceProvider,
) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """

    batch.maybe_evict_swa()

    seq_lens_gpu = batch.seq_lens
    bs = seq_lens_gpu.shape[0]

    if batch.token_to_kv_pool_allocator.page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(
            batch.token_to_kv_pool_allocator,
            bs * token_per_req,
            space=space,
        )
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, seq_lens_gpu - 1
        ]
        seq_lens_next = seq_lens_gpu + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            allocator=batch.token_to_kv_pool_allocator,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            space=space,
            token_per_req=token_per_req,
            req_pool_indices=batch.req_pool_indices,
            dsv4_state_lens=_compute_dsv4_state_lens(batch, is_decode=True),
            batch=batch,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + seq_lens_gpu
    else:
        locs = seq_lens_gpu.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    # DSV4-NPU hook: post-decode write of c4/c128/swa per-req tables from the
    # stashed bundle. No-op on non-DSV4 paths (out_cache_loc_dsv4 stays None).
    if _is_npu:
        maybe_write_dsv4_decode(
            batch,
            batch.seq_lens_cpu + token_per_req,
            token_per_req,
        )

    for req in batch.reqs:
        req.kv.kv_allocated_len += token_per_req

    return out_cache_loc


def init_decode_prealloc_kv(req: Req, fill_len: int) -> None:
    if req.kv is None:
        req.kv = ReqKvInfo(kv_allocated_len=fill_len, swa_evicted_seqlen=0)
    else:
        req.kv.kv_allocated_len = fill_len


def alloc_for_decode_prealloc(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: Req,
    fill_len: int,
    delta_len: int,
    prefix_len: int,
    total_prefix_len: int,
    prefix_indices: Optional[torch.Tensor],
    uses_swa_tail: bool,
    swa_tail_len: int,
) -> torch.Tensor:
    init_decode_prealloc_kv(req, fill_len)

    if allocator.page_size == 1:
        kv_loc = allocator.alloc(delta_len)
    else:
        device = allocator.device
        last_loc = (
            prefix_indices[-1:].to(dtype=torch.int64, device=device)
            if prefix_len > 0
            else torch.tensor([-1], dtype=torch.int64, device=device)
        )
        if uses_swa_tail:
            # Tail-only SWA allocation: only valid when prefix_len == 0.
            # When prefix_len > 0 (radix cache hit), we fall back to
            # alloc_extend which allocates SWA at full page count; the
            # SWA budget in that case may slightly under-estimate.
            kv_loc = allocator.alloc_extend_swa_tail(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=fill_len,
                swa_tail_len=swa_tail_len,
            )
            req.kv.swa_evicted_seqlen = fill_len - swa_tail_len
        else:
            kv_loc = allocator.alloc_extend(
                prefix_lens=torch.tensor(
                    [total_prefix_len], dtype=torch.int64, device=device
                ),
                prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=delta_len,
            )
    return kv_loc
