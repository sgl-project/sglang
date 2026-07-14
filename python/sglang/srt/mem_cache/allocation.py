from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.memory.common import (
    get_last_loc_triton,
    get_last_loc_triton_safe,
)
from sglang.kernels.ops.memory.req_to_token_pool import (
    GatherReqToTokenPool,
    WriteReqToTokenPool,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_write_dsv4_decode,
    maybe_write_dsv4_extend,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.common import (
    MAMBA_STATE_PER_REQ_NO_CACHE,
    MAMBA_STATE_PER_REQ_PREFIX_CACHE,
    MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY,
    available_and_evictable_str,
    evict_from_tree_cache,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import (
    is_cpu,
    is_hip,
    is_npu,
    next_power_of_2,
    support_triton,
)
from sglang.srt.utils.common import ceil_align

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import assign_req_to_token_pool_cpu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import DSV4StateLens

logger = logging.getLogger(__name__)


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_write_lens_tensor: torch.Tensor,
    prefix_write_lens_cpu: torch.Tensor,
    alloc_start_lens_tensor: torch.Tensor,
    alloc_start_lens_cpu: torch.Tensor,
    alloc_end_lens_tensor: torch.Tensor,
    alloc_end_lens_cpu: torch.Tensor,
    alloc_extend_lens_tensor: torch.Tensor,
    alloc_extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
) -> None:
    WriteReqToTokenPool.execute(
        req_to_token_pool.req_to_token,
        req_pool_indices=req_pool_indices_tensor,
        req_pool_indices_cpu=req_pool_indices_cpu,
        prefix_write_lens=prefix_write_lens_tensor,
        prefix_write_lens_cpu=prefix_write_lens_cpu,
        alloc_start_lens=alloc_start_lens_tensor,
        alloc_start_lens_cpu=alloc_start_lens_cpu,
        alloc_end_lens=alloc_end_lens_tensor,
        alloc_end_lens_cpu=alloc_end_lens_cpu,
        alloc_extend_lens=alloc_extend_lens_tensor,
        alloc_extend_lens_cpu=alloc_extend_lens_cpu,
        prefix_tensors=prefix_tensors,
        out_cache_loc=out_cache_loc,
        use_triton=support_triton(get_server_args().attention_backend),
    )


def gather_out_cache_loc_extend(
    req_to_token_pool: ReqToTokenPool,
    *,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    extend_num_tokens: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    req_to_token: torch.Tensor = req_to_token_pool.req_to_token
    return GatherReqToTokenPool.execute(
        req_to_token,
        req_pool_indices=req_pool_indices_tensor,
        req_pool_indices_cpu=req_pool_indices_cpu,
        prefix_lens=prefix_lens_tensor,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=seq_lens_tensor,
        seq_lens_cpu=seq_lens_cpu,
        extend_lens=extend_lens_tensor,
        extend_lens_cpu=extend_lens_cpu,
        extend_num_tokens=extend_num_tokens,
        out_dtype=out_dtype,
        use_triton=support_triton(get_server_args().attention_backend),
    )


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    attn_backend = get_server_args().attention_backend
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


def _compute_dsv4_state_lens(batch, *, is_decode: bool):
    """Per-req c{4,128}_state pool alloc lens (``DSV4StateLens``) for this step.
    None on CUDA / non-V4 paths (allocator has no ``compute_dsv4_state_lens_*``).
    """
    allocator = batch.token_to_kv_pool_allocator
    if not hasattr(allocator, "compute_dsv4_state_lens_extend"):
        return None
    if is_decode:
        return allocator.compute_dsv4_state_lens_decode(batch.reqs)
    return allocator.compute_dsv4_state_lens_extend(
        batch.reqs, batch.seq_lens_cpu.tolist()
    )


def assert_alloc_extend_lens_page_aligned(
    prefix_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_num_tokens: int,
    page_size: int,
) -> None:
    if _is_npu or page_size == 1:
        return

    assert bool(torch.all(prefix_lens_cpu % page_size == 0)), (
        f"alloc_extend prefix lens must be page-aligned: "
        f"{prefix_lens_cpu=}, {page_size=}"
    )
    assert bool(
        torch.all(seq_lens_cpu % page_size == 0)
    ), f"alloc_extend seq lens must be page-aligned: {seq_lens_cpu=}, {page_size=}"
    assert extend_num_tokens % page_size == 0, (
        f"alloc_extend token count must be page-aligned: "
        f"{extend_num_tokens=}, {page_size=}"
    )


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional[DSV4StateLens] = None,
    batch=None,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    assert_alloc_extend_lens_page_aligned(
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=extend_num_tokens,
        page_size=allocator.page_size,
    )
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    is_dsv4 = req_pool_indices is not None and hasattr(allocator, "c4_attn_allocator")
    extra_alloc_kwargs = {}
    if is_dsv4:
        extra_alloc_kwargs["req_pool_indices"] = req_pool_indices
        # Per-call per-req tables for the c-pool / state last_loc lookup.
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
    """Allocate request slots from the pool.

    Fail-loud: raises ``RuntimeError`` if the pool can't satisfy the batch. An
    alloc failure here means the admission budget (``PrefillAdder``) was wrong
    and should surface rather than be masked.
    """
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        # Byte-coordinated for the shared allocator (accounts for the peer full
        # sub-pool's bytes); plain slot free count for the non-shared one.
        mamba_available_size = (
            req_to_token_pool.mamba_allocator.schedulable_available_size()
        )
        # Eviction headroom factor: 3x (or lazy variant) for radix COW, 1x for chunk.
        if tree_cache.supports_mamba():
            factor = (
                MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY
                if req_to_token_pool.enable_mamba_extra_buffer_lazy
                else MAMBA_STATE_PER_REQ_PREFIX_CACHE
            )
        else:
            factor = MAMBA_STATE_PER_REQ_NO_CACHE
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
            f"{req_to_token_pool.available_size()=}, {num_reqs=}, "
        )
    return req_pool_indices


def _alloc_page_size(batch: ScheduleBatch) -> int:
    return batch.tree_cache.token_to_kv_pool_allocator.page_size


def _validate_main_page_aligned_alloc(batch: ScheduleBatch) -> None:
    allocator = batch.tree_cache.token_to_kv_pool_allocator
    if not _is_npu and allocator.page_size > 1:
        allocator.validate_main_page_aligned_alloc()


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns ``(out_cache_loc, req_pool_indices_device, req_pool_indices_cpu)``
    (the last is the host/CPU mirror). ``alloc_req_slots`` raises ``RuntimeError``
    if the pool can't satisfy the batch (fail-loud — see its docstring).
    """
    _validate_main_page_aligned_alloc(batch)

    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    prefix_tensors: list[torch.Tensor] = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu: torch.Tensor = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu: torch.Tensor = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device: torch.Tensor = prefix_lens_cpu.to(
        batch.device, non_blocking=True
    )
    extend_lens_device: torch.Tensor = extend_lens_cpu.to(
        batch.device, non_blocking=True
    )
    alloc_page_size: int = _alloc_page_size(batch)
    uses_aligned_lens: bool = not _is_npu and alloc_page_size > 1
    alloc_start_lens_cpu: torch.Tensor
    alloc_end_lens_cpu: torch.Tensor
    alloc_extend_lens_cpu: torch.Tensor
    alloc_start_lens_device: torch.Tensor
    alloc_end_lens_device: torch.Tensor
    alloc_extend_lens_device: torch.Tensor
    if uses_aligned_lens:
        alloc_start_lens: list[int] = [
            max(
                prefix_len,
                req.kv.kv_allocated_len if req.kv is not None else 0,
            )
            for req, prefix_len in zip(batch.reqs, batch.prefix_lens)
        ]
        alloc_end_lens: list[int] = [
            ceil_align(seq_len, alloc_page_size)
            for seq_len in batch.seq_lens_cpu.tolist()
        ]
        alloc_extend_lens: list[int] = [
            alloc_end - alloc_start
            for alloc_start, alloc_end in zip(alloc_start_lens, alloc_end_lens)
        ]
        assert all(extend_len >= 0 for extend_len in alloc_extend_lens)
        alloc_start_lens_cpu = torch.tensor(alloc_start_lens, dtype=torch.int64)
        alloc_end_lens_cpu = torch.tensor(alloc_end_lens, dtype=torch.int64)
        alloc_extend_lens_cpu = torch.tensor(alloc_extend_lens, dtype=torch.int64)
        alloc_start_lens_device = alloc_start_lens_cpu.to(
            batch.device, non_blocking=True
        )
        alloc_end_lens_device = alloc_end_lens_cpu.to(batch.device, non_blocking=True)
        alloc_extend_lens_device = alloc_extend_lens_cpu.to(
            batch.device, non_blocking=True
        )
    else:
        alloc_start_lens_cpu = prefix_lens_cpu
        alloc_end_lens_cpu = batch.seq_lens_cpu
        alloc_extend_lens_cpu = extend_lens_cpu
        alloc_start_lens_device = prefix_lens_device
        alloc_end_lens_device = batch.seq_lens
        alloc_extend_lens_device = extend_lens_device
    alloc_extend_num_tokens: int = int(alloc_extend_lens_cpu.sum().item())

    # Allocate req slots (raises RuntimeError if the pool is exhausted)
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if alloc_page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        out_cache_loc = alloc_token_slots(
            tree_cache=batch.tree_cache,
            num_tokens=alloc_extend_num_tokens,
        )

    # Write to req_to_token_pool
    write_cache_indices(
        out_cache_loc=out_cache_loc,
        req_pool_indices_tensor=req_pool_indices_device,
        req_pool_indices_cpu=req_pool_indices_cpu,
        prefix_write_lens_tensor=prefix_lens_device,
        prefix_write_lens_cpu=prefix_lens_cpu,
        alloc_start_lens_tensor=alloc_start_lens_device,
        alloc_start_lens_cpu=alloc_start_lens_cpu,
        alloc_end_lens_tensor=alloc_end_lens_device,
        alloc_end_lens_cpu=alloc_end_lens_cpu,
        alloc_extend_lens_tensor=alloc_extend_lens_device,
        alloc_extend_lens_cpu=alloc_extend_lens_cpu,
        prefix_tensors=prefix_tensors,
        req_to_token_pool=batch.req_to_token_pool,
    )

    gathered: torch.Tensor = gather_out_cache_loc_extend(
        batch.req_to_token_pool,
        req_pool_indices_tensor=req_pool_indices_device,
        req_pool_indices_cpu=req_pool_indices_cpu,
        prefix_lens_tensor=prefix_lens_device,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens_tensor=batch.seq_lens,
        seq_lens_cpu=batch.seq_lens_cpu,
        extend_lens_tensor=extend_lens_device,
        extend_lens_cpu=extend_lens_cpu,
        extend_num_tokens=batch.extend_num_tokens,
        out_dtype=out_cache_loc.dtype,
    )
    if envs.SGLANG_DEBUG_MEMORY_POOL.get():
        assert gathered.numel() == batch.extend_num_tokens
        assert bool(torch.all(gathered != 0))
        if uses_aligned_lens:
            for req_pool_index, alloc_start, alloc_end in zip(
                req_pool_indices,
                alloc_start_lens_cpu.tolist(),
                alloc_end_lens_cpu.tolist(),
            ):
                allocated_slots = batch.req_to_token_pool.req_to_token[
                    req_pool_index, alloc_start:alloc_end
                ]
                assert bool(torch.all(allocated_slots != 0))
    out_cache_loc = gathered

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_extend(
            batch,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            batch.seq_lens_cpu,
        )

    from sglang.srt.managers.schedule_batch import ReqKvInfo

    for req, allocated_len in zip(batch.reqs, alloc_end_lens_cpu.tolist()):
        if req.kv is None:
            req.kv = ReqKvInfo(
                kv_allocated_len=allocated_len,
                swa_evicted_seqlen=0,
            )
        else:
            req.kv.kv_allocated_len = allocated_len

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional[DSV4StateLens] = None,
    batch=None,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    # DSV4-NPU allocator also needs req_pool_indices + per-req state lens and
    # returns a DSV4OutCacheLoc bundle; hasattr-gated so others stay unchanged.
    is_dsv4 = req_pool_indices is not None and hasattr(allocator, "c4_attn_allocator")
    extra_alloc_kwargs = {}
    if is_dsv4:
        extra_alloc_kwargs["req_pool_indices"] = req_pool_indices
        # Per-call per-req tables for the last_loc lookup.
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
    _validate_main_page_aligned_alloc(batch)

    seq_lens_gpu = batch.seq_lens
    batch_size = seq_lens_gpu.shape[0]
    alloc_page_size = _alloc_page_size(batch)
    write_locs = _compute_decode_write_locs(batch)
    page_plan: Optional[_PageAlignedDecodePlan] = None
    if not _is_npu and alloc_page_size > 1:
        page_plan = _plan_page_aligned_decode(
            batch,
            write_locs=write_locs,
            token_per_req=token_per_req,
            alloc_page_size=alloc_page_size,
        )

    batch.maybe_evict_swa()

    page_allocation: Optional[_PageAlignedDecodeAllocation] = None
    raw_out_cache_loc: Optional[torch.Tensor]
    if alloc_page_size == 1:
        # Non-paged allocation
        raw_out_cache_loc = alloc_token_slots(
            batch.tree_cache,
            batch_size * token_per_req,
        )
    # Paged allocation
    elif _is_npu:
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, seq_lens_gpu - 1
        ]
        seq_lens_next = seq_lens_gpu + token_per_req
        raw_out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
            req_pool_indices=batch.req_pool_indices,
            dsv4_state_lens=_compute_dsv4_state_lens(batch, is_decode=True),
            batch=batch,
        )
    else:
        assert page_plan is not None
        page_allocation = _allocate_page_aligned_decode(
            batch,
            plan=page_plan,
            alloc_page_size=alloc_page_size,
        )
        raw_out_cache_loc = None

    if alloc_page_size == 1 or _is_npu:
        assert raw_out_cache_loc is not None
        batch.req_to_token_pool.write(
            (batch.req_pool_indices, write_locs.device),
            raw_out_cache_loc.to(torch.int32),
        )
        for req in batch.reqs:
            req.kv.kv_allocated_len += token_per_req
    else:
        assert page_plan is not None
        assert page_allocation is not None
        _publish_page_aligned_decode(
            batch,
            plan=page_plan,
            allocation=page_allocation,
            alloc_page_size=alloc_page_size,
        )

    out_dtype = (
        raw_out_cache_loc.dtype if raw_out_cache_loc is not None else torch.int64
    )
    out_cache_loc = batch.req_to_token_pool.req_to_token[
        batch.req_pool_indices,
        write_locs.device,
    ].to(out_dtype)
    _debug_decode_allocation(
        batch,
        write_locs=write_locs,
        out_cache_loc=out_cache_loc,
        raw_out_cache_loc=raw_out_cache_loc,
        page_plan=page_plan,
        page_allocation=page_allocation,
        alloc_page_size=alloc_page_size,
    )

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_decode(
            batch,
            batch.seq_lens_cpu + token_per_req,
            token_per_req,
        )

    return out_cache_loc


@dataclass(frozen=True)
class _DecodeWriteLocs:
    device: torch.Tensor
    cpu: torch.Tensor


@dataclass(frozen=True)
class _PageAlignedDecodePlan:
    write_locs: _DecodeWriteLocs
    allocated_old_cpu: torch.Tensor
    allocated_next_cpu: torch.Tensor
    crossing_indices_cpu: torch.Tensor


@dataclass(frozen=True)
class _PageAlignedDecodeAllocation:
    crossing_indices: torch.Tensor
    page_blocks: Optional[torch.Tensor]


def _compute_decode_write_locs(batch: ScheduleBatch) -> _DecodeWriteLocs:
    if batch.model_config.is_encoder_decoder:
        assert batch.encoder_lens is not None
        assert batch.encoder_lens_cpu is not None
        encoder_lens_cpu = torch.tensor(
            batch.encoder_lens_cpu,
            dtype=batch.seq_lens_cpu.dtype,
            device=batch.seq_lens_cpu.device,
        )
        return _DecodeWriteLocs(
            device=batch.encoder_lens + batch.seq_lens,
            cpu=encoder_lens_cpu + batch.seq_lens_cpu,
        )
    return _DecodeWriteLocs(
        device=batch.seq_lens.clone(),
        cpu=batch.seq_lens_cpu.clone(),
    )


def _plan_page_aligned_decode(
    batch: ScheduleBatch,
    *,
    write_locs: _DecodeWriteLocs,
    token_per_req: int,
    alloc_page_size: int,
) -> _PageAlignedDecodePlan:
    assert token_per_req == 1
    allocated_old_cpu = torch.tensor(
        [req.kv.kv_allocated_len for req in batch.reqs],
        dtype=write_locs.cpu.dtype,
        device=write_locs.cpu.device,
    )
    assert bool(torch.all(allocated_old_cpu % alloc_page_size == 0)), (
        f"decode allocation watermarks must be page-aligned: "
        f"{allocated_old_cpu=}, {alloc_page_size=}"
    )
    assert bool(torch.all(write_locs.cpu <= allocated_old_cpu)), (
        f"decode write locations exceed allocation watermarks: "
        f"{write_locs.cpu=}, {allocated_old_cpu=}"
    )

    crossing_mask_cpu = write_locs.cpu == allocated_old_cpu
    in_page_mask_cpu = ~crossing_mask_cpu
    assert bool(
        torch.all(
            write_locs.cpu[in_page_mask_cpu] + 1 <= allocated_old_cpu[in_page_mask_cpu]
        )
    ), (
        f"decode in-page writes exceed allocation watermarks: "
        f"{write_locs.cpu=}, {allocated_old_cpu=}"
    )
    crossing_indices_cpu = torch.nonzero(
        crossing_mask_cpu,
        as_tuple=False,
    ).flatten()
    allocated_next_cpu = allocated_old_cpu.clone()
    allocated_next_cpu[crossing_indices_cpu] += alloc_page_size
    return _PageAlignedDecodePlan(
        write_locs=write_locs,
        allocated_old_cpu=allocated_old_cpu,
        allocated_next_cpu=allocated_next_cpu,
        crossing_indices_cpu=crossing_indices_cpu,
    )


def _allocate_page_aligned_decode(
    batch: ScheduleBatch,
    *,
    plan: _PageAlignedDecodePlan,
    alloc_page_size: int,
) -> _PageAlignedDecodeAllocation:
    crossing_indices = plan.crossing_indices_cpu.to(
        device=batch.device,
        non_blocking=True,
    )
    if crossing_indices.numel() == 0:
        return _PageAlignedDecodeAllocation(
            crossing_indices=crossing_indices,
            page_blocks=None,
        )

    page_blocks = alloc_token_slots(
        tree_cache=batch.tree_cache,
        num_tokens=crossing_indices.numel() * alloc_page_size,
    )
    return _PageAlignedDecodeAllocation(
        crossing_indices=crossing_indices,
        page_blocks=page_blocks.reshape(-1, alloc_page_size),
    )


def _publish_page_aligned_decode(
    batch: ScheduleBatch,
    *,
    plan: _PageAlignedDecodePlan,
    allocation: _PageAlignedDecodeAllocation,
    alloc_page_size: int,
) -> None:
    if allocation.page_blocks is not None:
        position_offsets = torch.arange(
            alloc_page_size,
            dtype=plan.write_locs.device.dtype,
            device=batch.device,
        )
        crossing_positions = (
            plan.write_locs.device[allocation.crossing_indices].unsqueeze(1)
            + position_offsets
        )
        crossing_req_pool_indices = batch.req_pool_indices[
            allocation.crossing_indices
        ].unsqueeze(1)
        batch.req_to_token_pool.write(
            (crossing_req_pool_indices, crossing_positions),
            allocation.page_blocks.to(torch.int32),
        )

    for req, allocated_len in zip(batch.reqs, plan.allocated_next_cpu.tolist()):
        req.kv.kv_allocated_len = allocated_len


def _debug_decode_allocation(
    batch: ScheduleBatch,
    *,
    write_locs: _DecodeWriteLocs,
    out_cache_loc: torch.Tensor,
    raw_out_cache_loc: Optional[torch.Tensor],
    page_plan: Optional[_PageAlignedDecodePlan],
    page_allocation: Optional[_PageAlignedDecodeAllocation],
    alloc_page_size: int,
) -> None:
    if not envs.SGLANG_DEBUG_MEMORY_POOL.get():
        return

    assert bool(torch.all(out_cache_loc != 0))
    if alloc_page_size == 1 or _is_npu:
        assert raw_out_cache_loc is not None
        torch.testing.assert_close(out_cache_loc, raw_out_cache_loc, rtol=0, atol=0)
        return

    assert page_plan is not None
    assert page_allocation is not None
    if page_allocation.page_blocks is not None:
        torch.testing.assert_close(
            out_cache_loc[page_allocation.crossing_indices],
            page_allocation.page_blocks[:, 0].to(out_cache_loc.dtype),
            rtol=0,
            atol=0,
        )
    for req in batch.reqs:
        assert req.kv.kv_allocated_len % alloc_page_size == 0
        assert req.kv.kv_allocated_len >= ceil_align(
            req.kv_committed_len,
            alloc_page_size,
        )


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


def assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
):
    if _is_cpu:
        assign_req_to_token_pool_cpu(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
        )
        return
    assign_req_to_token_pool[(batch_size,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        next_power_of_2(batch_size),
    )


def _alloc_paged_token_slots_extend_npu(*args, **kwargs):
    from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
        alloc_paged_token_slots_extend_npu,
    )

    return alloc_paged_token_slots_extend_npu(*args, **kwargs)


ALLOC_EXTEND_FUNCS = defaultdict(
    lambda: alloc_paged_token_slots_extend,
    {"npu": _alloc_paged_token_slots_extend_npu},
)


def alloc_for_spec_decode(
    tree_cache: BasePrefixCache,
    req_to_token_pool: ReqToTokenPool,
    *,
    reqs: list[Req],
    req_pool_indices: torch.Tensor,
    cur_kv_lens: torch.Tensor,
    cur_kv_lens_cpu: torch.Tensor,
    nxt_kv_lens: torch.Tensor,
    nxt_kv_lens_cpu: torch.Tensor,
    num_needed_tokens: int,
    batch: Optional[ScheduleBatch] = None,
) -> None:
    alloc_page_size: int = tree_cache.token_to_kv_pool_allocator.page_size
    alloc_nxt_kv_lens_cpu: torch.Tensor
    alloc_nxt_kv_lens: torch.Tensor
    alloc_num_needed_tokens: int
    if not _is_npu and alloc_page_size > 1:
        alloc_nxt_kv_lens_cpu = (
            (nxt_kv_lens_cpu + alloc_page_size - 1) // alloc_page_size * alloc_page_size
        )
        alloc_nxt_kv_lens = (
            (nxt_kv_lens + alloc_page_size - 1) // alloc_page_size * alloc_page_size
        )
        alloc_num_needed_tokens = int(
            (alloc_nxt_kv_lens_cpu - cur_kv_lens_cpu).sum().item()
        )
    else:
        alloc_nxt_kv_lens_cpu = nxt_kv_lens_cpu
        alloc_nxt_kv_lens = nxt_kv_lens
        alloc_num_needed_tokens = num_needed_tokens

    assert_alloc_extend_lens_page_aligned(
        prefix_lens_cpu=cur_kv_lens_cpu,
        seq_lens_cpu=alloc_nxt_kv_lens_cpu,
        extend_num_tokens=alloc_num_needed_tokens,
        page_size=alloc_page_size,
    )

    if alloc_nxt_kv_lens_cpu.numel() > 0:
        max_allocated_len: int = int(alloc_nxt_kv_lens_cpu.max().item())
        row_width: int = req_to_token_pool.req_to_token.shape[1]
        assert max_allocated_len <= row_width, (
            f"spec decode allocation endpoint ({max_allocated_len}) exceeds "
            f"req_to_token row width ({row_width}); page_size={alloc_page_size}"
        )

    if alloc_num_needed_tokens > 0:
        if alloc_page_size == 1:
            out_cache_loc = alloc_token_slots(
                tree_cache=tree_cache,
                num_tokens=alloc_num_needed_tokens,
            )
        else:
            last_loc = get_last_loc(
                req_to_token_pool.req_to_token, req_pool_indices, cur_kv_lens
            )
            device_type = getattr(
                batch.device, "type", str(batch.device).split(":", 1)[0]
            )
            # TODO(temporary-inside-chain): Replace the non-NPU page-aligned spec producer with its direct hook.
            out_cache_loc = ALLOC_EXTEND_FUNCS[device_type](
                tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                alloc_nxt_kv_lens,
                alloc_nxt_kv_lens_cpu,
                last_loc,
                alloc_num_needed_tokens,
                req_pool_indices=req_pool_indices,
                batch=batch,
            )
        # Updating req_to_token is a write to a shared tensor: it must not overlap
        # with the previous batch's forward, which also reads req_to_token.
        assign_req_to_token_pool_func(
            req_pool_indices,
            req_to_token_pool.req_to_token,
            cur_kv_lens,
            alloc_nxt_kv_lens,
            out_cache_loc,
            len(reqs),
        )

    for i, req in enumerate(reqs):
        req.kv.kv_allocated_len = max(
            req.kv.kv_allocated_len,
            int(alloc_nxt_kv_lens_cpu[i]),
        )
