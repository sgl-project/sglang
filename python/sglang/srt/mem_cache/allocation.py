from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.kernels.ops.memory.req_to_token_pool import (
    AssignExtendCacheLocs,
    AssignReqToTokenPool,
    WriteReqToTokenPool,
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
from sglang.srt.utils import support_triton
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    alloc_starts_tensor: torch.Tensor,
    alloc_starts_cpu: torch.Tensor,
    alloc_ends_tensor: torch.Tensor,
    alloc_ends_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
) -> None:
    WriteReqToTokenPool.execute(
        req_to_token_pool.req_to_token,
        req_pool_indices=req_pool_indices_tensor,
        req_pool_indices_cpu=req_pool_indices_cpu,
        prefix_lens=prefix_lens_tensor,
        prefix_lens_cpu=prefix_lens_cpu,
        alloc_starts=alloc_starts_tensor,
        alloc_starts_cpu=alloc_starts_cpu,
        alloc_ends=alloc_ends_tensor,
        alloc_ends_cpu=alloc_ends_cpu,
        prefix_tensors=prefix_tensors,
        out_cache_loc=out_cache_loc,
        use_triton=support_triton(get_server_args().attention_backend),
    )


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


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns ``(out_cache_loc, req_pool_indices_device, req_pool_indices_cpu)``
    (the last is the host/CPU mirror). ``alloc_req_slots`` raises ``RuntimeError``
    if the pool can't satisfy the batch (fail-loud — see its docstring).
    """
    allocator = batch.token_to_kv_pool_allocator
    if allocator.uses_legacy_real_length_alloc:
        from sglang.srt.hardware_backend.npu.allocation_legacy import (
            alloc_for_extend_legacy,
        )

        return alloc_for_extend_legacy(batch)

    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    plan = _plan_extend_alloc(
        reqs=batch.reqs,
        prefix_lens=batch.prefix_lens,
        seq_lens=batch.seq_lens_cpu.tolist(),
        page_size=allocator.page_size,
    )

    prefix_tensors = [r.prefix_indices for r in batch.reqs]
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    alloc_starts_device = plan.alloc_starts_cpu.to(batch.device, non_blocking=True)
    alloc_ends_device = plan.alloc_ends_cpu.to(batch.device, non_blocking=True)

    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    new_pages = _alloc_new_pages(
        batch.tree_cache, need_size=plan.need_size, oom_label="Prefill"
    )

    write_cache_indices(
        out_cache_loc=new_pages,
        req_pool_indices_tensor=req_pool_indices_device,
        req_pool_indices_cpu=req_pool_indices_cpu,
        prefix_lens_tensor=prefix_lens_device,
        prefix_lens_cpu=prefix_lens_cpu,
        alloc_starts_tensor=alloc_starts_device,
        alloc_starts_cpu=plan.alloc_starts_cpu,
        alloc_ends_tensor=alloc_ends_device,
        alloc_ends_cpu=plan.alloc_ends_cpu,
        prefix_tensors=prefix_tensors,
        req_to_token_pool=batch.req_to_token_pool,
    )

    out_cache_loc = AssignExtendCacheLocs.execute(
        batch.req_to_token_pool.req_to_token,
        req_pool_indices=req_pool_indices_device,
        req_pool_indices_cpu=req_pool_indices_cpu,
        start_offset=prefix_lens_device,
        start_offset_cpu=prefix_lens_cpu,
        end_offset=batch.seq_lens,
        end_offset_cpu=batch.seq_lens_cpu,
        out_tokens=batch.extend_num_tokens,
        batch_size=len(batch.reqs),
        device=batch.device,
        ragged=True,
    )

    _record_extend_allocation(reqs=batch.reqs, alloc_ends=plan.alloc_ends_cpu.tolist())

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """
    allocator = batch.token_to_kv_pool_allocator
    if allocator.uses_legacy_real_length_alloc:
        from sglang.srt.hardware_backend.npu.allocation_legacy import (
            alloc_for_decode_legacy,
        )

        return alloc_for_decode_legacy(batch, token_per_req)

    batch.maybe_evict_swa()

    locs_cpu, locs_device = _decode_write_positions(batch)
    plan = _plan_decode_alloc(
        reqs=batch.reqs,
        locs=locs_cpu.tolist(),
        token_per_req=token_per_req,
        page_size=allocator.page_size,
    )

    if plan.need_size > 0:
        new_pages = _alloc_new_pages(
            batch.tree_cache,
            need_size=plan.need_size,
            oom_label="Decode",
            logical_only=True,
        )
        _write_new_pages(
            req_to_token=batch.req_to_token_pool.req_to_token,
            req_pool_indices=batch.req_pool_indices,
            req_pool_indices_cpu=batch.req_pool_indices_cpu,
            alloc_starts=plan.alloc_starts_cpu.to(batch.device, non_blocking=True),
            alloc_starts_cpu=plan.alloc_starts_cpu,
            alloc_ends=plan.alloc_ends_cpu.to(batch.device, non_blocking=True),
            alloc_ends_cpu=plan.alloc_ends_cpu,
            out_cache_loc=new_pages,
            batch_size=len(batch.reqs),
        )

    out_cache_loc = AssignExtendCacheLocs.execute(
        batch.req_to_token_pool.req_to_token,
        req_pool_indices=batch.req_pool_indices,
        req_pool_indices_cpu=batch.req_pool_indices_cpu,
        start_offset=locs_device,
        start_offset_cpu=locs_cpu,
        end_offset=locs_device + token_per_req,
        end_offset_cpu=locs_cpu + token_per_req,
        batch_size=len(batch.reqs),
        out_tokens=len(batch.reqs) * token_per_req,
        device=batch.device,
        ragged=False,
    )

    for req, alloc_end in zip(batch.reqs, plan.alloc_ends_cpu.tolist()):
        req.kv.kv_allocated_len = alloc_end

    return out_cache_loc


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
    batch: Optional[ScheduleBatch] = None,
) -> None:
    allocator = tree_cache.token_to_kv_pool_allocator
    if allocator.uses_legacy_real_length_alloc:
        from sglang.srt.hardware_backend.npu.allocation_legacy import (
            alloc_for_spec_decode_legacy,
        )

        alloc_for_spec_decode_legacy(
            tree_cache,
            req_to_token_pool,
            reqs=reqs,
            req_pool_indices=req_pool_indices,
            cur_kv_lens=cur_kv_lens,
            cur_kv_lens_cpu=cur_kv_lens_cpu,
            nxt_kv_lens=nxt_kv_lens,
            nxt_kv_lens_cpu=nxt_kv_lens_cpu,
            batch=batch,
        )
        return

    page_size = allocator.page_size
    alloc_ends_cpu = _ceil_tensor_to_page(nxt_kv_lens_cpu, page_size)
    need_size = int((alloc_ends_cpu - cur_kv_lens_cpu).sum())

    if need_size > 0:
        new_pages = _alloc_new_pages(
            tree_cache, need_size=need_size, oom_label="Speculative decode"
        )
        # Updating req_to_token is a write to a shared tensor: it must not overlap
        # with the previous batch's forward, which also reads req_to_token.
        _write_new_pages(
            req_to_token=req_to_token_pool.req_to_token,
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=batch.req_pool_indices_cpu,
            alloc_starts=cur_kv_lens,
            alloc_starts_cpu=cur_kv_lens_cpu,
            alloc_ends=_ceil_tensor_to_page(nxt_kv_lens, page_size),
            alloc_ends_cpu=alloc_ends_cpu,
            out_cache_loc=new_pages,
            batch_size=len(reqs),
        )

    for req, alloc_end in zip(reqs, alloc_ends_cpu.tolist()):
        req.kv.kv_allocated_len = max(req.kv.kv_allocated_len, alloc_end)


class AllocPlan(msgspec.Struct):
    alloc_starts_cpu: torch.Tensor
    alloc_ends_cpu: torch.Tensor
    need_size: int


def _plan_extend_alloc(
    *, reqs: list[Req], prefix_lens: list[int], seq_lens: list[int], page_size: int
) -> AllocPlan:
    alloc_starts: list[int] = []
    alloc_ends: list[int] = []
    need_size = 0

    for req, prefix_len, seq_len in zip(reqs, prefix_lens, seq_lens):
        allocated_old = req.kv.kv_allocated_len if req.kv is not None else 0
        alloc_start = max(prefix_len, allocated_old)
        alloc_end = max(allocated_old, ceil_align(seq_len, page_size))
        assert alloc_start % page_size == 0, (prefix_len, allocated_old, page_size)
        alloc_starts.append(alloc_start)
        alloc_ends.append(alloc_end)
        need_size += alloc_end - alloc_start

    return AllocPlan(
        alloc_starts_cpu=torch.tensor(alloc_starts, dtype=torch.int64),
        alloc_ends_cpu=torch.tensor(alloc_ends, dtype=torch.int64),
        need_size=need_size,
    )


def _plan_decode_alloc(
    *, reqs: list[Req], locs: list[int], token_per_req: int, page_size: int
) -> AllocPlan:
    alloc_starts: list[int] = []
    alloc_ends: list[int] = []
    need_size = 0

    for req, loc in zip(reqs, locs):
        allocated_old = req.kv.kv_allocated_len
        alloc_end = max(allocated_old, ceil_align(loc + token_per_req, page_size))
        assert allocated_old % page_size == 0, (allocated_old, page_size)
        alloc_starts.append(allocated_old)
        alloc_ends.append(alloc_end)
        need_size += alloc_end - allocated_old

    return AllocPlan(
        alloc_starts_cpu=torch.tensor(alloc_starts, dtype=torch.int64),
        alloc_ends_cpu=torch.tensor(alloc_ends, dtype=torch.int64),
        need_size=need_size,
    )


def _decode_write_positions(batch: ScheduleBatch) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.model_config.is_encoder_decoder:
        encoder_lens_cpu = torch.tensor(batch.encoder_lens_cpu, dtype=torch.int64)
        return (
            batch.seq_lens_cpu + encoder_lens_cpu,
            batch.encoder_lens + batch.seq_lens,
        )
    return batch.seq_lens_cpu, batch.seq_lens


def _record_extend_allocation(*, reqs: list[Req], alloc_ends: list[int]) -> None:
    from sglang.srt.managers.schedule_batch import ReqKvInfo

    for req, alloc_end in zip(reqs, alloc_ends):
        if req.kv is None:
            req.kv = ReqKvInfo(kv_allocated_len=alloc_end, swa_evicted_seqlen=0)
        else:
            req.kv.kv_allocated_len = alloc_end


def _alloc_new_pages(
    tree_cache: BasePrefixCache,
    *,
    need_size: int,
    oom_label: str,
    logical_only: bool = False,
) -> torch.Tensor:
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, need_size)

    alloc = allocator.alloc_logical_only if logical_only else allocator.alloc
    out_cache_loc = alloc(need_size)

    if out_cache_loc is None:
        error_msg = (
            f"{oom_label} out of memory. Try to lower your batch size.\n"
            f"Try to allocate {need_size} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def _write_new_pages(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    alloc_starts: torch.Tensor,
    alloc_starts_cpu: torch.Tensor,
    alloc_ends: torch.Tensor,
    alloc_ends_cpu: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
) -> None:
    if not support_triton(get_server_args().attention_backend):
        _write_new_pages_torch(
            req_to_token=req_to_token,
            req_pool_indices_cpu=req_pool_indices_cpu,
            alloc_starts_cpu=alloc_starts_cpu,
            alloc_ends_cpu=alloc_ends_cpu,
            out_cache_loc=out_cache_loc,
            batch_size=batch_size,
        )
        return

    AssignReqToTokenPool.execute(
        req_to_token,
        req_pool_indices=req_pool_indices,
        start_offset=alloc_starts,
        end_offset=alloc_ends,
        out_cache_loc=out_cache_loc,
        batch_size=batch_size,
    )


def _write_new_pages_torch(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    alloc_starts_cpu: torch.Tensor,
    alloc_ends_cpu: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
) -> None:
    out_cache_offset = 0
    for index in range(batch_size):
        alloc_start = int(alloc_starts_cpu[index])
        alloc_end = int(alloc_ends_cpu[index])
        alloc_len = alloc_end - alloc_start
        req_to_token[int(req_pool_indices_cpu[index]), alloc_start:alloc_end] = (
            out_cache_loc[out_cache_offset : out_cache_offset + alloc_len]
        )
        out_cache_offset += alloc_len


def _ceil_tensor_to_page(lens: torch.Tensor, page_size: int) -> torch.Tensor:
    return (lens + page_size - 1) // page_size * page_size
