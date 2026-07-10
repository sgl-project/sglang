from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

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
from sglang.srt.mem_cache.triton_ops.common import (
    gather_req_to_token_pool_triton,
    get_last_loc_triton,
    get_last_loc_triton_safe,
    write_req_to_token_pool_triton,
)
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import is_cuda, is_hip, is_npu, next_power_of_2, support_triton
from sglang.srt.utils.common import ceil_align, is_pin_memory_available

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()

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
    if support_triton(get_server_args().attention_backend):
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            dtype=torch.uint64,
            pin_memory=is_pin_memory_available(req_to_token_pool.device),
        ).to(req_to_token_pool.device, non_blocking=True)
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_write_lens_tensor,
            alloc_start_lens_tensor,
            alloc_end_lens_tensor,
            alloc_extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_write_len = prefix_write_lens_cpu[i].item()
            alloc_start = alloc_start_lens_cpu[i].item()
            alloc_end = alloc_end_lens_cpu[i].item()
            alloc_extend_len = alloc_extend_lens_cpu[i].item()

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_write_len)),
                prefix_tensors[i],
            )
            # The gap [prefix_write_len, alloc_start) is intentionally left
            # unwritten: it holds slots written by a previous chunk of the same
            # request, whose values are already in the pool.
            req_to_token_pool.write(
                (req_idx, slice(alloc_start, alloc_end)),
                out_cache_loc[pt : pt + alloc_extend_len],
            )
            pt += alloc_extend_len


def gather_out_cache_loc_extend(
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    req_to_token_pool: ReqToTokenPool,
) -> torch.Tensor:
    req_to_token = req_to_token_pool.req_to_token
    num_reqs = req_pool_indices_cpu.shape[0]
    num_out_tokens = int(extend_lens_cpu.sum().item())

    assert req_pool_indices_tensor.shape[0] == num_reqs
    assert prefix_lens_tensor.shape[0] == prefix_lens_cpu.shape[0] == num_reqs
    assert seq_lens_tensor.shape[0] == seq_lens_cpu.shape[0] == num_reqs
    assert extend_lens_tensor.shape[0] == extend_lens_cpu.shape[0] == num_reqs
    assert (
        req_pool_indices_cpu.device.type
        == prefix_lens_cpu.device.type
        == seq_lens_cpu.device.type
        == extend_lens_cpu.device.type
        == "cpu"
    )
    assert torch.equal(seq_lens_cpu - prefix_lens_cpu, extend_lens_cpu)
    assert req_to_token.dtype == torch.int32
    assert (
        req_pool_indices_tensor.device
        == prefix_lens_tensor.device
        == seq_lens_tensor.device
        == extend_lens_tensor.device
        == req_to_token.device
    )

    if support_triton(get_server_args().attention_backend):
        # The kernel output buffer stays int32 (req_to_token dtype) and is
        # promoted here after the kernel returns, so Triton never issues a
        # mixed-width store (HIP miscompiles int32->int64 stores; see
        # get_last_loc_triton_safe).
        out_cache_loc_i32 = torch.empty(
            num_out_tokens, dtype=req_to_token.dtype, device=req_to_token.device
        )
        gather_req_to_token_pool_triton[(num_reqs,)](
            req_to_token,
            req_pool_indices_tensor,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc_i32,
            req_to_token.shape[1],
        )
        return out_cache_loc_i32.to(torch.int64)

    if num_reqs == 0:
        return torch.empty(0, dtype=torch.int64, device=req_to_token.device)

    chunks: list[torch.Tensor] = []
    for i in range(num_reqs):
        req_idx = req_pool_indices_cpu[i].item()
        prefix_len = prefix_lens_cpu[i].item()
        seq_len = seq_lens_cpu[i].item()
        chunks.append(req_to_token[req_idx, prefix_len:seq_len])
    return torch.cat(chunks).to(torch.int64)


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
    # DCP swaps in an allocator whose page_size is server_args.page_size *
    # dcp_size, so it can be > 1 even when tree_cache.page_size is 1; branch on
    # the real allocator's page_size there. Elsewhere the two are equal.
    if (_is_hip or _is_cuda) and get_server_args().dcp_size > 1:
        return batch.tree_cache.token_to_kv_pool_allocator.page_size
    return batch.tree_cache.page_size


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns ``(out_cache_loc, req_pool_indices_device, req_pool_indices_cpu)``
    (the last is the host/CPU mirror). ``alloc_req_slots`` raises ``RuntimeError``
    if the pool can't satisfy the batch (fail-loud — see its docstring).
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots (raises RuntimeError if the pool is exhausted)
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocator-facing lens: the alloc interval per request is
    # [allocated_old, ceil_align(seq)), page-aligned on both ends, while the
    # forward interval stays [prefix, seq). DSV4-NPU keeps real lens because
    # its side pools (swa/c4/c128/state) track the real-lens watermark.
    page_size = _alloc_page_size(batch)
    seq_lens_list: list[int] = batch.seq_lens_cpu.tolist()
    if _is_npu:
        alloc_start_lens_list: list[int] = list(batch.prefix_lens)
        alloc_end_lens_list: list[int] = seq_lens_list
    else:
        alloc_start_lens_list = [
            req.kv.kv_allocated_len if req.kv is not None else prefix_len
            for req, prefix_len in zip(batch.reqs, batch.prefix_lens)
        ]
        alloc_end_lens_list = [
            ceil_align(seq_len, page_size) for seq_len in seq_lens_list
        ]
    for alloc_start_len, alloc_end_len in zip(
        alloc_start_lens_list, alloc_end_lens_list
    ):
        assert alloc_start_len <= alloc_end_len, (
            f"alloc interval is negative: {alloc_start_len=}, {alloc_end_len=}, "
            f"{page_size=}"
        )
    alloc_start_lens_cpu = torch.tensor(alloc_start_lens_list, dtype=torch.int64)
    alloc_end_lens_cpu = torch.tensor(alloc_end_lens_list, dtype=torch.int64)
    alloc_extend_lens_cpu = alloc_end_lens_cpu - alloc_start_lens_cpu
    alloc_extend_num_tokens = int(alloc_extend_lens_cpu.sum().item())
    alloc_start_lens_device = alloc_start_lens_cpu.to(batch.device, non_blocking=True)
    alloc_end_lens_device = alloc_end_lens_cpu.to(batch.device, non_blocking=True)
    alloc_extend_lens_device = alloc_extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, alloc_extend_num_tokens)
    else:
        # Paged allocation - build last_loc at position alloc_start - 1. Fresh
        # requests (req.kv is None) must use the prefix tail: their req_to_token
        # row was just allocated and is stale until write_cache_indices below.
        req_to_token = batch.req_to_token_pool.req_to_token
        last_loc = [
            (
                req_to_token[
                    req.req_pool_idx, alloc_start_len - 1 : alloc_start_len
                ].to(torch.int64)
                if req.kv is not None and alloc_start_len > 0
                else (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            )
            for req, t, alloc_start_len in zip(
                batch.reqs, prefix_tensors, alloc_start_lens_list
            )
        ]
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=alloc_start_lens_device,
            prefix_lens_cpu=alloc_start_lens_cpu,
            seq_lens=alloc_end_lens_device,
            seq_lens_cpu=alloc_end_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=alloc_extend_num_tokens,
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
        alloc_start_lens_device,
        alloc_start_lens_cpu,
        alloc_end_lens_device,
        alloc_end_lens_cpu,
        alloc_extend_lens_device,
        alloc_extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )

    out_cache_loc_derived = gather_out_cache_loc_extend(
        req_pool_indices_tensor=req_pool_indices_device,
        req_pool_indices_cpu=req_pool_indices_cpu,
        prefix_lens_tensor=prefix_lens_device,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens_tensor=batch.seq_lens,
        seq_lens_cpu=batch.seq_lens_cpu,
        extend_lens_tensor=extend_lens_device,
        extend_lens_cpu=extend_lens_cpu,
        req_to_token_pool=batch.req_to_token_pool,
    )
    if envs.SGLANG_DEBUG_MEMORY_POOL.get():
        # The alloc interval [alloc_start, alloc_end) and the forward interval
        # [prefix, seq) legitimately differ, so the gather cannot be compared
        # against out_cache_loc; only check that every gathered slot is a real
        # (non-padding, non-stale-zero) pool index.
        assert torch.all(out_cache_loc_derived > 0)

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_extend(
            batch,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            batch.seq_lens_cpu,
        )

    from sglang.srt.managers.schedule_batch import ReqKvInfo

    for req, alloc_end_len in zip(batch.reqs, alloc_end_lens_list):
        if req.kv is None:
            req.kv = ReqKvInfo(kv_allocated_len=alloc_end_len, swa_evicted_seqlen=0)
        else:
            req.kv.kv_allocated_len = alloc_end_len

    return out_cache_loc_derived, req_pool_indices_device, req_pool_indices_cpu


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

    assert token_per_req == 1

    batch.maybe_evict_swa()

    seq_lens_gpu = batch.seq_lens
    bs = seq_lens_gpu.shape[0]

    page_size = _alloc_page_size(batch)
    if page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, seq_lens_gpu - 1
        ]
        seq_lens_next = seq_lens_gpu + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
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

    if page_size > 1 and not _is_npu:
        # Page-crossing steps get a whole fresh page from the allocator (the
        # new slot is the page head since seq % page == 0 there); publish the
        # full page [slot, slot + page) into the request's row so that
        # [0, kv_allocated_len) stays fully written. Row positions use `locs`
        # (encoder-offset aware); the crossing test uses the decoder KV length.
        crossing_idx_cpu = torch.nonzero(batch.seq_lens_cpu % page_size == 0).flatten()
        if crossing_idx_cpu.numel() > 0:
            crossing_idx = crossing_idx_cpu.to(batch.device, non_blocking=True)
            if envs.SGLANG_DEBUG_MEMORY_POOL.get():
                assert torch.all(out_cache_loc[crossing_idx] % page_size == 0)
            page_offsets = torch.arange(
                page_size, dtype=torch.int64, device=batch.device
            )
            crossing_rows = batch.req_pool_indices[crossing_idx]
            crossing_positions = locs[crossing_idx].unsqueeze(1) + page_offsets
            crossing_values = out_cache_loc[crossing_idx].unsqueeze(1) + page_offsets
            batch.req_to_token_pool.write(
                (crossing_rows.unsqueeze(1), crossing_positions),
                crossing_values.to(torch.int32),
            )

    out_cache_loc_derived = batch.req_to_token_pool.req_to_token[
        batch.req_pool_indices, locs
    ].to(torch.int64)
    if envs.SGLANG_DEBUG_MEMORY_POOL.get():
        assert torch.equal(out_cache_loc_derived, out_cache_loc)

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_decode(
            batch,
            batch.seq_lens_cpu + token_per_req,
            token_per_req,
        )

    # ceil_align with the NPU real-lens page (1) degenerates to the previous
    # += token_per_req bookkeeping because allocated == seq before a decode.
    bookkeeping_page_size = 1 if _is_npu else page_size
    for req, seq_len in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
        req.kv.kv_allocated_len = max(
            req.kv.kv_allocated_len,
            ceil_align(seq_len + token_per_req, bookkeeping_page_size),
        )

    return out_cache_loc_derived


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
    if num_needed_tokens > 0:
        if tree_cache.token_to_kv_pool_allocator.page_size == 1:
            out_cache_loc = alloc_token_slots(tree_cache, num_needed_tokens)
        else:
            last_loc = get_last_loc(
                req_to_token_pool.req_to_token, req_pool_indices, cur_kv_lens
            )
            device_type = getattr(
                batch.device, "type", str(batch.device).split(":", 1)[0]
            )
            out_cache_loc = ALLOC_EXTEND_FUNCS[device_type](
                tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                nxt_kv_lens,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
                req_pool_indices=req_pool_indices,
                batch=batch,
            )
        # Updating req_to_token is a write to a shared tensor: it must not overlap
        # with the previous batch's forward, which also reads req_to_token.
        assign_req_to_token_pool_func(
            req_pool_indices,
            req_to_token_pool.req_to_token,
            cur_kv_lens,
            nxt_kv_lens,
            out_cache_loc,
            len(reqs),
        )

    for i, req in enumerate(reqs):
        req.kv.kv_allocated_len = max(req.kv.kv_allocated_len, int(nxt_kv_lens_cpu[i]))
