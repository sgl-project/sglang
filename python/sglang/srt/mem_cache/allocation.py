from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.memory.common import (
    get_last_loc_triton,
    get_last_loc_triton_safe,
    write_req_to_token_pool_triton,
)
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
from sglang.srt.runtime_context import attention_backends, get_parallel
from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
    support_triton,
)
from sglang.srt.utils.common import is_pin_memory_available

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()
_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import assign_req_to_token_pool_cpu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)


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
    # This writer's one caller is `alloc_for_extend`, so the prefill half
    # decides; the fallback below pays several `.item()` syncs per request.
    prefill_backend, _ = attention_backends()
    if support_triton(prefill_backend):
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
    prefill_backend, decode_backend = attention_backends()
    uses_triton_dispatch = prefill_backend not in (
        "ascend",
        "torch_native",
    ) and decode_backend not in ("ascend", "torch_native")

    if (_is_hip or _is_npu) and uses_triton_dispatch:
        # HIP and NPU DSV4: the legacy get_last_loc_triton kernel emits a
        # mixed-width int32->int64 store that Triton mis-compiles on HIP,
        # producing out-of-range last_loc values under EAGLE +
        # page_size>1 (e.g. with aiter unified attention or the triton
        # attention backend), and can fault in the equivalent NPU DSV4
        # allocation path. Route those paths through the variant whose
        # in-kernel result remains int32 and is promoted only after launch.
        # Other hardware/backends keep the original dispatcher below.
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
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

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

    return out_cache_loc


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    req_pool_indices: Optional[torch.Tensor] = None,
    batch=None,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    is_dsv4 = req_pool_indices is not None and hasattr(allocator, "c128_attn_allocator")
    extra_alloc_kwargs = {}
    if is_dsv4:
        extra_alloc_kwargs["req_pool_indices"] = req_pool_indices
        # Per-call per-req table for the C128 KV last_loc lookup.
        if batch is not None:
            extra_alloc_kwargs["req_to_token_pool"] = batch.req_to_token_pool

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

    return out_cache_loc


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
                tree_cache.evict_for_alloc(
                    EvictParams(num_tokens=0, mamba_num=mamba_num)
                )
    req_pool_indices = req_to_token_pool.alloc(reqs)
    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, {num_reqs=}, "
        )
    return req_pool_indices


def _alloc_page_size(batch: ScheduleBatch) -> int:
    # DCP swaps in an allocator whose page_size is the configured page_size *
    # dcp_size, so it can be > 1 even when tree_cache.page_size is 1; branch on
    # the real allocator's page_size there. Elsewhere the two are equal.
    if (_is_hip or _is_cuda) and get_parallel().dcp_enabled:
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

    reuse_kv = None
    if batch.is_dllm():
        reuse_kv = [r.kv.holds_kv and bool(r.dllm_incomplete_ids) for r in batch.reqs]

    # Create tensors for allocation
    pin_memory = is_pin_memory_available(batch.device)
    prefix_lens_cpu = torch.tensor(
        batch.prefix_lens, dtype=torch.int64, pin_memory=pin_memory
    )
    extend_lens_cpu = torch.tensor(
        batch.extend_lens, dtype=torch.int64, pin_memory=pin_memory
    )
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots (raises RuntimeError if the pool is exhausted)
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(
        req_pool_indices, dtype=torch.int64, pin_memory=pin_memory
    )
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    alloc_page_size = _alloc_page_size(batch)
    if reuse_kv is not None and any(reuse_kv):
        out_cache_loc = _alloc_extend_loc_with_kv_reuse(
            batch,
            reuse_kv,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            extend_lens_cpu,
            req_pool_indices_device,
            alloc_page_size,
        )
    elif alloc_page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.full((1,), -1, device=batch.device))
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
            req_pool_indices=req_pool_indices_device,
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
    try:
        batch.req_to_token_pool.alloc_aux_to_lengths(
            req_pool_indices_cpu=req_pool_indices_cpu,
            target_seq_lens_cpu=batch.seq_lens_cpu,
        )
    except Exception:
        batch.tree_cache.token_to_kv_pool_allocator.free(out_cache_loc)
        raise

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_extend(
            batch,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            batch.seq_lens_cpu,
        )

    for req, seq_len in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
        req.kv.kv_allocated_len = seq_len
        req.kv.kv_committed_len = seq_len

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def _alloc_extend_loc_with_kv_reuse(
    batch: ScheduleBatch,
    reuse_kv: list[bool],
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    req_pool_indices_device: torch.Tensor,
    alloc_page_size: int,
) -> torch.Tensor:
    device = batch.device
    req_to_token = batch.req_to_token_pool.req_to_token

    for i, req in enumerate(batch.reqs):
        if not reuse_kv[i]:
            continue
        prefix_len = int(prefix_lens_cpu[i])
        extend_len = int(extend_lens_cpu[i])
        retained_len = len(req.dllm_incomplete_ids)
        if extend_len != retained_len:
            raise RuntimeError("dLLM FDFO retained KV must be reused as a full block.")
        if prefix_len + extend_len > req.kv.kv_allocated_len:
            raise RuntimeError("dLLM FDFO retained KV is missing.")

    alloc_extend_lens = [
        0 if reuse_kv[i] else int(extend_lens_cpu[i]) for i in range(len(reuse_kv))
    ]
    alloc_extend_num_tokens = sum(alloc_extend_lens)

    fresh_slots = None
    if alloc_extend_num_tokens > 0:
        if alloc_page_size == 1:
            fresh_slots = alloc_token_slots(batch.tree_cache, alloc_extend_num_tokens)
        else:
            alloc_seq_lens_cpu = torch.tensor(
                [
                    (
                        int(prefix_lens_cpu[i])
                        if reuse_kv[i]
                        else int(batch.seq_lens_cpu[i])
                    )
                    for i in range(len(reuse_kv))
                ],
                dtype=torch.int64,
            )
            last_loc = [
                (t[-1:] if len(t) > 0 else torch.tensor([-1], device=device))
                for t in (r.prefix_indices for r in batch.reqs)
            ]
            fresh_slots = alloc_paged_token_slots_extend(
                tree_cache=batch.tree_cache,
                prefix_lens=prefix_lens_cpu.to(device, non_blocking=True),
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=alloc_seq_lens_cpu.to(device, non_blocking=True),
                seq_lens_cpu=alloc_seq_lens_cpu,
                last_loc=torch.cat(last_loc),
                extend_num_tokens=alloc_extend_num_tokens,
                req_pool_indices=req_pool_indices_device,
                batch=batch,
            )

    reuse_dtype = fresh_slots.dtype if fresh_slots is not None else torch.int64
    parts: list[torch.Tensor] = []
    fresh_ptr = 0
    for i in range(len(reuse_kv)):
        prefix_len = int(prefix_lens_cpu[i])
        extend_len = int(extend_lens_cpu[i])
        if reuse_kv[i]:
            req_idx = int(req_pool_indices_cpu[i])
            parts.append(
                req_to_token[req_idx, prefix_len : prefix_len + extend_len].to(
                    reuse_dtype
                )
            )
        else:
            parts.append(fresh_slots[fresh_ptr : fresh_ptr + extend_len])
            fresh_ptr += extend_len

    return torch.cat(parts)


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
    req_pool_indices: Optional[torch.Tensor] = None,
    batch=None,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    # DSV4-NPU allocator also needs req_pool_indices for C128 KV allocation and
    # returns a DSV4OutCacheLoc bundle; hasattr-gated so others stay unchanged.
    is_dsv4 = req_pool_indices is not None and hasattr(allocator, "c128_attn_allocator")
    extra_alloc_kwargs = {}
    if is_dsv4:
        extra_alloc_kwargs["req_pool_indices"] = req_pool_indices
        # Per-call per-req C128 table for the last_loc lookup.
        if batch is not None:
            extra_alloc_kwargs["req_to_token_pool"] = batch.req_to_token_pool

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

    batch.maybe_evict_swa()

    seq_lens_gpu = batch.seq_lens
    bs = seq_lens_gpu.shape[0]

    if _alloc_page_size(batch) == 1:
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

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_decode(
            batch,
            batch.seq_lens_cpu + token_per_req,
            token_per_req,
        )

    try:
        batch.req_to_token_pool.alloc_aux_to_lengths(
            req_pool_indices_cpu=batch.req_pool_indices_cpu,
            target_seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
        )
    except Exception:
        batch.tree_cache.token_to_kv_pool_allocator.free(out_cache_loc)
        raise

    for req in batch.reqs:
        req.kv.kv_allocated_len += token_per_req
        req.kv.kv_committed_len += token_per_req

    return out_cache_loc


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

    nxt_kv_lens_list = nxt_kv_lens_cpu.tolist()
    for req, nxt_kv_len in zip(reqs, nxt_kv_lens_list, strict=True):
        req.kv.kv_allocated_len = max(req.kv.kv_allocated_len, nxt_kv_len)
