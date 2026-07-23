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
from sglang.QuantKernel.gpu_flush_int2 import (
    gpu_flush_int2_apply,
    gpu_flush_int2_plan,
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
from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
    support_triton,
)
from sglang.srt.utils.common import ceil_align, is_pin_memory_available

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()
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
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
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


def _is_mixed_kv_enabled(batch: ScheduleBatch) -> bool:
    kvcache = batch.token_to_kv_pool_allocator.get_kvcache()
    return isinstance(kvcache, UnifiedInt2HPKVPool) and kvcache.mixed_kv_enabled()


def _mixed_window_lengths(
    seq_len: int, hp_prefix_tokens: int, hp_recent_tokens: int
) -> tuple[int, int, int]:
    prefix_len = min(seq_len, hp_prefix_tokens)
    recent_len = min(max(seq_len - prefix_len, 0), hp_recent_tokens)
    quant_len = seq_len - prefix_len - recent_len
    return prefix_len, recent_len, quant_len


def _mixed_extend_layout_counts(
    pre_len: int,
    seq_len: int,
    hp_prefix_tokens: int,
    hp_recent_tokens: int,
    n_q: int,
    is_final_chunk: bool = True,
) -> tuple[int, int, int, int, int]:
    """Return per-request mixed-KV extend counts.

    Default (final chunk) layout: ``[HP-prefix][quant-middle][HP-recent]``.
    Non-final chunks collapse to ``[HP-prefix][quant-all]`` and skip
    HP-recent — chunk N's trailing positions are never the request's final
    HP-recent window, so allocating HP-recent there only wraps the
    per-request ring and recycles slots whose K/V is still referenced via
    ``req_to_token`` at earlier chunk N positions.
    """
    if is_final_chunk:
        prefix_keep, recent_keep, _ = _mixed_window_lengths(
            seq_len, hp_prefix_tokens, hp_recent_tokens
        )
        recent_start = seq_len - recent_keep
        hp_prefix_count = max(0, min(prefix_keep, seq_len) - pre_len)
        quant_count = max(0, recent_start - max(pre_len, prefix_keep))
        hp_recent_count = max(0, seq_len - max(pre_len, recent_start))
    else:
        prefix_keep = min(seq_len, hp_prefix_tokens)
        hp_prefix_count = max(0, min(prefix_keep, seq_len) - pre_len)
        quant_count = max(0, seq_len - max(pre_len, prefix_keep))
        hp_recent_count = 0
    quant_alloc_count = ceil_align(quant_count, n_q)

    # Per-request flush counter: count steps until this request's next flush.
    # ``H_0`` is the current HP-recent size after this extend chunk is admitted.
    # For non-final chunks we skip HP-recent entirely (``hp_recent_count=0``),
    # and for chunked-prefill final chunks the actual HP-recent count is
    # ``hp_recent_count`` (positions newly admitted in this chunk), NOT the
    # ``hp_recent_tokens`` target. If we use the target as ``H_0`` while the
    # actual count is small, ``counter_init`` underestimates how many decode
    # steps must elapse before the flush kernel's demote range
    # ``[seq_len - hp_recent - flush_overflow : seq_len - hp_recent]`` falls
    # entirely inside HP-recent. Early flushes hit the chunk-1 quant region
    # (valid=0 for all 8 slots → whole-page return). Worse, at the *last*
    # premature flush before the demote range crosses into HP-recent, the
    # 8-position window straddles the boundary: 3-5 valid=0 and 3-5 valid=1
    # → page partially used + partially returned. The used slots get written
    # to ``req_to_token``; the returned slots go to the free pool. When the
    # request finishes, ``cache_finished_req`` frees the used slots → same
    # page added to the free pool TWICE. That is the +N quant-page duplicate
    # the strict idle leak check trips on.
    h0_total = hp_recent_count
    counter_init = max(0, (hp_recent_tokens + n_q - 1) - h0_total)
    return (
        hp_prefix_count,
        hp_recent_count,
        quant_count,
        quant_alloc_count,
        counter_init,
    )


def _alloc_for_extend_mixed(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_pool = batch.token_to_kv_pool_allocator.get_kvcache()
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    n_q = int(kv_pool.N_Q)
    hp_recent_target = int(kv_pool.hp_recent_tokens)

    hp_prefix_counts: list[int] = []
    hp_recent_counts: list[int] = []
    quant_counts: list[int] = []
    quant_alloc_counts: list[int] = []
    flush_counter_inits: list[int] = []
    seq_lens_cpu_list = batch.seq_lens_cpu.tolist()
    is_final_chunk_list = [
        int(seq_lens_cpu_list[i]) >= len(batch.reqs[i].origin_input_ids)
        for i in range(len(batch.reqs))
    ]
    for i, (pre_len, seq_len) in enumerate(zip(batch.prefix_lens, seq_lens_cpu_list)):
        (
            hp_prefix_count,
            hp_recent_count,
            quant_count,
            quant_alloc_count,
            counter_init,
        ) = _mixed_extend_layout_counts(
            int(pre_len),
            int(seq_len),
            int(kv_pool.hp_prefix_tokens),
            hp_recent_target,
            n_q,
            is_final_chunk=bool(is_final_chunk_list[i]),
        )

        hp_prefix_counts.append(hp_prefix_count)
        hp_recent_counts.append(hp_recent_count)
        quant_counts.append(quant_count)
        quant_alloc_counts.append(quant_alloc_count)
        flush_counter_inits.append(counter_init)

    total_quant_alloc = sum(quant_alloc_counts)
    # HP-prefix is N_Q-paged in the shared pool; round each per-req count up.
    hp_prefix_alloc_counts = [ceil_align(int(c), n_q) for c in hp_prefix_counts]
    total_hp_prefix_alloc = sum(hp_prefix_alloc_counts)

    allocator = batch.token_to_kv_pool_allocator
    pooled_need = total_quant_alloc + total_hp_prefix_alloc
    if pooled_need > 0:
        evict_from_tree_cache(batch.tree_cache, pooled_need)

    # ``evict_from_tree_cache`` checks total free (quant + HP-prefix). A
    # single call asking for ``pooled_need`` tokens may free mostly HP-prefix
    # when leaves happen to be prefix-heavy, leaving the quant tier short.
    # Loop with a small bound to converge on enough quant pages.
    if (
        total_quant_alloc > 0
        and batch.tree_cache is not None
        and not batch.tree_cache.is_chunk_cache()
    ):
        for _ in range(4):
            free_quant_slots = (
                allocator.free_pages.numel() + allocator.release_pages.numel()
            ) * allocator.N_Q
            if free_quant_slots >= total_quant_alloc:
                break
            result = batch.tree_cache.evict(EvictParams(num_tokens=total_quant_alloc))
            if result.num_tokens_evicted == 0:
                break

    quant_alloc = (
        allocator.alloc_quant(total_quant_alloc)
        if total_quant_alloc > 0
        else torch.empty((0,), dtype=torch.int64, device=batch.device)
    )
    if total_quant_alloc > 0 and quant_alloc is None:
        raise RuntimeError(
            "Mixed KV windows failed to allocate quant slots after eviction. "
            f"{allocator.debug_print()}"
        )
    hp_prefix_alloc = allocator.alloc_hp_prefix(
        req_pool_indices_device, hp_prefix_alloc_counts
    )
    hp_recent_alloc = allocator.alloc_hp_recent(
        req_pool_indices_device, hp_recent_counts
    )

    per_req_locs = []
    hp_prefix_pt = 0
    hp_recent_pt = 0
    quant_pt = 0
    for i_req, (
        req,
        pre_len,
        hp_prefix_count,
        hp_prefix_alloc_count,
        hp_recent_count,
        quant_count,
        quant_alloc_count,
    ) in enumerate(
        zip(
            batch.reqs,
            batch.prefix_lens,
            hp_prefix_counts,
            hp_prefix_alloc_counts,
            hp_recent_counts,
            quant_counts,
            quant_alloc_counts,
        )
    ):
        # Non-final chunked-prefill chunk: cap insert at the start of the
        # request-owned tail so cache_unfinished_req / cache_finished_req
        # don't insert the tail into the radix tree.
        if not is_final_chunk_list[i_req]:
            non_final_tail_start = max(
                0, int(seq_lens_cpu_list[i_req]) - int(kv_pool.hp_recent_tokens)
            )
            existing_cutoff = req.mixed_kv_quant_slack_cutoff_len
            req.mixed_kv_quant_slack_cutoff_len = (
                non_final_tail_start
                if existing_cutoff is None
                else min(existing_cutoff, non_final_tail_start)
            )
        req_parts = []
        req_hp_prefix = None
        req_hp_recent = None
        if hp_prefix_count > 0:
            req_hp_prefix = hp_prefix_alloc[
                hp_prefix_pt : hp_prefix_pt + hp_prefix_count
            ]
        if hp_prefix_alloc_count > hp_prefix_count:
            # HP-prefix slack: trailing slots of a partially-filled page;
            # request-owned until release, freed by tier-routing in `free`.
            req_hp_prefix_slack = hp_prefix_alloc[
                hp_prefix_pt + hp_prefix_count : hp_prefix_pt + hp_prefix_alloc_count
            ]
            req.mixed_kv_quant_slack_indices = torch.cat(
                [
                    req.mixed_kv_quant_slack_indices.to(batch.device),
                    req_hp_prefix_slack,
                ]
            )
            slack_page_start = int(pre_len) + (hp_prefix_count // n_q) * n_q
            existing_cutoff = req.mixed_kv_quant_slack_cutoff_len
            req.mixed_kv_quant_slack_cutoff_len = (
                slack_page_start
                if existing_cutoff is None
                else min(existing_cutoff, slack_page_start)
            )
        hp_prefix_pt += hp_prefix_alloc_count

        if hp_recent_count > 0:
            req_hp_recent = hp_recent_alloc[
                hp_recent_pt : hp_recent_pt + hp_recent_count
            ]
            hp_recent_pt += hp_recent_count
        if quant_count > 0:
            req_quant = quant_alloc[quant_pt : quant_pt + quant_count]
        else:
            req_quant = None
        if quant_alloc_count > quant_count:
            req_quant_slack = quant_alloc[
                quant_pt + quant_count : quant_pt + quant_alloc_count
            ]
            req.mixed_kv_quant_slack_indices = torch.cat(
                [req.mixed_kv_quant_slack_indices.to(batch.device), req_quant_slack]
            )
            quant_start = max(int(pre_len), int(kv_pool.hp_prefix_tokens))
            slack_page_start = quant_start + (quant_count // n_q) * n_q
            existing_cutoff = req.mixed_kv_quant_slack_cutoff_len
            req.mixed_kv_quant_slack_cutoff_len = (
                slack_page_start
                if existing_cutoff is None
                else min(existing_cutoff, slack_page_start)
            )
        quant_pt += quant_alloc_count

        # Reconstruct logical order: [hp-prefix][quant-middle][hp-recent].
        if req_hp_prefix is not None:
            req_parts.append(req_hp_prefix)
        if req_quant is not None:
            req_parts.append(req_quant)
        if req_hp_recent is not None:
            req_parts.append(req_hp_recent)
        valid_parts = [part for part in req_parts if part is not None]
        per_req_locs.append(
            torch.cat(valid_parts)
            if valid_parts
            else torch.empty((0,), dtype=torch.int64, device=batch.device)
        )

    out_cache_loc = (
        torch.cat(per_req_locs)
        if per_req_locs
        else torch.empty((0,), dtype=torch.int64, device=batch.device)
    )
    prefix_tensors = [r.prefix_indices for r in batch.reqs]
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

    # Seed each request's per-request flush counter. This is overwritten on
    # every chunk of a chunked extend (the latest H_0 is what matters), and
    # on the first chunk for a fresh admission. We do a single async H2D
    # copy + scatter so the decode hot path sees the counter without a
    # later sync.
    counter_inits_cpu = torch.tensor(flush_counter_inits, dtype=torch.int32)
    counter_inits_device = counter_inits_cpu.to(batch.device, non_blocking=True)
    kv_pool._flush_counter[req_pool_indices_device] = counter_inits_device

    from sglang.srt.managers.schedule_batch import ReqKvInfo

    for req, seq_len in zip(batch.reqs, seq_lens_cpu_list):
        seq_len = int(seq_len)
        if req.kv is None:
            req.kv = ReqKvInfo(kv_allocated_len=seq_len, swa_evicted_seqlen=0)
        else:
            req.kv.kv_allocated_len = seq_len

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def _alloc_for_decode_mixed(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    # One HP-recent slot per req per step; over-provision bs*N_Q quant slots
    # per step and let the per-req flush counter decide which use them as
    # demote targets vs return them via ``returned_slot_ids``.
    if token_per_req != 1:
        raise NotImplementedError(
            "Mixed KV decode currently supports token_per_req=1 only."
        )

    allocator = batch.token_to_kv_pool_allocator
    kv_pool = allocator.get_kvcache()
    bs = batch.seq_lens.shape[0]
    # ``flush_interval`` is hardcoded to ``N_Q`` in ``UnifiedInt2HPKVPool``.
    flush_interval = int(kv_pool.N_Q)
    assert kv_pool.flush_interval == flush_interval

    req_pool_indices_int64 = batch.req_pool_indices.to(torch.int64)

    # Per-request flush gating: shape-static RMW, no host sync.
    counters = kv_pool._flush_counter[req_pool_indices_int64]
    flush_mask = counters == 0
    new_counters = torch.where(
        flush_mask,
        torch.full_like(counters, flush_interval - 1),
        counters - 1,
    )
    kv_pool._flush_counter[req_pool_indices_int64] = new_counters

    # Worst case: every req flushes -> bs*N_Q quant slots needed.
    quant_need = bs * flush_interval
    # ``evict_from_tree_cache`` gates on ``allocator.available_size()`` which
    # for the unified pool sums quant + HP-prefix free slots. When quant is
    # drained but HP-prefix has slack, the combined check skips eviction and
    # ``alloc_quant`` below crashes. Force quant-tier-specific eviction here.
    quant_pages_have = allocator.free_pages.numel() + allocator.release_pages.numel()
    if (
        quant_pages_have < bs
        and batch.tree_cache is not None
        and not batch.tree_cache.is_chunk_cache()
    ):
        # Tree leaves may be quant or HP-prefix; some leaves are big. Loop a
        # few times in case the first leaves popped are HP-prefix, but cap
        # work so we don't spin if everything left is pinned.
        for attempt in range(8):
            prev_quant = quant_pages_have
            prev_hp = (
                allocator.hp_prefix_free_pages.numel()
                + allocator.hp_prefix_release_pages.numel()
            )
            # Ramp up the budget each attempt: 1x, 2x, 4x ... up to 16x.
            mult = 1 << min(attempt, 4)
            evict_slots = max(bs - quant_pages_have, 1) * flush_interval * mult
            batch.tree_cache.evict(EvictParams(num_tokens=evict_slots))
            quant_pages_have = (
                allocator.free_pages.numel() + allocator.release_pages.numel()
            )
            if quant_pages_have >= bs:
                break
            cur_hp = (
                allocator.hp_prefix_free_pages.numel()
                + allocator.hp_prefix_release_pages.numel()
            )
            if quant_pages_have == prev_quant and cur_hp == prev_hp:
                # Tree had nothing to evict — leaves all pinned. Stop.
                break

    out_cache_loc = allocator.alloc_hp_recent(
        req_pool_indices_int64, [token_per_req] * bs
    )

    dst_quant_slots = allocator.alloc_quant(quant_need)
    if dst_quant_slots is None:
        raise RuntimeError(
            "Mixed KV windows failed to allocate quant flush slots. "
            f"{allocator.debug_print()}"
        )

    # Build the protected boundary on device in one go.  This is the
    # tree-owned prefix that flush must not overwrite; ``prefix_indices`` may
    # additionally contain request-owned tail slots for chunk continuation.
    prefix_lens_cpu = torch.tensor(
        [int(r.cache_protected_len) for r in batch.reqs], dtype=torch.int32
    )
    prefix_lens_gpu = prefix_lens_cpu.to(batch.device, non_blocking=True)
    seq_lens_int32 = batch.seq_lens.to(torch.int32)

    # Phase 1 (no-race with previous forward): plan kernel reads
    # ``req_to_token`` and produces ``returned_slot_ids`` etc. Followed by
    # ``allocator.free``, whose ``torch.unique`` host-syncs only against this
    # short pre-wait prefix instead of the previous forward.
    plan = gpu_flush_int2_plan(
        seq_lens=seq_lens_int32,
        prefix_lens=prefix_lens_gpu,
        req_pool_indices=req_pool_indices_int64,
        dst_quant_slots=dst_quant_slots,
        req_to_token=batch.req_to_token_pool.req_to_token,
        flush_mask=flush_mask,
        hp_prefix_tokens=kv_pool.hp_prefix_tokens,
        hp_recent_tokens=kv_pool.hp_recent_tokens,
        hp_global_offset=kv_pool.hp_global_offset,
        flush_interval=flush_interval,
    )

    if plan is not None:
        # Free everything returned by the kernel in one call: flushed HP
        # slots (freed from HP tier) and unused quant slots from
        # non-flushing requests (whole pages, since per-request
        # all-or-nothing). The allocator decodes tier from each global slot
        # id.
        allocator.free(plan.returned_slot_ids)

    # Phase 2 (must wait): the apply kernels write ``req_to_token`` at
    # positions inside the previous forward's read range. Order
    # schedule_stream after ``forward_done`` here, not at the top of the
    # event loop, so the host syncs above and any retract/eviction frees
    # don't stall behind the previous forward.
    kv_pool.wait_pending_forward()

    if plan is not None:
        gpu_flush_int2_apply(
            plan,
            req_pool_indices=req_pool_indices_int64,
            req_to_token=batch.req_to_token_pool.req_to_token,
            hp_k_ptrs=kv_pool._flush_hp_k_ptrs,
            hp_v_ptrs=kv_pool._flush_hp_v_ptrs,
            quant_k_ptrs=kv_pool._flush_quant_k_ptrs,
            quant_v_ptrs=kv_pool._flush_quant_v_ptrs,
            k_sz_ptrs=kv_pool._flush_k_sz_ptrs,
            v_sz_ptrs=kv_pool._flush_v_sz_ptrs,
            hp_k_sample=kv_pool.hp_k_buffer[0],
            hp_v_sample=kv_pool.hp_v_buffer[0],
            quant_k_sample=kv_pool.k_buffer[0],
            quant_v_sample=kv_pool.v_buffer[0],
            k_sz_sample=kv_pool.k_scales_zeros[0],
            v_sz_sample=kv_pool.v_scales_zeros[0],
            hp_k_strides=kv_pool._flush_hp_k_stride,
            hp_v_strides=kv_pool._flush_hp_v_stride,
            quant_k_strides=kv_pool._flush_quant_k_stride,
            quant_v_strides=kv_pool._flush_quant_v_stride,
            k_sz_strides=kv_pool._flush_k_sz_stride,
            v_sz_strides=kv_pool._flush_v_sz_stride,
            num_heads=kv_pool.head_num,
            head_dim=kv_pool.head_dim,
            v_head_dim=kv_pool.v_head_dim,
            k_num_scale_groups=kv_pool.k_num_scale_groups,
            v_num_scale_groups=kv_pool.v_num_scale_groups,
            num_layers=kv_pool.layer_num,
            k_clip_ratio=kv_pool._k_clip_ratio,
            v_clip_ratio=kv_pool._v_clip_ratio,
        )

    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + batch.seq_lens
    else:
        locs = batch.seq_lens.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    for req in batch.reqs:
        req.kv.kv_allocated_len += token_per_req

    return out_cache_loc


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

    if _is_mixed_kv_enabled(batch):
        return _alloc_for_extend_mixed(batch)

    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    reuse_kv = None
    if batch.is_dllm():
        reuse_kv = [
            r.req_pool_idx is not None and bool(r.dllm_incomplete_ids)
            for r in batch.reqs
        ]

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

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_extend(
            batch,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            batch.seq_lens_cpu,
        )

    from sglang.srt.managers.schedule_batch import ReqKvInfo

    for req, seq_len in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
        if req.kv is None:
            req.kv = ReqKvInfo(kv_allocated_len=seq_len, swa_evicted_seqlen=0)
        else:
            req.kv.kv_allocated_len = seq_len

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
        if req.kv is None or prefix_len + extend_len > req.kv.kv_allocated_len:
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
                dsv4_state_lens=_compute_dsv4_state_lens(batch, is_decode=False),
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

    batch.maybe_evict_swa()

    if _is_mixed_kv_enabled(batch):
        return _alloc_for_decode_mixed(batch, token_per_req)

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

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_decode(
            batch,
            batch.seq_lens_cpu + token_per_req,
            token_per_req,
        )

    for req in batch.reqs:
        req.kv.kv_allocated_len += token_per_req

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

    for i, req in enumerate(reqs):
        req.kv.kv_allocated_len = max(req.kv.kv_allocated_len, int(nxt_kv_lens_cpu[i]))
