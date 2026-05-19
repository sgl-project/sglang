"""Triton implementation of the canary plan accumulator.

Replaces the host-side Python loop in
:func:`sglang.jit_kernel.kv_cache_canary_plan_ref.plan_batch_from_forward_batch`
with an on-device Triton kernel that gathers verify slots, gathers
write slots + chain seeds, computes the per-req entry-start prefix sum,
and applies the SWA full→SWA index LUT — all in one kernel launch and
without any D2H / ``.tolist()`` roundtrips.

The wrapper :func:`plan_batch_from_forward_batch` populates a
pre-allocated :class:`BatchPlanGpu` in place and is cuda-graph-capture
safe (only the input tensor *values* change between forwards; addresses
stay fixed). The byte-equal contract against the Python reference is
enforced by ``test/registered/canary/test_unit_plan_kernel.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kv_cache_canary import (
    CANARY_EXPECTED_SKIP_SENTINEL,
)
from sglang.jit_kernel.kv_cache_canary_plan_ref import BatchPlanGpu
from sglang.srt.kv_cache_canary.config import CanaryConfig

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@triton.jit
def _plan_kernel(
    # Input pointers
    req_pool_indices_ptr,
    input_ids_ptr,
    positions_ptr,
    out_cache_loc_ptr,
    prefix_lens_ptr,
    extend_seq_lens_ptr,
    req_to_token_ptr,
    full_to_swa_lut_ptr,
    # Output pointers
    verify_slot_indices_ptr,
    verify_positions_ptr,
    verify_prev_slot_indices_ptr,
    write_slot_indices_ptr,
    write_token_ids_ptr,
    write_positions_ptr,
    expected_write_token_ids_ptr,
    expected_write_positions_ptr,
    write_req_seed_slot_indices_ptr,
    write_req_entry_starts_ptr,
    write_req_entry_counts_ptr,
    verify_num_valid_ptr,
    write_req_num_valid_ptr,
    # Sizes
    bs,
    req_to_token_stride,
    swa_lut_len,
    # Constexprs
    BS_BLOCK: tl.constexpr,
    INNER_BLOCK: tl.constexpr,
    VERIFY_CAPACITY: tl.constexpr,
    WRITE_CAPACITY: tl.constexpr,
    WRITE_REQ_CAPACITY: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    HAS_SWA_LUT: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    SKIP_SENTINEL: tl.constexpr,
):
    r = tl.program_id(0)

    bs_offs = tl.arange(0, BS_BLOCK)
    bs_mask = bs_offs < bs
    rpi_all = tl.load(req_pool_indices_ptr + bs_offs, mask=bs_mask, other=0)
    prefix_lens_all = tl.load(prefix_lens_ptr + bs_offs, mask=bs_mask, other=0)
    extend_lens_all = tl.load(extend_seq_lens_ptr + bs_offs, mask=bs_mask, other=0)

    not_padding_all = (rpi_all != 0) & bs_mask

    if SWA_WINDOW > 0:
        clipped = prefix_lens_all - SWA_WINDOW
        win_start_all = tl.where(clipped > 0, clipped, 0)
        verify_lens_all = prefix_lens_all - win_start_all
    else:
        win_start_all = tl.zeros((BS_BLOCK,), dtype=prefix_lens_all.dtype)
        verify_lens_all = prefix_lens_all

    verify_lens_all = tl.where(not_padding_all, verify_lens_all, 0)
    verify_lens_all = tl.where(verify_lens_all > 0, verify_lens_all, 0)

    write_lens_all = tl.where(not_padding_all, extend_lens_all, 0)
    write_lens_all = tl.where(write_lens_all > 0, write_lens_all, 0)

    write_req_active_all = tl.where(write_lens_all > 0, 1, 0).to(prefix_lens_all.dtype)

    verify_starts_all = tl.cumsum(verify_lens_all, axis=0) - verify_lens_all
    cursor_all = tl.cumsum(extend_lens_all, axis=0) - extend_lens_all
    write_starts_all = tl.cumsum(write_lens_all, axis=0) - write_lens_all
    write_req_starts_all = (
        tl.cumsum(write_req_active_all, axis=0) - write_req_active_all
    )

    if r == 0:
        total_verify = tl.sum(verify_lens_all, axis=0)
        total_write_reqs = tl.sum(write_req_active_all, axis=0)
        tl.store(verify_num_valid_ptr, total_verify.to(tl.int32))
        tl.store(write_req_num_valid_ptr, total_write_reqs.to(tl.int32))

    if r >= bs:
        return

    row_select = bs_offs == r
    zero_tile = tl.zeros((BS_BLOCK,), dtype=prefix_lens_all.dtype)
    my_rpi = tl.sum(tl.where(row_select, rpi_all, zero_tile), axis=0)
    my_prefix_len = tl.sum(tl.where(row_select, prefix_lens_all, zero_tile), axis=0)
    my_verify_start = tl.sum(tl.where(row_select, verify_starts_all, zero_tile), axis=0)
    my_verify_len = tl.sum(tl.where(row_select, verify_lens_all, zero_tile), axis=0)
    my_win_start = tl.sum(tl.where(row_select, win_start_all, zero_tile), axis=0)
    my_cursor = tl.sum(tl.where(row_select, cursor_all, zero_tile), axis=0)
    my_write_start = tl.sum(tl.where(row_select, write_starts_all, zero_tile), axis=0)
    my_write_len = tl.sum(tl.where(row_select, write_lens_all, zero_tile), axis=0)
    my_write_req_idx = tl.sum(
        tl.where(row_select, write_req_starts_all, zero_tile), axis=0
    )
    my_not_padding = my_rpi != 0

    if my_not_padding & (my_verify_len > 0):
        verify_iters = (my_verify_len + INNER_BLOCK - 1) // INNER_BLOCK
        for it in range(verify_iters):
            j_offs = it * INNER_BLOCK + tl.arange(0, INNER_BLOCK)
            j_mask = j_offs < my_verify_len
            pos = my_win_start + j_offs
            slot_full = tl.load(
                req_to_token_ptr + my_rpi * req_to_token_stride + pos,
                mask=j_mask,
                other=0,
            )
            prev_pos = pos - 1
            prev_pos_valid = (pos > 0) & j_mask
            prev_slot_full = tl.load(
                req_to_token_ptr + my_rpi * req_to_token_stride + prev_pos,
                mask=prev_pos_valid,
                other=0,
            )
            if HAS_SWA_LUT:
                slot_sentinel = slot_full < 0
                slot_safe = tl.where(slot_sentinel, 0, slot_full)
                slot_safe = tl.where(
                    slot_safe >= swa_lut_len, swa_lut_len - 1, slot_safe
                )
                slot_xlat = tl.load(
                    full_to_swa_lut_ptr + slot_safe,
                    mask=j_mask & (~slot_sentinel),
                    other=0,
                )
                slot = tl.where(slot_sentinel, slot_full, slot_xlat)

                prev_sentinel = prev_slot_full < 0
                prev_safe = tl.where(prev_sentinel, 0, prev_slot_full)
                prev_safe = tl.where(
                    prev_safe >= swa_lut_len, swa_lut_len - 1, prev_safe
                )
                prev_xlat = tl.load(
                    full_to_swa_lut_ptr + prev_safe,
                    mask=prev_pos_valid & (~prev_sentinel),
                    other=0,
                )
                prev_slot_translated = tl.where(
                    prev_sentinel, prev_slot_full, prev_xlat
                )
            else:
                slot = slot_full
                prev_slot_translated = prev_slot_full

            chain_head_tile = tl.full((INNER_BLOCK,), -1, prefix_lens_all.dtype)
            prev_slot = tl.where(prev_pos_valid, prev_slot_translated, chain_head_tile)

            out_offs = my_verify_start + j_offs
            cap_mask = out_offs < VERIFY_CAPACITY
            write_mask = j_mask & cap_mask
            tl.store(
                verify_slot_indices_ptr + out_offs,
                slot.to(tl.int64),
                mask=write_mask,
            )
            tl.store(
                verify_positions_ptr + out_offs,
                pos.to(tl.int64),
                mask=write_mask,
            )
            tl.store(
                verify_prev_slot_indices_ptr + out_offs,
                prev_slot.to(tl.int64),
                mask=write_mask,
            )

    if my_not_padding & (my_write_len > 0):
        seed_full = tl.load(
            req_to_token_ptr + my_rpi * req_to_token_stride + (my_prefix_len - 1),
            mask=my_prefix_len > 0,
            other=0,
        )
        if HAS_SWA_LUT:
            seed_sentinel = seed_full < 0
            seed_safe = tl.where(seed_sentinel, 0, seed_full)
            seed_safe = tl.where(seed_safe >= swa_lut_len, swa_lut_len - 1, seed_safe)
            seed_xlat = tl.load(
                full_to_swa_lut_ptr + seed_safe,
                mask=(my_prefix_len > 0) & (~seed_sentinel),
                other=0,
            )
            seed_translated = tl.where(seed_sentinel, seed_full, seed_xlat)
        else:
            seed_translated = seed_full
        seed_slot = tl.where(my_prefix_len > 0, seed_translated, -1)

        req_cap_mask = my_write_req_idx < WRITE_REQ_CAPACITY
        tl.store(
            write_req_seed_slot_indices_ptr + my_write_req_idx,
            seed_slot.to(tl.int64),
            mask=req_cap_mask,
        )
        tl.store(
            write_req_entry_starts_ptr + my_write_req_idx,
            my_write_start.to(tl.int64),
            mask=req_cap_mask,
        )
        tl.store(
            write_req_entry_counts_ptr + my_write_req_idx,
            my_write_len.to(tl.int64),
            mask=req_cap_mask,
        )

        write_iters = (my_write_len + INNER_BLOCK - 1) // INNER_BLOCK
        for it in range(write_iters):
            k_offs = it * INNER_BLOCK + tl.arange(0, INNER_BLOCK)
            k_mask = k_offs < my_write_len
            src_idx = my_cursor + k_offs
            token_id = tl.load(input_ids_ptr + src_idx, mask=k_mask, other=0)
            slot_full = tl.load(out_cache_loc_ptr + src_idx, mask=k_mask, other=0)
            if HAS_POSITIONS:
                pos = tl.load(positions_ptr + src_idx, mask=k_mask, other=0)
            else:
                pos = my_prefix_len + k_offs

            if HAS_SWA_LUT:
                slot_sentinel = slot_full < 0
                slot_safe = tl.where(slot_sentinel, 0, slot_full)
                slot_safe = tl.where(
                    slot_safe >= swa_lut_len, swa_lut_len - 1, slot_safe
                )
                slot_xlat = tl.load(
                    full_to_swa_lut_ptr + slot_safe,
                    mask=k_mask & (~slot_sentinel),
                    other=0,
                )
                slot = tl.where(slot_sentinel, slot_full, slot_xlat)
            else:
                slot = slot_full

            out_offs = my_write_start + k_offs
            cap_mask = out_offs < WRITE_CAPACITY
            write_mask = k_mask & cap_mask
            tl.store(
                write_slot_indices_ptr + out_offs,
                slot.to(tl.int64),
                mask=write_mask,
            )
            tl.store(
                write_token_ids_ptr + out_offs,
                token_id.to(tl.int64),
                mask=write_mask,
            )
            tl.store(
                write_positions_ptr + out_offs,
                pos.to(tl.int64),
                mask=write_mask,
            )
            skip_tile = tl.full((INNER_BLOCK,), SKIP_SENTINEL, tl.int64)
            tl.store(
                expected_write_token_ids_ptr + out_offs,
                skip_tile,
                mask=write_mask,
            )
            tl.store(
                expected_write_positions_ptr + out_offs,
                skip_tile,
                mask=write_mask,
            )


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def plan_batch_from_forward_batch(
    *,
    plan_out: BatchPlanGpu,
    req_pool_indices: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    pseudo_mode: int,
) -> None:
    """Per-forward plan kernel: turn ForwardBatch primitives into the flat plan tensors
    that ``canary_step`` consumes.

    BatchPlan layout — the output ``plan_out`` carries three parallel tile sets, each sized to
    a cuda-graph-captured fixed maximum (capacity); the active prefix is reported via the
    counters ``plan_out.num_verify`` / ``num_write`` / ``num_write_reqs``:

    - **Per-verify-entry** (length ``num_verify``): ``(slot_idx, position, prev_slot_idx)``
      for each slot to check. ``prev_slot_idx == -1`` flags the chain-seed entry (position 0
      of a req, or the head of an SWA window — anchor on
      :data:`sglang.jit_kernel.kv_cache_canary.CANARY_CHAIN_ANCHOR` instead of a predecessor).
      Padding tail entries (``i >= num_verify``) are unspecified; ``canary_step`` early-exits
      via its own ``verify_num_valid``.
    - **Per-write-entry** (length ``num_write``): ``(slot_idx, token_id, position)`` for each
      slot to fingerprint this step, flattened across reqs in the same order the reqs appear in
      ``req_pool_indices``.
    - **Per-write-req** (length ``num_write_reqs``): ``(seed_slot_idx, entry_start,
      entry_count, req_pool_idx)`` per req that has at least one write entry. ``seed_slot_idx
      == -1`` flags ``K_req_old == 0`` (chain anchors on
      :data:`sglang.jit_kernel.kv_cache_canary.CANARY_CHAIN_ANCHOR`). ``entry_start`` is the
      exclusive prefix-sum offset into the per-write-entry arrays.

    A call performs five fused operations against ``plan_out`` in a single kernel invocation:

    1. **Gather verify slots** — for each req ``r`` and each position ``p`` in its verify range,
       write ``slot_idx = req_to_token[req_pool_indices[r], p]``, optionally translated via
       ``full_to_swa_index_mapping`` for the SWA group.
    2. **Gather verify predecessors** — fill ``verify_prev_slot_indices`` from
       ``req_to_token[..., p-1]``, with ``-1`` sentinel at chain heads.
    3. **Clip verify range** — for the SWA group (``swa_window_size > 0``), restrict
       ``[0, prefix_lens[r])`` down to ``[max(0, prefix_lens[r] - swa_window_size),
       prefix_lens[r])``. FULL group (``swa_window_size == 0``) covers the full prefix.
    4. **Gather write slots + chain seed** — copy ``out_cache_loc`` / ``input_ids`` /
       ``positions`` for every token this step into the per-write-entry tile set; look up each
       req's chain-seed slot from ``req_to_token[..., prefix_lens[r] - 1]`` (or ``-1`` if
       ``prefix_lens[r] == 0``).
    5. **Per-req scan** — exclusive prefix-sum over per-req write counts populates
       ``write_req_entry_starts``; per-req metadata is copied into ``write_req_*``.

    Returns ``None``; ``plan_out`` is mutated in place. The active counters are written as
    device int32 scalars and consumed by the downstream ``canary_step`` launch (as its
    ``verify_num_valid`` / ``write_req_num_valid``).

    Padding rows whose ``req_pool_indices[r] == 0`` (cuda-graph padding sentinel) contribute
    zero verify entries and zero write entries — captured-time padding costs nothing at
    replay.

    Args:
        plan_out:                    Pre-allocated :class:`BatchPlanGpu` — every per-entry
                                     tensor is sized to a cuda-graph-captured fixed maximum;
                                     this call fills the active prefix in-place and writes the
                                     active counters. Caller owns allocation and lifecycle.
        req_pool_indices:            ``int32 [bs]`` — per-row ``ReqToTokenPool`` row index.
                                     ``0`` is the padding sentinel (the req contributes zero
                                     verify and zero write entries).
        input_ids:                   ``int32 [num_tokens_padded]`` — flat token ids for this
                                     step's write set, concatenated across reqs in the same
                                     order as ``req_pool_indices``. Tail entries beyond
                                     ``sum(extend_seq_lens)`` are cuda-graph padding and are
                                     never read.
        positions:                   ``int32 [num_tokens_padded]`` — sequence position of each
                                     ``input_ids`` entry, same indexing.
        out_cache_loc:               ``int32 [num_tokens_padded]`` — **full-pool** slot index
                                     of each ``input_ids`` entry. Always in full-pool index
                                     space; SWA translation is performed by this kernel via
                                     ``full_to_swa_index_mapping``.
        prefix_lens:                 ``int32 [bs]`` — per-req prefix length already written
                                     *before* this step. Normalized by the caller: extend modes
                                     pass ``extend_prefix_lens``; decode passes ``seq_lens - 1``.
        extend_seq_lens:             ``int32 [bs]`` — per-req number of tokens being written
                                     this step. ``1`` per req for pure decode.
        req_to_token:                ``int32 [max_reqs, max_seq_len]`` — the live
                                     ``ReqToTokenPool`` table; full-pool slot indices.
        swa_window_size:             ``0`` for the FULL canary group (verify covers
                                     ``[0, prefix_lens[r])``). Positive SWA window length for
                                     the SWA group (verify clipped to
                                     ``[max(0, prefix_lens[r] - swa_window_size),
                                     prefix_lens[r])``).
        full_to_swa_index_mapping:   ``int32 [full_pool_size + 1]`` or ``None``. When not None,
                                     every slot index that lands in ``plan_out`` is translated
                                     through this LUT first; the trailing sentinel ``-1`` row
                                     maps out-of-window full-pool slots to a skip-this-entry
                                     marker. Required (non-None) iff ``swa_window_size > 0``.
        pseudo_mode:                 Oracle selector for ``plan_out.expected_write_*``. ``0`` =
                                     real-mode (no oracle; ``expected_write_*`` is filled with
                                     ``-1`` skip-sentinels and ``canary_step`` pays no per-entry
                                     cost). ``> 0`` selects one of the pseudo-mode oracles
                                     defined by ``CanaryPseudoMode``; the kernel computes the
                                     expected ``(token, position)`` inline from
                                     ``(req_pool_idx, position, seed)``.

    Calling contract:

    - Pure side-effect: returns ``None``; only ``plan_out``'s tensors are mutated. Inputs are
      read-only.
    - **No host work, no D2H**: every read and write happens on the device. Safe to invoke
      inside cuda-graph capture; replay reuses the captured tensor addresses, so callers must
      refill ``input_ids`` / ``positions`` / ``out_cache_loc`` / ``req_pool_indices`` /
      ``prefix_lens`` / ``extend_seq_lens`` in-place before ``graph.replay()`` (``req_to_token``
      is usually already in place via the live ``ReqToTokenPool``).
    - **Single launch covers all three tile sets** — verify gather, write gather, per-req scan,
      and SWA LUT translation are fused so that a single kernel invocation populates
      ``plan_out`` end-to-end. There is no separate cumsum or LUT pass.
    - **Padding cost is zero at replay**: capture-time padded rows whose
      ``req_pool_indices == 0`` take an in-kernel early-exit; the per-entry tail beyond
      ``num_*`` is left at whatever sentinel value the kernel writes (typically ``-1``) and is
      masked off by ``canary_step``'s ``*_num_valid`` reads.

    The full per-field semantics (verify range derivation, chain-seed selection, SWA clip +
    LUT ordering, cumsum boundary conditions, pseudo-mode oracle math) are pinned by the
    executable reference
    :func:`sglang.jit_kernel.kv_cache_canary_plan_ref.plan_batch_from_forward_batch_ref`,
    which the Triton path must match byte-for-byte.
    """
    bs = int(req_pool_indices.shape[0])
    if bs == 0:
        plan_out.verify_num_valid.zero_()
        plan_out.write_req_num_valid.zero_()
        return

    if swa_window_size > 0 and full_to_swa_index_mapping is None:
        raise ValueError(
            "kv-canary plan: swa_window_size > 0 requires full_to_swa_index_mapping"
        )
    if pseudo_mode != 0:
        raise NotImplementedError(
            "kv-canary plan kernel: pseudo_mode != 0 is not implemented yet"
        )

    device = plan_out.verify_slot_indices.device
    req_pool_indices_i64 = req_pool_indices.to(device=device, dtype=torch.int64)
    input_ids_i64 = input_ids.to(device=device, dtype=torch.int64)
    out_cache_loc_i64 = out_cache_loc.to(device=device, dtype=torch.int64)
    prefix_lens_i64 = prefix_lens.to(device=device, dtype=torch.int64)
    extend_seq_lens_i64 = extend_seq_lens.to(device=device, dtype=torch.int64)
    req_to_token_i64 = req_to_token.to(device=device, dtype=torch.int64)
    has_positions = positions is not None and positions.numel() > 0
    if has_positions:
        positions_i64 = positions.to(device=device, dtype=torch.int64)
    else:
        positions_i64 = torch.zeros(1, dtype=torch.int64, device=device)

    if full_to_swa_index_mapping is not None:
        lut_i64 = full_to_swa_index_mapping.to(device=device, dtype=torch.int64)
        lut_len = int(lut_i64.shape[0])
    else:
        lut_i64 = torch.zeros(1, dtype=torch.int64, device=device)
        lut_len = 0

    verify_capacity = int(plan_out.verify_slot_indices.shape[0])
    write_capacity = int(plan_out.write_slot_indices.shape[0])
    write_req_capacity = int(plan_out.write_req_seed_slot_indices.shape[0])

    req_to_token_stride = int(req_to_token_i64.stride(0))

    bs_block = _next_power_of_two(max(bs, 1))
    inner_block = 64

    # Match the host ref's tail-reset semantics: write-tile tail is zeroed,
    # expected-write tile tail is set to the skip sentinel. The kernel
    # populates the active prefix on top of these defaults.
    plan_out.write_slot_indices.zero_()
    plan_out.write_token_ids.zero_()
    plan_out.write_positions.zero_()
    plan_out.expected_write_token_ids.fill_(int(CANARY_EXPECTED_SKIP_SENTINEL))
    plan_out.expected_write_positions.fill_(int(CANARY_EXPECTED_SKIP_SENTINEL))

    grid = (bs,)
    _plan_kernel[grid](
        req_pool_indices_i64,
        input_ids_i64,
        positions_i64,
        out_cache_loc_i64,
        prefix_lens_i64,
        extend_seq_lens_i64,
        req_to_token_i64,
        lut_i64,
        plan_out.verify_slot_indices,
        plan_out.verify_positions,
        plan_out.verify_prev_slot_indices,
        plan_out.write_slot_indices,
        plan_out.write_token_ids,
        plan_out.write_positions,
        plan_out.expected_write_token_ids,
        plan_out.expected_write_positions,
        plan_out.write_req_seed_slot_indices,
        plan_out.write_req_entry_starts,
        plan_out.write_req_entry_counts,
        plan_out.verify_num_valid,
        plan_out.write_req_num_valid,
        bs,
        req_to_token_stride,
        lut_len,
        BS_BLOCK=bs_block,
        INNER_BLOCK=inner_block,
        VERIFY_CAPACITY=verify_capacity,
        WRITE_CAPACITY=write_capacity,
        WRITE_REQ_CAPACITY=write_req_capacity,
        SWA_WINDOW=int(swa_window_size),
        HAS_SWA_LUT=full_to_swa_index_mapping is not None,
        HAS_POSITIONS=has_positions,
        SKIP_SENTINEL=int(CANARY_EXPECTED_SKIP_SENTINEL),
    )


def plan_batch_from_forward_batch_triton(
    *,
    forward_batch: "ForwardBatch",
    config: CanaryConfig,
    plan_out: BatchPlanGpu,
    swa_index_lut: Optional[torch.Tensor] = None,
) -> bool:
    """High-level adapter that mirrors the ref impl's call signature.

    Extracts the primitive tensors from ``forward_batch`` on the host
    (forward_mode dispatch, ``prefix_lens`` selection) and dispatches the
    Triton wrapper. Returns ``True`` when the kernel ran, ``False`` when
    the batch was rejected (e.g. unsupported forward mode, missing
    extend lens) — in which case ``plan_out``'s active counters are
    zeroed and the recorded launch becomes a no-op.

    This is the wiring shim used by ``api.run_head`` / ``api.prepare_replay``
    and by the differential test driver. The contract matches
    :func:`sglang.jit_kernel.kv_cache_canary_plan_ref.plan_batch_from_forward_batch`
    byte-for-byte under :func:`fill_batch_plan_gpu_from_plan` projection
    (the differential test asserts this).
    """
    if forward_batch.out_cache_loc is None or forward_batch.out_cache_loc.numel() == 0:
        plan_out.verify_num_valid.zero_()
        plan_out.write_req_num_valid.zero_()
        return False

    forward_mode = forward_batch.forward_mode
    if forward_mode is None:
        plan_out.verify_num_valid.zero_()
        plan_out.write_req_num_valid.zero_()
        return False

    is_extend = forward_mode.is_extend() or forward_mode.is_mixed()
    if is_extend:
        if (
            forward_batch.extend_seq_lens is None
            or forward_batch.extend_prefix_lens is None
        ):
            plan_out.verify_num_valid.zero_()
            plan_out.write_req_num_valid.zero_()
            return False
        extend_seq_lens = forward_batch.extend_seq_lens
        prefix_lens = forward_batch.extend_prefix_lens
    elif forward_mode.is_decode() or forward_mode.is_target_verify():
        bs_local = int(forward_batch.req_pool_indices.shape[0])
        extend_seq_lens = torch.ones(
            bs_local,
            dtype=torch.int64,
            device=forward_batch.req_pool_indices.device,
        )
        prefix_lens = forward_batch.seq_lens - 1
    else:
        plan_out.verify_num_valid.zero_()
        plan_out.write_req_num_valid.zero_()
        return False

    req_to_token_pool = forward_batch.req_to_token_pool
    if req_to_token_pool is None:
        plan_out.verify_num_valid.zero_()
        plan_out.write_req_num_valid.zero_()
        return False

    swa_window = int(config.swa_window_size) if config.swa_window_size else 0
    device = plan_out.verify_slot_indices.device
    positions_tensor = (
        forward_batch.positions
        if forward_batch.positions is not None
        else torch.empty(0, dtype=torch.int64, device=device)
    )

    plan_batch_from_forward_batch(
        plan_out=plan_out,
        req_pool_indices=forward_batch.req_pool_indices,
        input_ids=forward_batch.input_ids,
        positions=positions_tensor,
        out_cache_loc=forward_batch.out_cache_loc,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token_pool.req_to_token,
        swa_window_size=swa_window,
        full_to_swa_index_mapping=swa_index_lut,
        pseudo_mode=0,
    )
    return True
