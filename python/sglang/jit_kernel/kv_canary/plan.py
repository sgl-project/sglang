from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kv_canary.verify import (
    VerifyPlan,
    _assert_contiguous,
)
from sglang.jit_kernel.kv_canary.write import WritePlan

# Upper bound on bs for _plan_offsets_kernel's block-level cumsum. Reqs larger than this exceed Triton's
# single-program tl.cumsum reach. Increase if real workloads ever push past it; the cap is intentionally
# generous so the wrapper never silently truncates.
_PLAN_BS_BLOCK_SIZE: int = 4096

# Inner-tile width for _plan_entries_kernel. Each (req, j-tile) program owns this many entries along the
# j-axis of the (bs, max_verify_per_req) logical grid.
_PLAN_VERIFY_INNER_BLOCK: int = 64

# Inner-tile width for _plan_extras_kernel. Each program copies this many extras into the verify tail.
_PLAN_EXTRAS_INNER_BLOCK: int = 64


def _resolve_swa_lut(
    lut: Optional[torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, int, bool]:
    """Return the (tensor, length, has_lut) triple to launch the plan kernel with.

    Triton requires a valid tensor pointer at every kernel-arg slot even when ``HAS_SWA_LUT`` is False, so
    when the caller passes ``None`` we substitute a one-element sentinel tensor and set ``lut_len=0``;
    the kernel's constexpr branch guarantees no dereference happens. Dtype matches the production LUT
    (int64) so Triton ``tl.load`` element typing stays consistent.
    """
    if lut is not None:
        return lut, int(lut.shape[0]), True
    return torch.zeros(1, dtype=torch.int64, device=device), 0, False


def canary_plan_step(
    *,
    verify_plan_out: VerifyPlan,
    write_plan_out: WritePlan,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    extra_verify_slot_indices: torch.Tensor,
    extra_verify_positions: torch.Tensor,
    extra_verify_prev_slot_indices: torch.Tensor,
    extra_verify_num_valid: torch.Tensor,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    verify_capacity: int,
) -> None:
    """Fill verify_plan_out + write_plan_out from ForwardBatch primitives + optional pre-walked flat verify
    entries. Single Triton launch.

    For each req r with fb_req_pool_indices[r] != 0 (0 = padding sentinel):

    - **Verify entries**: one per pos in [window_start, fb_prefix_lens[r]), where window_start = max(0,
      fb_prefix_lens[r] - swa_window_size) if SWA else 0. slot_idx = req_to_token[fb_req_pool_indices[r], pos]
      (SWA-translated via full_to_swa_index_mapping if non-None); prev_slot_idx =
      req_to_token[fb_req_pool_indices[r], pos-1] for pos > 0, else -1. (SWA windows do NOT reset the chain —
      the writer chains across the entire prefix; sweep verify within an SWA window dereferences the real
      predecessor for chain-link reconstruction.)
    - **Write metadata** (when fb_extend_seq_lens[r] > 0): contribute fb_extend_seq_lens[r] to the per-req
      write count (for write_offsets cumsum). Per-req chain seed = req_to_token[fb_req_pool_indices[r],
      fb_prefix_lens[r]-1] (SWA-translated), or -1 if fb_prefix_lens[r] == 0. Per-token write data
      (fb_input_ids / fb_positions / fb_out_cache_loc) is NOT materialized here — canary_write_step reads it
      directly from ForwardBatch via write_offsets.

    Extra flat verify entries (extra_verify_*[: extra_verify_num_valid[0]]) are appended to verify_plan_out
    **after** the per-req-derived entries. Used by radix-cache-orphan sweep; caller is responsible for
    SWA-translating these entries before passing in (plan kernel does NOT translate the extras).

    Sweep callers pass fb_extend_seq_lens = all-zero → write_plan_out is filled with write_num_valid_reqs = 0;
    downstream skips canary_write_step.

    Args:
        verify_plan_out: Pre-allocated VerifyPlan; filled in-place.
        write_plan_out: Pre-allocated WritePlan; filled in-place. write_num_valid_reqs = 0 for sweep callers.
        fb_req_pool_indices: ForwardBatch.req_pool_indices; per-row ReqToTokenPool row index, shape [bs],
            int64. 0 is the padding sentinel.
        fb_prefix_lens: Per-req prefix length already written before this step, shape [bs], int64. Caller
            normalizes: extend → ForwardBatch.extend_prefix_lens, decode → ForwardBatch.seq_lens - 1, sweep
            over running → seq_lens.
        fb_extend_seq_lens: ForwardBatch.extend_seq_lens; per-req tokens being written this step, shape [bs],
            int64. 1 for pure decode; 0 for sweep.
        req_to_token: ReqToTokenPool.req_to_token; full-pool slot index table, shape [max_reqs, max_seq_len],
            int32.
        extra_verify_slot_indices: Pre-walked extra verify slots, shape [extra_verify_capacity], int64.
            Caller-translated to the target index space.
        extra_verify_positions: Same shape, int64. Expected position per extra entry.
        extra_verify_prev_slot_indices: Same shape, int64. -1 for chain-seed extras.
        extra_verify_num_valid: Active extra entry count, shape [1], int32. 0 for per-forward and running-sweep
            callers.
        swa_window_size: 0 for the FULL canary group; positive window length for the SWA group.
        full_to_swa_index_mapping: SWA LUT, shape [full_pool_size + 1], int64, or None. Required (non-None) iff
            swa_window_size > 0. Used to translate verify slot indices and chain-seed slot indices at plan time.
            Loaded element-typed via Triton ``tl.load``; intermediate translated slot values are int64 inside the
            kernel and stored in the int64 plan schema.

    Implementation:
        - Three sub-kernels with action-named identifiers, launched in sequence:
          1. ``_plan_offsets_kernel`` (1-D grid ``(1,)``, single program over all ``bs`` reqs):
             reads fb_req_pool_indices[r], fb_prefix_lens[r], fb_extend_seq_lens[r] for each r; computes
             verify_count = (prefix_lens - window_start) and write_count = extend_seq_lens (both 0 if rp == 0
             padding); gathers seed_slot_full = req_to_token[rp, prefix_lens - 1] (or -1 if prefix_lens == 0),
             SWA-translates seed_slot via full_to_swa_index_mapping[seed_slot_full] if non-None; runs
             block-level cumsum (``tl.cumsum``) to produce verify_offsets[bs+1] and
             write_plan_out.write_offsets[bs+1] in-place; scatters write_seed slots; writes scalar totals
             ``verify_plan_out.verify_num_valid`` and ``write_plan_out.write_num_valid_reqs``.
          2. ``_plan_entries_kernel`` (2-D grid ``(bs, max_j_tiles)``, masked by per-req verify_count): for
             each (r, j) with j < verify_count[r], gather slot = req_to_token[fb_req_pool_indices[r],
             window_start[r] + j] (SWA-translated), prev_slot = req_to_token[..., window_start[r] + j - 1]
             when (window_start[r] + j) > 0 (also translated) else -1, position = window_start[r] + j;
             scatter (slot, position, prev_slot) into verify_plan_out at flat index verify_offsets[r] + j.
          3. ``_plan_extras_kernel`` (1-D grid ``(k_tiles,)``, only launched when extra_verify_num_valid > 0):
             copy extra_verify_*[: num_valid] into verify_plan_out.verify_* starting at flat index
             verify_offsets[bs]. Extras are caller-pre-translated, no LUT pass.
        - All output tensors are addressed at addresses baked into the cuda-graph capture; the 2nd and 3rd
          kernels are conditionally launched based on cached host-side scalars from the 1st (graph-safe
          under stream-capture conditionals).

    Calling contract:
        - Pure side-effect; no host work, no D2H.
        - Safe in cuda-graph capture; caller refills all input tensors in-place before replay.
        - Single kernel launch fills both plans end-to-end.
        - Padding rows contribute zero entries.

    Pinned by Python reference
    :func:`sglang.jit_kernel.kv_canary.plan_ref.canary_plan_step_torch_reference`; Triton must match
    byte-for-byte.
    """
    bs = int(fb_req_pool_indices.shape[0])
    if bs > _PLAN_BS_BLOCK_SIZE:
        raise ValueError(
            f"kv-canary: canary_plan_step supports at most bs={_PLAN_BS_BLOCK_SIZE} reqs per launch, "
            f"got bs={bs}. Bump _PLAN_BS_BLOCK_SIZE if real workloads need this."
        )
    if swa_window_size > 0 and full_to_swa_index_mapping is None:
        raise ValueError(
            "kv-canary: canary_plan_step requires full_to_swa_index_mapping when swa_window_size > 0"
        )

    _assert_contiguous(
        verify_plan_out.verify_slot_indices, "verify_plan_out.verify_slot_indices"
    )
    _assert_contiguous(
        verify_plan_out.verify_positions, "verify_plan_out.verify_positions"
    )
    _assert_contiguous(
        verify_plan_out.verify_prev_slot_indices,
        "verify_plan_out.verify_prev_slot_indices",
    )
    _assert_contiguous(
        verify_plan_out.verify_num_valid, "verify_plan_out.verify_num_valid"
    )
    _assert_contiguous(verify_plan_out.enable, "verify_plan_out.enable")
    _assert_contiguous(write_plan_out.write_offsets, "write_plan_out.write_offsets")
    _assert_contiguous(
        write_plan_out.write_seed_slot_indices, "write_plan_out.write_seed_slot_indices"
    )
    _assert_contiguous(
        write_plan_out.write_num_valid_reqs, "write_plan_out.write_num_valid_reqs"
    )
    _assert_contiguous(fb_req_pool_indices, "fb_req_pool_indices")
    _assert_contiguous(fb_prefix_lens, "fb_prefix_lens")
    _assert_contiguous(fb_extend_seq_lens, "fb_extend_seq_lens")
    _assert_contiguous(req_to_token, "req_to_token")
    _assert_contiguous(extra_verify_slot_indices, "extra_verify_slot_indices")
    _assert_contiguous(extra_verify_positions, "extra_verify_positions")
    _assert_contiguous(extra_verify_prev_slot_indices, "extra_verify_prev_slot_indices")
    _assert_contiguous(extra_verify_num_valid, "extra_verify_num_valid")
    if full_to_swa_index_mapping is not None:
        _assert_contiguous(full_to_swa_index_mapping, "full_to_swa_index_mapping")

    device = verify_plan_out.verify_slot_indices.device
    verify_offsets_scratch = torch.zeros(
        _PLAN_BS_BLOCK_SIZE + 1, dtype=torch.int64, device=device
    )

    plan_verify_capacity = int(verify_plan_out.verify_slot_indices.shape[0])
    if verify_capacity != plan_verify_capacity:
        raise ValueError(
            f"kv-canary: canary_plan_step verify_capacity={verify_capacity} does not match "
            f"verify_plan_out.verify_slot_indices.shape[0]={plan_verify_capacity}"
        )
    write_req_capacity_plus_one = int(write_plan_out.write_offsets.shape[0])
    write_req_capacity = int(write_plan_out.write_seed_slot_indices.shape[0])
    extras_capacity = int(extra_verify_slot_indices.shape[0])

    lut_tensor, lut_len, has_swa_lut = _resolve_swa_lut(
        full_to_swa_index_mapping, device
    )

    req_to_token_stride0 = int(req_to_token.stride(0))

    # Match the ref's tail-reset semantics: write_offsets positions past index bs are zeroed so a smaller
    # batch never leaks stale prefix-sum entries from a larger previous call. In-place .zero_() is
    # cuda-graph-safe (no allocation) and avoids one Triton launch.
    write_plan_out.write_offsets.zero_()

    # Offsets kernel: per-req count + seed gather + block-level cumsum, single program; the num_valid
    # scalars are written by the same program (it has the totals in registers already).
    _plan_offsets_kernel[(1,)](
        fb_req_pool_indices,
        fb_prefix_lens,
        fb_extend_seq_lens,
        req_to_token,
        lut_tensor,
        extra_verify_num_valid,
        verify_offsets_scratch,
        write_plan_out.write_offsets,
        write_plan_out.write_seed_slot_indices,
        verify_plan_out.verify_num_valid,
        verify_plan_out.enable,
        write_plan_out.write_num_valid_reqs,
        bs,
        req_to_token_stride0,
        lut_len,
        BS_BLOCK=_PLAN_BS_BLOCK_SIZE,
        SWA_WINDOW=int(swa_window_size),
        HAS_SWA_LUT=has_swa_lut,
        WRITE_OFFSETS_LEN=write_req_capacity_plus_one,
        WRITE_REQ_CAPACITY=write_req_capacity,
        VERIFY_CAPACITY=verify_capacity,
    )

    # Entries kernel: per-(req, j-tile) verify entry materialization. The j-axis upper bound is
    # verify_capacity (each req cannot contribute more than verify_capacity entries); we mask per-req actual
    # count read back from verify_offsets_scratch inside the kernel.
    if bs > 0 and verify_capacity > 0:
        max_j_tiles = (
            verify_capacity + _PLAN_VERIFY_INNER_BLOCK - 1
        ) // _PLAN_VERIFY_INNER_BLOCK
        grid_entries = (bs, max_j_tiles)
        _plan_entries_kernel[grid_entries](
            fb_req_pool_indices,
            fb_prefix_lens,
            req_to_token,
            lut_tensor,
            verify_offsets_scratch,
            verify_plan_out.verify_slot_indices,
            verify_plan_out.verify_positions,
            verify_plan_out.verify_prev_slot_indices,
            req_to_token_stride0,
            lut_len,
            verify_capacity,
            INNER_BLOCK=_PLAN_VERIFY_INNER_BLOCK,
            SWA_WINDOW=int(swa_window_size),
            HAS_SWA_LUT=has_swa_lut,
        )

    # Extras kernel: append extras into the verify tail. The base index lives in verify_offsets_scratch[bs].
    if extras_capacity > 0:
        max_k_tiles = (
            extras_capacity + _PLAN_EXTRAS_INNER_BLOCK - 1
        ) // _PLAN_EXTRAS_INNER_BLOCK
        grid_extras = (max_k_tiles,)
        _plan_extras_kernel[grid_extras](
            extra_verify_slot_indices,
            extra_verify_positions,
            extra_verify_prev_slot_indices,
            extra_verify_num_valid,
            verify_offsets_scratch,
            verify_plan_out.verify_slot_indices,
            verify_plan_out.verify_positions,
            verify_plan_out.verify_prev_slot_indices,
            bs,
            verify_capacity,
            INNER_BLOCK=_PLAN_EXTRAS_INNER_BLOCK,
        )


@triton.jit
def _plan_offsets_kernel(
    # Input pointers.
    req_pool_indices_ptr,
    prefix_lens_ptr,
    extend_seq_lens_ptr,
    req_to_token_ptr,
    full_to_swa_lut_ptr,
    extra_verify_num_valid_ptr,
    # Output pointers.
    out_verify_offsets_ptr,
    out_write_offsets_ptr,
    out_write_seed_slot_indices_ptr,
    out_verify_num_valid_ptr,
    out_verify_enable_ptr,
    out_write_num_valid_reqs_ptr,
    # Runtime sizes.
    bs,
    req_to_token_stride0,
    swa_lut_len,
    # Compile-time constants.
    BS_BLOCK: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    HAS_SWA_LUT: tl.constexpr,
    WRITE_OFFSETS_LEN: tl.constexpr,
    WRITE_REQ_CAPACITY: tl.constexpr,
    VERIFY_CAPACITY: tl.constexpr,
):
    """Offsets kernel: per-req counts, seeds, exclusive-prefix-sum offsets, scalar totals.

    Single program; BLOCK_BS-wide tiles cover the full bs (caller ensures bs <= BS_BLOCK). All cumsum is done
    via block-level ``tl.cumsum`` in one program — no cross-program sync needed.
    """
    bs_offs = tl.arange(0, BS_BLOCK)  # [BS_BLOCK]
    bs_mask = bs_offs < bs  # [BS_BLOCK] bool

    # Per-req inputs (int64 for canary-owned metadata; req_to_token keeps its pool dtype).
    rpi = tl.load(req_pool_indices_ptr + bs_offs, mask=bs_mask, other=0)  # [BS_BLOCK]
    prefix_lens = tl.load(
        prefix_lens_ptr + bs_offs, mask=bs_mask, other=0
    )  # [BS_BLOCK]
    extend_lens = tl.load(
        extend_seq_lens_ptr + bs_offs, mask=bs_mask, other=0
    )  # [BS_BLOCK]

    is_active = (rpi != 0) & bs_mask  # [BS_BLOCK] bool
    has_prefix = is_active & (prefix_lens > 0)  # [BS_BLOCK] bool

    window_starts = _compute_window_start(prefix_lens, SWA_WINDOW)  # [BS_BLOCK]

    verify_lens = prefix_lens - window_starts  # [BS_BLOCK]
    verify_lens = tl.where(verify_lens > 0, verify_lens, 0)
    verify_lens = tl.where(is_active, verify_lens, 0)

    write_lens = tl.where(extend_lens > 0, extend_lens, 0)  # [BS_BLOCK]
    write_lens = tl.where(is_active, write_lens, 0)

    has_write_contribution = has_prefix & (write_lens > 0)  # [BS_BLOCK] bool

    # Seed slot per req. prefix_lens == 0 means no prefix → -1 sentinel. Padding row → no write contribution
    # → -1 sentinel either way; we also mask write_lens onto seed below to match the ref's "no write → -1".
    safe_prefix_pos = tl.where(prefix_lens > 0, prefix_lens - 1, 0)  # [BS_BLOCK]
    stride_i64 = req_to_token_stride0  # scalar
    seed_full = tl.load(  # [BS_BLOCK]
        req_to_token_ptr + rpi.to(tl.int64) * stride_i64 + safe_prefix_pos.to(tl.int64),
        mask=has_prefix,
        other=0,
    )

    if HAS_SWA_LUT:
        seed_translated = _swa_translate_tile(  # [BS_BLOCK]
            seed_full,
            has_prefix,
            full_to_swa_lut_ptr,
            swa_lut_len,
        )
    else:
        seed_translated = seed_full

    # Reqs with no write contribution should expose seed = -1 (ref's _seed_slot is masked by write_lens > 0).
    minus_one = tl.full((BS_BLOCK,), -1, dtype=seed_translated.dtype)  # [BS_BLOCK]
    seed_slot = tl.where(
        has_write_contribution, seed_translated, minus_one
    )  # [BS_BLOCK]

    # Inclusive cumsum → exclusive offsets via subtraction.
    verify_inclusive = tl.cumsum(verify_lens, axis=0)  # [BS_BLOCK]
    write_inclusive = tl.cumsum(write_lens, axis=0)  # [BS_BLOCK]
    verify_exclusive = verify_inclusive - verify_lens  # [BS_BLOCK]
    write_exclusive = write_inclusive - write_lens  # [BS_BLOCK]

    # Scatter exclusive offsets into the [bs+1]-sized output tensor. Positions [0, bs) get the exclusive sum;
    # position bs gets the total (totals = verify_inclusive at index bs - 1 if bs > 0, else 0).
    out_offsets_mask = bs_mask  # [BS_BLOCK] bool
    tl.store(
        out_verify_offsets_ptr + bs_offs,
        verify_exclusive.to(tl.int64),
        mask=out_offsets_mask,
    )
    write_offsets_mask = bs_offs < WRITE_OFFSETS_LEN  # [BS_BLOCK] bool
    tl.store(
        out_write_offsets_ptr + bs_offs,
        write_exclusive.to(tl.int64),
        mask=write_offsets_mask & bs_mask,
    )

    # Scatter seed slots (capped to write_req_capacity).
    seed_mask = bs_mask & (bs_offs < WRITE_REQ_CAPACITY)  # [BS_BLOCK] bool
    tl.store(
        out_write_seed_slot_indices_ptr + bs_offs,
        seed_slot.to(tl.int64),
        mask=seed_mask,
    )

    # Totals: sum of all per-req lens. Same value as the last inclusive entry but tl.sum is robust to bs == 0.
    total_verify = tl.sum(verify_lens, axis=0)  # scalar
    total_write = tl.sum(write_lens, axis=0)  # scalar

    # Store the [bs] slot of verify_offsets and write_offsets (one element past the last per-req entry).
    # verify_offsets scratch has length BS_BLOCK + 1 so the bs slot is always in range.
    tl.store(out_verify_offsets_ptr + bs, total_verify.to(tl.int64))
    # write_offsets has length WRITE_OFFSETS_LEN = write_req_capacity + 1; only store if in range.
    write_tail_in_range = bs < WRITE_OFFSETS_LEN  # scalar bool
    tl.store(
        out_write_offsets_ptr + bs,
        total_write.to(tl.int64),
        mask=write_tail_in_range,
    )

    # Scalar writes: verify_num_valid is clamped to the verify_capacity tensor extent so the verify kernel
    # never indexes past the buffer; enable carries the overflow bit (0 when requested > capacity) so the
    # verify kernel skips the whole launch and the host can warn-log this step.
    extras_count = tl.load(extra_verify_num_valid_ptr)  # scalar
    extras_count = tl.where(extras_count > 0, extras_count, 0)
    requested = total_verify + extras_count  # scalar
    overflow = requested > VERIFY_CAPACITY  # scalar bool
    enable = tl.where(overflow, 0, 1)  # scalar
    clamped = tl.where(overflow, VERIFY_CAPACITY, requested)  # scalar
    tl.store(out_verify_num_valid_ptr, clamped.to(tl.int32))
    tl.store(out_verify_enable_ptr, tl.full((), enable, tl.int32))
    tl.store(out_write_num_valid_reqs_ptr, tl.full((), bs, tl.int32))


@triton.jit
def _plan_entries_kernel(
    # Input pointers.
    req_pool_indices_ptr,
    prefix_lens_ptr,
    req_to_token_ptr,
    full_to_swa_lut_ptr,
    verify_offsets_ptr,
    # Output pointers.
    out_verify_slot_indices_ptr,
    out_verify_positions_ptr,
    out_verify_prev_slot_indices_ptr,
    # Runtime sizes.
    req_to_token_stride0,
    swa_lut_len,
    verify_capacity,
    # Compile-time constants.
    INNER_BLOCK: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    HAS_SWA_LUT: tl.constexpr,
):
    """Entries kernel: materialize per-req verify entries. Grid = (bs, j_tiles).

    Each program owns one (req, j-tile) cell. Verify capacity is the upper bound on entries-per-req used to
    pick the grid; per-req actual count comes from ``verify_offsets[r+1] - verify_offsets[r]``.
    """
    r = tl.program_id(0)  # scalar
    tile_idx = tl.program_id(1)  # scalar

    rpi = tl.load(req_pool_indices_ptr + r)  # scalar
    prefix_lens = tl.load(prefix_lens_ptr + r)  # scalar

    # Skip padding rows entirely.
    if rpi == 0:
        return

    window_start = _compute_window_start(prefix_lens, SWA_WINDOW)  # scalar

    verify_start = tl.load(verify_offsets_ptr + r)  # scalar
    verify_end = tl.load(verify_offsets_ptr + r + 1)  # scalar
    my_verify_len = verify_end - verify_start  # scalar

    if my_verify_len <= 0:
        return

    j_offs = tile_idx * INNER_BLOCK + tl.arange(0, INNER_BLOCK)  # [INNER_BLOCK]
    j_mask = j_offs < my_verify_len  # [INNER_BLOCK] bool

    positions = window_start + j_offs  # [INNER_BLOCK]
    rpi_i64 = rpi.to(tl.int64)  # scalar
    stride_i64 = req_to_token_stride0  # scalar
    positions_i64 = positions.to(tl.int64)  # [INNER_BLOCK]

    slot_full = tl.load(  # [INNER_BLOCK]
        req_to_token_ptr + rpi_i64 * stride_i64 + positions_i64,
        mask=j_mask,
        other=0,
    )

    prev_pos_valid = (positions > 0) & j_mask  # [INNER_BLOCK] bool
    prev_positions_i64 = (positions - 1).to(tl.int64)  # [INNER_BLOCK]
    safe_prev_positions_i64 = tl.where(
        prev_pos_valid, prev_positions_i64, 0
    )  # [INNER_BLOCK]
    prev_slot_full = tl.load(  # [INNER_BLOCK]
        req_to_token_ptr + rpi_i64 * stride_i64 + safe_prev_positions_i64,
        mask=prev_pos_valid,
        other=0,
    )

    if HAS_SWA_LUT:
        slot = _swa_translate_tile(
            slot_full, j_mask, full_to_swa_lut_ptr, swa_lut_len
        )  # [INNER_BLOCK]
        prev_translated = _swa_translate_tile(  # [INNER_BLOCK]
            prev_slot_full,
            prev_pos_valid,
            full_to_swa_lut_ptr,
            swa_lut_len,
        )
    else:
        slot = slot_full
        prev_translated = prev_slot_full

    chain_head_tile = tl.full((INNER_BLOCK,), -1, dtype=slot.dtype)  # [INNER_BLOCK]
    prev_slot = tl.where(
        prev_pos_valid, prev_translated, chain_head_tile
    )  # [INNER_BLOCK]

    out_offs = verify_start + j_offs  # [INNER_BLOCK]
    cap_mask = out_offs < verify_capacity  # [INNER_BLOCK] bool
    write_mask = j_mask & cap_mask  # [INNER_BLOCK] bool

    tl.store(
        out_verify_slot_indices_ptr + out_offs,
        slot.to(tl.int64),
        mask=write_mask,
    )
    tl.store(
        out_verify_positions_ptr + out_offs,
        positions.to(tl.int64),
        mask=write_mask,
    )
    tl.store(
        out_verify_prev_slot_indices_ptr + out_offs,
        prev_slot.to(tl.int64),
        mask=write_mask,
    )


@triton.jit
def _compute_window_start(prefix_lens, SWA_WINDOW: tl.constexpr):
    """Per-req window start: max(prefix_lens - SWA_WINDOW, 0) when SWA, else 0.
    Works for tile and scalar inputs (broadcasts via prefix_lens shape).
    """
    if SWA_WINDOW > 0:
        clipped = prefix_lens - SWA_WINDOW
        return tl.where(clipped > 0, clipped, 0)
    else:
        return prefix_lens - prefix_lens


@triton.jit
def _swa_translate_tile(raw, mask, lut_ptr, lut_len):
    """SWA-translate a tile of slot indices. Sentinels (raw < 0) are passed through unchanged.

    ``lut_len`` is the LUT's length (Python int from the host wrapper); when 0 the LUT is unused (the caller
    will only enter this branch when HAS_SWA_LUT is True, so lut_len is always > 0 in practice).
    """
    sentinel = raw < 0
    safe = tl.where(sentinel, 0, raw)
    if lut_len > 0:
        safe = tl.where(safe >= lut_len, lut_len - 1, safe)
    xlat = tl.load(lut_ptr + safe, mask=mask & (~sentinel), other=0)
    return tl.where(sentinel, raw, xlat)


@triton.jit
def _plan_extras_kernel(
    # Input pointers.
    extra_slot_ptr,
    extra_positions_ptr,
    extra_prev_slot_ptr,
    extra_num_valid_ptr,
    verify_offsets_ptr,
    # Output pointers.
    out_verify_slot_indices_ptr,
    out_verify_positions_ptr,
    out_verify_prev_slot_indices_ptr,
    # Runtime sizes.
    bs,
    verify_capacity,
    # Compile-time constants.
    INNER_BLOCK: tl.constexpr,
):
    """Extras kernel: append extras into the verify tail at base = verify_offsets[bs]. Grid = (k_tiles,).

    Extras are caller-pre-translated; this kernel only copies (no LUT pass).
    """
    tile_idx = tl.program_id(0)  # scalar
    k_offs = tile_idx * INNER_BLOCK + tl.arange(0, INNER_BLOCK)  # [INNER_BLOCK]

    extras_count = tl.load(extra_num_valid_ptr)  # scalar
    extras_count = tl.where(extras_count > 0, extras_count, 0)
    in_range_mask = k_offs < extras_count  # [INNER_BLOCK] bool

    base_idx = tl.load(verify_offsets_ptr + bs)  # scalar

    slots = tl.load(
        extra_slot_ptr + k_offs, mask=in_range_mask, other=0
    )  # [INNER_BLOCK]
    positions = tl.load(
        extra_positions_ptr + k_offs, mask=in_range_mask, other=0
    )  # [INNER_BLOCK]
    prevs = tl.load(
        extra_prev_slot_ptr + k_offs, mask=in_range_mask, other=0
    )  # [INNER_BLOCK]

    out_offs = base_idx + k_offs  # [INNER_BLOCK]
    cap_mask = out_offs < verify_capacity  # [INNER_BLOCK] bool
    write_mask = in_range_mask & cap_mask  # [INNER_BLOCK] bool

    tl.store(
        out_verify_slot_indices_ptr + out_offs, slots.to(tl.int64), mask=write_mask
    )
    tl.store(
        out_verify_positions_ptr + out_offs, positions.to(tl.int64), mask=write_mask
    )
    tl.store(
        out_verify_prev_slot_indices_ptr + out_offs, prevs.to(tl.int64), mask=write_mask
    )
