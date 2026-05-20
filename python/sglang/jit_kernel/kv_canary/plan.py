"""Triton host wrapper for the canary plan accumulator.

Defines :func:`canary_plan_step` — the single-launch Triton plan kernel
that fills a :class:`~sglang.jit_kernel.kv_canary.verify.VerifyPlan`
and a :class:`~sglang.jit_kernel.kv_canary.write.WritePlan` from
ForwardBatch primitives plus optional pre-walked flat verify extras.

Byte-equal pinned by
:func:`sglang.jit_kernel.kv_canary.plan_ref.canary_plan_step_torch_reference`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kv_canary.verify import (
    VerifyPlan,
    _assert_contiguous,
)
from sglang.jit_kernel.kv_canary.write import WritePlan

# Upper bound on bs for the phase-1 block-level cumsum. Reqs larger than this exceed Triton's single-program
# tl.cumsum reach. Increase if real workloads ever push past it; the cap is intentionally generous so the
# wrapper never silently truncates.
_PLAN_BS_BLOCK_SIZE: int = 1024

# Inner-tile width for the verify materialization phase. Each (req, j-tile) program owns this many entries
# along the j-axis of the (bs, max_verify_per_req) logical grid.
_PLAN_VERIFY_INNER_BLOCK: int = 64

# Inner-tile width for the extras-append phase. Each program copies this many extras into the verify tail.
_PLAN_EXTRAS_INNER_BLOCK: int = 64


@dataclass(frozen=True, slots=True, kw_only=True)
class _PlanScratch:
    """Per-device scratch used by ``canary_plan_step`` across its sub-kernels.

    The verify cumsum produced by ``_plan_offsets_kernel`` must survive into ``_plan_entries_kernel`` (the
    entry materializer needs each req's flat-output offset) but ``VerifyPlan`` does not carry an offsets
    tensor of its own. We cache a stable scratch tensor on the host wrapper so its data_ptr() is captured
    into any cuda-graph and reused on replay.

    Fields:
        verify_offsets: Exclusive prefix sum of per-req verify counts, shape [_PLAN_BS_BLOCK_SIZE + 1], int32.
            Slot ``[bs]`` carries the total verify entry count (= base index for extras append).
        swa_lut_sentinel: One-element int32 zero tensor used as a stable LUT pointer when the caller passes
            ``full_to_swa_index_mapping=None``. Cached so the wrapper never allocates per call; its data_ptr()
            is pinned into any cuda-graph capture and the kernel never dereferences it (HAS_SWA_LUT is False).
    """

    verify_offsets: torch.Tensor
    swa_lut_sentinel: torch.Tensor


_PLAN_SCRATCH_CACHE: dict[torch.device, _PlanScratch] = {}


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
            int32. 0 is the padding sentinel.
        fb_prefix_lens: Per-req prefix length already written before this step, shape [bs], int32. Caller
            normalizes: extend → ForwardBatch.extend_prefix_lens, decode → ForwardBatch.seq_lens - 1, sweep
            over running → seq_lens.
        fb_extend_seq_lens: ForwardBatch.extend_seq_lens; per-req tokens being written this step, shape [bs],
            int32. 1 for pure decode; 0 for sweep.
        req_to_token: ReqToTokenPool.req_to_token; full-pool slot index table, shape [max_reqs, max_seq_len],
            int32.
        extra_verify_slot_indices: Pre-walked extra verify slots, shape [extra_verify_capacity], int32.
            Caller-translated to the target index space.
        extra_verify_positions: Same shape, int32. Expected position per extra entry.
        extra_verify_prev_slot_indices: Same shape, int32. -1 for chain-seed extras.
        extra_verify_num_valid: Active extra entry count, shape [1], int32. 0 for per-forward and running-sweep
            callers.
        swa_window_size: 0 for the FULL canary group; positive window length for the SWA group.
        full_to_swa_index_mapping: SWA LUT, shape [full_pool_size + 1], int32, or None. Required (non-None) iff
            swa_window_size > 0. Used to translate verify slot indices and chain-seed slot indices at plan time.

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
    scratch = _get_plan_scratch(device=device)
    verify_offsets_scratch = scratch.verify_offsets

    verify_capacity = int(verify_plan_out.verify_slot_indices.shape[0])
    write_req_capacity_plus_one = int(write_plan_out.write_offsets.shape[0])
    write_req_capacity = int(write_plan_out.write_seed_slot_indices.shape[0])
    extras_capacity = int(extra_verify_slot_indices.shape[0])

    has_swa_lut = full_to_swa_index_mapping is not None
    if has_swa_lut:
        lut_tensor = full_to_swa_index_mapping
        lut_len = int(full_to_swa_index_mapping.shape[0])
    else:
        # Sentinel one-element tensor keeps the pointer well-defined; kernel never dereferences it when
        # HAS_SWA_LUT is False. Cached on the scratch so we never allocate per call.
        lut_tensor = scratch.swa_lut_sentinel
        lut_len = 0

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
        write_plan_out.write_num_valid_reqs,
        bs,
        req_to_token_stride0,
        lut_len,
        BS_BLOCK=_PLAN_BS_BLOCK_SIZE,
        SWA_WINDOW=int(swa_window_size),
        HAS_SWA_LUT=has_swa_lut,
        WRITE_OFFSETS_LEN=write_req_capacity_plus_one,
        WRITE_REQ_CAPACITY=write_req_capacity,
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


def _get_plan_scratch(*, device: torch.device) -> _PlanScratch:
    """Return the cached scratch for ``device``; allocate once on first call.

    Allocation happens at most once per device. The returned tensor's data_ptr() is stable for the lifetime of
    the process, which is what cuda-graph capture requires: the very first canary_plan_step call (eager mode
    or under capture) pins the address, and every later call — including replays — reuses it.
    """
    cached = _PLAN_SCRATCH_CACHE.get(device)
    if cached is not None:
        return cached
    scratch = _PlanScratch(
        verify_offsets=torch.zeros(
            _PLAN_BS_BLOCK_SIZE + 1, dtype=torch.int32, device=device
        ),
        swa_lut_sentinel=torch.zeros(1, dtype=torch.int32, device=device),
    )
    _PLAN_SCRATCH_CACHE[device] = scratch
    return scratch


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
    verify_offsets_ptr,
    write_offsets_ptr,
    write_seed_slot_indices_ptr,
    verify_num_valid_ptr,
    write_num_valid_reqs_ptr,
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
):
    """Offsets kernel: per-req counts, seeds, exclusive-prefix-sum offsets, scalar totals.

    Single program; BLOCK_BS-wide tiles cover the full bs (caller ensures bs <= BS_BLOCK). All cumsum is done
    via block-level ``tl.cumsum`` in one program — no cross-program sync needed.
    """
    bs_offs = tl.arange(0, BS_BLOCK)
    bs_mask = bs_offs < bs

    # Per-req inputs (int32 loads matching the dtype of fb_* / req_to_token).
    rpi = tl.load(req_pool_indices_ptr + bs_offs, mask=bs_mask, other=0)
    prefix_lens = tl.load(prefix_lens_ptr + bs_offs, mask=bs_mask, other=0)
    extend_lens = tl.load(extend_seq_lens_ptr + bs_offs, mask=bs_mask, other=0)

    not_padding = (rpi != 0) & bs_mask

    if SWA_WINDOW > 0:
        clipped = prefix_lens - SWA_WINDOW
        window_starts = tl.where(clipped > 0, clipped, 0)
    else:
        window_starts = tl.zeros((BS_BLOCK,), dtype=prefix_lens.dtype)

    verify_lens = prefix_lens - window_starts
    verify_lens = tl.where(verify_lens > 0, verify_lens, 0)
    verify_lens = tl.where(not_padding, verify_lens, 0)

    write_lens = tl.where(extend_lens > 0, extend_lens, 0)
    write_lens = tl.where(not_padding, write_lens, 0)

    # Seed slot per req. prefix_lens == 0 means no prefix → -1 sentinel. Padding row → no write contribution
    # → -1 sentinel either way; we also mask write_lens onto seed below to match the ref's "no write → -1".
    safe_prefix_pos = tl.where(prefix_lens > 0, prefix_lens - 1, 0)
    stride_i64 = req_to_token_stride0.to(tl.int64)
    seed_full = tl.load(
        req_to_token_ptr + rpi.to(tl.int64) * stride_i64 + safe_prefix_pos.to(tl.int64),
        mask=bs_mask & (prefix_lens > 0),
        other=0,
    )

    if HAS_SWA_LUT:
        seed_sentinel = seed_full < 0
        seed_safe = tl.where(seed_sentinel, 0, seed_full)
        if swa_lut_len > 0:
            seed_safe = tl.where(seed_safe >= swa_lut_len, swa_lut_len - 1, seed_safe)
        seed_xlat = tl.load(
            full_to_swa_lut_ptr + seed_safe,
            mask=bs_mask & (prefix_lens > 0) & (~seed_sentinel),
            other=0,
        )
        seed_translated = tl.where(seed_sentinel, seed_full, seed_xlat)
    else:
        seed_translated = seed_full

    # Reqs with no write contribution should expose seed = -1 (ref's _seed_slot is masked by write_lens > 0).
    minus_one = tl.full((BS_BLOCK,), -1, dtype=seed_translated.dtype)
    seed_slot = tl.where(
        (prefix_lens > 0) & (write_lens > 0), seed_translated, minus_one
    )

    # Inclusive cumsum → exclusive offsets via subtraction.
    verify_inclusive = tl.cumsum(verify_lens, axis=0)
    write_inclusive = tl.cumsum(write_lens, axis=0)
    verify_exclusive = verify_inclusive - verify_lens
    write_exclusive = write_inclusive - write_lens

    # Scatter exclusive offsets into the [bs+1]-sized output tensor. Positions [0, bs) get the exclusive sum;
    # position bs gets the total (totals = verify_inclusive at index bs - 1 if bs > 0, else 0).
    out_offsets_mask = bs_mask
    tl.store(
        verify_offsets_ptr + bs_offs,
        verify_exclusive.to(tl.int32),
        mask=out_offsets_mask,
    )
    write_offsets_mask = bs_offs < WRITE_OFFSETS_LEN
    tl.store(
        write_offsets_ptr + bs_offs,
        write_exclusive.to(tl.int32),
        mask=write_offsets_mask & bs_mask,
    )

    # Scatter seed slots (capped to write_req_capacity).
    seed_mask = bs_mask & (bs_offs < WRITE_REQ_CAPACITY)
    tl.store(
        write_seed_slot_indices_ptr + bs_offs,
        seed_slot.to(tl.int32),
        mask=seed_mask,
    )

    # Totals: sum of all per-req lens. Same value as the last inclusive entry but tl.sum is robust to bs == 0.
    total_verify = tl.sum(verify_lens, axis=0)
    total_write = tl.sum(write_lens, axis=0)

    # Store the [bs] slot of verify_offsets and write_offsets (one element past the last per-req entry).
    # verify_offsets scratch has length BS_BLOCK + 1 so the bs slot is always in range.
    tl.store(verify_offsets_ptr + bs, total_verify.to(tl.int32))
    # write_offsets has length WRITE_OFFSETS_LEN = write_req_capacity + 1; only store if in range.
    write_tail_in_range = bs < WRITE_OFFSETS_LEN
    tl.store(
        write_offsets_ptr + bs,
        total_write.to(tl.int32),
        mask=write_tail_in_range,
    )

    # Scalar writes: verify_num_valid = total_verify + extras_count; write_num_valid_reqs = bs.
    extras_count = tl.load(extra_verify_num_valid_ptr)
    extras_count = tl.where(extras_count > 0, extras_count, 0)
    tl.store(verify_num_valid_ptr, (total_verify + extras_count).to(tl.int32))
    tl.store(write_num_valid_reqs_ptr, tl.full((), bs, tl.int32))


@triton.jit
def _plan_entries_kernel(
    # Input pointers.
    req_pool_indices_ptr,
    prefix_lens_ptr,
    req_to_token_ptr,
    full_to_swa_lut_ptr,
    verify_offsets_ptr,
    # Output pointers.
    verify_slot_indices_ptr,
    verify_positions_ptr,
    verify_prev_slot_indices_ptr,
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
    r = tl.program_id(0)
    tile_idx = tl.program_id(1)

    rpi = tl.load(req_pool_indices_ptr + r)
    prefix_lens = tl.load(prefix_lens_ptr + r)

    # Skip padding rows entirely.
    if rpi == 0:
        return

    if SWA_WINDOW > 0:
        clipped = prefix_lens - SWA_WINDOW
        window_start = tl.where(clipped > 0, clipped, 0)
    else:
        window_start = prefix_lens - prefix_lens

    verify_start = tl.load(verify_offsets_ptr + r)
    verify_end = tl.load(verify_offsets_ptr + r + 1)
    my_verify_len = verify_end - verify_start

    if my_verify_len <= 0:
        return

    j_offs = tile_idx * INNER_BLOCK + tl.arange(0, INNER_BLOCK)
    j_mask = j_offs < my_verify_len

    positions = window_start + j_offs
    rpi_i64 = rpi.to(tl.int64)
    stride_i64 = req_to_token_stride0.to(tl.int64)
    positions_i64 = positions.to(tl.int64)

    slot_full = tl.load(
        req_to_token_ptr + rpi_i64 * stride_i64 + positions_i64,
        mask=j_mask,
        other=0,
    )

    prev_pos_valid = (positions > 0) & j_mask
    prev_positions_i64 = (positions - 1).to(tl.int64)
    safe_prev_positions_i64 = tl.where(prev_pos_valid, prev_positions_i64, 0)
    prev_slot_full = tl.load(
        req_to_token_ptr + rpi_i64 * stride_i64 + safe_prev_positions_i64,
        mask=prev_pos_valid,
        other=0,
    )

    if HAS_SWA_LUT:
        slot = _swa_translate_tile(slot_full, j_mask, full_to_swa_lut_ptr, swa_lut_len)
        prev_translated = _swa_translate_tile(
            prev_slot_full,
            prev_pos_valid,
            full_to_swa_lut_ptr,
            swa_lut_len,
        )
    else:
        slot = slot_full
        prev_translated = prev_slot_full

    chain_head_tile = tl.full((INNER_BLOCK,), -1, dtype=slot.dtype)
    prev_slot = tl.where(prev_pos_valid, prev_translated, chain_head_tile)

    out_offs = verify_start + j_offs
    cap_mask = out_offs < verify_capacity
    write_mask = j_mask & cap_mask

    tl.store(
        verify_slot_indices_ptr + out_offs,
        slot.to(tl.int32),
        mask=write_mask,
    )
    tl.store(
        verify_positions_ptr + out_offs,
        positions.to(tl.int32),
        mask=write_mask,
    )
    tl.store(
        verify_prev_slot_indices_ptr + out_offs,
        prev_slot.to(tl.int32),
        mask=write_mask,
    )


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
    verify_slot_indices_ptr,
    verify_positions_ptr,
    verify_prev_slot_indices_ptr,
    # Runtime sizes.
    bs,
    verify_capacity,
    # Compile-time constants.
    INNER_BLOCK: tl.constexpr,
):
    """Extras kernel: append extras into the verify tail at base = verify_offsets[bs]. Grid = (k_tiles,).

    Extras are caller-pre-translated; this kernel only copies (no LUT pass).
    """
    tile_idx = tl.program_id(0)
    k_offs = tile_idx * INNER_BLOCK + tl.arange(0, INNER_BLOCK)

    extras_count = tl.load(extra_num_valid_ptr)
    extras_count = tl.where(extras_count > 0, extras_count, 0)
    in_range_mask = k_offs < extras_count

    base_idx = tl.load(verify_offsets_ptr + bs)

    slots = tl.load(extra_slot_ptr + k_offs, mask=in_range_mask, other=0)
    positions = tl.load(extra_positions_ptr + k_offs, mask=in_range_mask, other=0)
    prevs = tl.load(extra_prev_slot_ptr + k_offs, mask=in_range_mask, other=0)

    out_offs = base_idx + k_offs
    cap_mask = out_offs < verify_capacity
    write_mask = in_range_mask & cap_mask

    tl.store(verify_slot_indices_ptr + out_offs, slots.to(tl.int32), mask=write_mask)
    tl.store(verify_positions_ptr + out_offs, positions.to(tl.int32), mask=write_mask)
    tl.store(
        verify_prev_slot_indices_ptr + out_offs, prevs.to(tl.int32), mask=write_mask
    )
