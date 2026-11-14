# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# The following kernel is imported from ATOM.
# Source: atom/model_ops/v4_kernels/paged_decode_indices.py

"""V4 paged-decode index scatter — single Triton kernel writes SWA window-
prefix paged offsets into the three ragged-packed destination buffers
(`kv_indices_swa` / `kv_indices_csa` / `kv_indices_hca`).

Replaces the prior chain (numpy `_build_window_topk_np` + `index_copy_`):

    window_topk_np = _build_window_topk_np(positions, win, ring_stride)  # [T, win]
    swa_paged_2d = torch.where(window_topk >= 0, slot * ring_stride + topk, -1)
    swa_paged_flat = swa_paged_2d.reshape(-1)
    swa_indices_gpu[:T*win].copy_(swa_paged_flat)
    csa_indices_gpu.index_copy_(0, csa_win_pos, swa_paged_flat)
    hca_indices_gpu.index_copy_(0, hca_win_pos, swa_paged_flat)

Two simplifications vs the prior implementation (see plan
`sequential-noodling-turing.md` for details):

1. The ring-index formula `ring = (pos - win + 1 + w) % ring_stride` is computed
   inline inside the kernel from `positions[t]`. The `[T, win]`
   `window_topk` intermediate buffer (mnbt·win·4 = 4 MB at typical config)
   is gone; no separate CPU build + H2D copy.
2. The destination layout is now ragged-packed (same as prefill): each
   token's SWA prefix segment has length `n = min(positions[t]+1, win)`
   (NOT a fixed `win` padded with `-1` sentinels). The caller's
   `swa_indptr` / `csa_indptr` / `hca_indptr` reflect this ragged sizing.

Bytewise correctness: for tokens with `position >= win-1` (all `n == win`),
the output is identical to the prior implementation. For shorter
positions, the prior layout wrote `(win - n)` leading `-1` entries that
the sparse-attention kernel masked out; the new layout omits those slots
entirely, saving sparse-attn loop iterations.

Caller contract:
- Grid = T (one program per token).
- `batch_id_per_token[:T]` may carry `-1` sentinels in the CG-padded tail —
  kernel checks and bails (matches `_attach_v4_per_fwd_meta` convention).
- `swa_indptr` / `csa_indptr` / `hca_indptr` must reflect the ragged-packed
  sizing: per-token slot count = `min(positions[t]+1, win) + n_compress[t]`
  where `n_compress[t]` is 0 for SWA, `min(n_committed_csa, index_topk)`
  for CSA, `n_committed_hca` for HCA.
- `swa_indices` / `csa_indices` / `hca_indices` capacity ≥ corresponding
  indptr[T]; this kernel only writes the SWA-prefix segment
  `[indptr[t], indptr[t] + n)` per token. The compress-tail is filled
  elsewhere (HCA: numpy fill in caller, CSA: `csa_translate_pack` per layer).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_paged_decode_indices_kernel(
    state_slot_per_seq_ptr,  # [bs] int32
    batch_id_per_token_ptr,  # [T+pad] int — sentinel -1 in pad tail
    positions_ptr,  # [T+pad] int — global token position
    swa_indptr_ptr,  # [T+1] int32 — ragged SWA-prefix cumsum
    csa_indptr_ptr,  # [T+1] int32 — ragged (SWA + CSA topk)
    hca_indptr_ptr,  # [T+1] int32 — ragged (SWA + HCA committed)
    swa_indices_ptr,  # [swa_total] int32, output
    csa_indices_ptr,  # [csa_total] int32, output (writes SWA-prefix segment only)
    hca_indices_ptr,  # [hca_total] int32, output (writes SWA-prefix segment only)
    ring_stride,  # win_with_spec — stride into unified_kv SWA region (paper §3.6.1)
    win: tl.constexpr,  # window_size — max SWA prefix slots
    BLOCK_N: tl.constexpr,  # next_pow2(win)
):
    """One program per token. Writes `n = min(positions[t]+1, win)` paged
    offsets to the SWA prefix segment of each of SWA/CSA/HCA index buffers.

    For token `t`:
        bid = batch_id_per_token[t]                  # bail if -1 (CG pad)
        slot = state_slot_per_seq[bid]
        pos = positions[t]
        n = min(pos + 1, win)
        # Old -1 sentinels were at the leading `win - n` cols; reparameterize
        # to skip them: i in [0, n) → abs_pos = pos - n + 1 + i ∈ [0, pos].
        for i in range(n):
            abs_pos = pos - n + 1 + i
            ring = abs_pos % ring_stride
            paged = slot * ring_stride + ring
            swa_indices[swa_indptr[t] + i] = paged
            csa_indices[csa_indptr[t] + i] = paged
            hca_indices[hca_indptr[t] + i] = paged
    """
    t = tl.program_id(0)
    bid = tl.load(batch_id_per_token_ptr + t)
    if bid < 0:
        return  # CG-padded sentinel — leave outputs untouched

    slot = tl.load(state_slot_per_seq_ptr + bid)
    pos = tl.load(positions_ptr + t)
    # `n` = actual valid SWA prefix count. Cast to match `win` (compile-time
    # int) — pos is i32/i64 from positions buffer.
    n = tl.minimum(pos + 1, win)
    swa_base = tl.load(swa_indptr_ptr + t)
    csa_base = tl.load(csa_indptr_ptr + t)
    hca_base = tl.load(hca_indptr_ptr + t)

    i = tl.arange(0, BLOCK_N)
    mask = i < n
    abs_pos = pos - n + 1 + i  # ∈ [0, pos] for valid i
    ring_idx = abs_pos % ring_stride
    paged = slot * ring_stride + ring_idx

    tl.store(swa_indices_ptr + swa_base + i, paged, mask=mask)
    tl.store(csa_indices_ptr + csa_base + i, paged, mask=mask)
    tl.store(hca_indices_ptr + hca_base + i, paged, mask=mask)


def write_v4_paged_decode_indices(
    *,
    state_slot_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    swa_indptr: torch.Tensor,
    csa_indptr: torch.Tensor,
    hca_indptr: torch.Tensor,
    swa_indices: torch.Tensor,
    csa_indices: torch.Tensor,
    hca_indices: torch.Tensor,
    T: int,
    win: int,
    ring_stride: int,
) -> None:
    """In-place fill SWA / CSA / HCA window-prefix offsets via a single
    Triton kernel. Replaces the prior `_build_window_topk_np` (CPU O(T·win))
    + `index_copy_` chain. All inputs are persistent forward_vars buffers —
    no allocator churn.

    Args (all GPU tensors except T/win/ring_stride):
      state_slot_per_seq:  [bs]   int32 — per-seq state cache slot.
      batch_id_per_token:  [>=T]  int   — token→seq map; -1 sentinel skipped.
      positions:           [>=T]  int   — global token position
                                   (forward_vars["positions"]); used to derive
                                   `n = min(pos+1, win)` per token + the ring
                                   index `(pos - n + 1 + i) % ring_stride`.
      swa_indptr:          [>=T+1] int32 — ragged SWA-prefix cumsum, where
                                   `swa_indptr[t+1] - swa_indptr[t] =
                                    min(positions[t]+1, win)`.
      csa_indptr:          [>=T+1] int32 — ragged CSA buffer indptr (SWA
                                   prefix + CSA topk per token).
      hca_indptr:          [>=T+1] int32 — ragged HCA buffer indptr (SWA
                                   prefix + HCA committed per token).
      swa_indices:         [>=swa_indptr[T]] int32 OUT — fully written by
                                   this kernel (no other source).
      csa_indices:         [>=csa_indptr[T]] int32 OUT — window-prefix
                                   `[csa_indptr[t], +n)` written here; CSA
                                   topk tail filled per-layer by
                                   `csa_translate_pack`.
      hca_indices:         [>=hca_indptr[T]] int32 OUT — same semantics; HCA
                                   compress tail filled in the caller via
                                   numpy fill.
      T:                   int — number of real tokens (grid size).
      win:                 int — SWA window size (typically 128 for V4-Pro).
      ring_stride:                  int — `win_with_spec = window_size + max_spec_steps`,
                                 stride into unified_kv SWA region per slot
                                 AND modulo for ring-index wrap.
    """
    if T == 0:
        return
    assert state_slot_per_seq.dim() == 1
    assert batch_id_per_token.dim() == 1 and batch_id_per_token.shape[0] >= T
    assert positions.dim() == 1 and positions.shape[0] >= T
    assert swa_indptr.dim() == 1 and swa_indptr.shape[0] >= T + 1
    assert csa_indptr.dim() == 1 and csa_indptr.shape[0] >= T + 1
    assert hca_indptr.dim() == 1 and hca_indptr.shape[0] >= T + 1
    assert swa_indices.dim() == 1
    assert csa_indices.dim() == 1
    assert hca_indices.dim() == 1

    BLOCK_N = triton.next_power_of_2(win)
    _v4_paged_decode_indices_kernel[(T,)](
        state_slot_per_seq,
        batch_id_per_token,
        positions,
        swa_indptr,
        csa_indptr,
        hca_indptr,
        swa_indices,
        csa_indices,
        hca_indices,
        ring_stride,
        win=win,
        BLOCK_N=BLOCK_N,
    )
