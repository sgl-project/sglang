"""Triton fused kernels for TurboQuant on-the-fly dequant + matmul.

Optimized for AMD MI355X (ROCm):
- Wavefront size 64 awareness in block size selection
- MFMA-friendly tile sizes
- Codebook fits in registers (16 float32 = 64 bytes)

Kernel: _turboquant_fused_matmul_kernel
  Input: x_rot (pre-rotated activations), packed 4-bit indices, codebook, norms
  Output: x_rot @ codebook[indices].T * (norms / scale)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    pass


def has_triton() -> bool:
    return _HAS_TRITON


if _HAS_TRITON:

    @triton.jit
    def _codebook_lookup(idx_val, cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7,
                         cb8, cb9, cb10, cb11, cb12, cb13, cb14, cb15):
        """Manual 16-entry codebook lookup to avoid gather on AMD."""
        val = tl.where(idx_val == 0, cb0, cb1)
        val = tl.where(idx_val == 2, cb2, val)
        val = tl.where(idx_val == 3, cb3, val)
        val = tl.where(idx_val == 4, cb4, val)
        val = tl.where(idx_val == 5, cb5, val)
        val = tl.where(idx_val == 6, cb6, val)
        val = tl.where(idx_val == 7, cb7, val)
        val = tl.where(idx_val == 8, cb8, val)
        val = tl.where(idx_val == 9, cb9, val)
        val = tl.where(idx_val == 10, cb10, val)
        val = tl.where(idx_val == 11, cb11, val)
        val = tl.where(idx_val == 12, cb12, val)
        val = tl.where(idx_val == 13, cb13, val)
        val = tl.where(idx_val == 14, cb14, val)
        val = tl.where(idx_val == 15, cb15, val)
        return val

    @triton.jit
    def _turboquant_fused_matmul_kernel(
        input_ptr,
        indices_ptr,
        codebook_ptr,
        norms_ptr,
        output_ptr,
        B,
        N,
        K,
        PACKED_K,
        SCALE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused: output[b,n] = norm[n]/scale * sum_k x_rot[b,k] * codebook[indices[n,k]]"""
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)

        rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_b = rb < B
        mask_n = rn < N

        # Preload 16-entry codebook into scalar registers (avoids gather)
        cb0 = tl.load(codebook_ptr + 0).to(tl.float32)
        cb1 = tl.load(codebook_ptr + 1).to(tl.float32)
        cb2 = tl.load(codebook_ptr + 2).to(tl.float32)
        cb3 = tl.load(codebook_ptr + 3).to(tl.float32)
        cb4 = tl.load(codebook_ptr + 4).to(tl.float32)
        cb5 = tl.load(codebook_ptr + 5).to(tl.float32)
        cb6 = tl.load(codebook_ptr + 6).to(tl.float32)
        cb7 = tl.load(codebook_ptr + 7).to(tl.float32)
        cb8 = tl.load(codebook_ptr + 8).to(tl.float32)
        cb9 = tl.load(codebook_ptr + 9).to(tl.float32)
        cb10 = tl.load(codebook_ptr + 10).to(tl.float32)
        cb11 = tl.load(codebook_ptr + 11).to(tl.float32)
        cb12 = tl.load(codebook_ptr + 12).to(tl.float32)
        cb13 = tl.load(codebook_ptr + 13).to(tl.float32)
        cb14 = tl.load(codebook_ptr + 14).to(tl.float32)
        cb15 = tl.load(codebook_ptr + 15).to(tl.float32)

        acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < K

            inp_off = rb[:, None] * K + rk[None, :]
            inp_mask = mask_b[:, None] & mask_k[None, :]
            inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0).to(
                tl.float32
            )

            byte_col = rk // 2
            is_high = (rk % 2) == 1
            byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
            w_mask = mask_n[:, None] & mask_k[None, :]
            packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(
                tl.uint8
            )
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            idx = tl.where(is_high[None, :], hi, lo).to(tl.int32)

            w_quant = _codebook_lookup(idx, cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7,
                                       cb8, cb9, cb10, cb11, cb12, cb13, cb14, cb15)

            acc += tl.dot(inp_tile, tl.trans(w_quant))

        norm_vals = tl.load(norms_ptr + rn, mask=mask_n, other=1.0)
        acc = acc * (norm_vals[None, :] / SCALE)

        out_off = rb[:, None] * N + rn[None, :]
        out_mask = mask_b[:, None] & mask_n[None, :]
        tl.store(
            output_ptr + out_off,
            acc.to(output_ptr.dtype.element_ty),
            mask=out_mask,
        )

    @triton.jit
    def _load_codebook_16(cb_ptr):
        """Load all 16 codebook entries into registers."""
        return (
            tl.load(cb_ptr + 0).to(tl.float32), tl.load(cb_ptr + 1).to(tl.float32),
            tl.load(cb_ptr + 2).to(tl.float32), tl.load(cb_ptr + 3).to(tl.float32),
            tl.load(cb_ptr + 4).to(tl.float32), tl.load(cb_ptr + 5).to(tl.float32),
            tl.load(cb_ptr + 6).to(tl.float32), tl.load(cb_ptr + 7).to(tl.float32),
            tl.load(cb_ptr + 8).to(tl.float32), tl.load(cb_ptr + 9).to(tl.float32),
            tl.load(cb_ptr + 10).to(tl.float32), tl.load(cb_ptr + 11).to(tl.float32),
            tl.load(cb_ptr + 12).to(tl.float32), tl.load(cb_ptr + 13).to(tl.float32),
            tl.load(cb_ptr + 14).to(tl.float32), tl.load(cb_ptr + 15).to(tl.float32),
        )

    @triton.jit
    def _turboquant_fused_dual_pass_kernel(
        input_ptr,
        idx1_ptr,
        cb1_ptr,
        norms1_ptr,
        idx2_ptr,
        cb2_ptr,
        norms2_ptr,
        output_ptr,
        B,
        N,
        K,
        PACKED_K,
        SCALE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused dual-pass (residual): computes pass1 + pass2 in one kernel."""
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)

        rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_b = rb < B
        mask_n = rn < N

        # Preload both codebooks into registers
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 = \
            _load_codebook_16(cb1_ptr)
        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 = \
            _load_codebook_16(cb2_ptr)

        acc1 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < K

            inp_off = rb[:, None] * K + rk[None, :]
            inp_mask = mask_b[:, None] & mask_k[None, :]
            inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0).to(
                tl.float32
            )

            byte_col = rk // 2
            is_high = (rk % 2) == 1
            w_mask = mask_n[:, None] & mask_k[None, :]

            # Pass 1
            byte_off1 = rn[:, None] * PACKED_K + byte_col[None, :]
            packed1 = tl.load(idx1_ptr + byte_off1, mask=w_mask, other=0).to(tl.uint8)
            lo1 = packed1 & 0x0F
            hi1 = (packed1 >> 4) & 0x0F
            i1 = tl.where(is_high[None, :], hi1, lo1).to(tl.int32)
            w1 = _codebook_lookup(i1, a0, a1, a2, a3, a4, a5, a6, a7,
                                  a8, a9, a10, a11, a12, a13, a14, a15)
            acc1 += tl.dot(inp_tile, tl.trans(w1))

            # Pass 2
            byte_off2 = rn[:, None] * PACKED_K + byte_col[None, :]
            packed2 = tl.load(idx2_ptr + byte_off2, mask=w_mask, other=0).to(tl.uint8)
            lo2 = packed2 & 0x0F
            hi2 = (packed2 >> 4) & 0x0F
            i2 = tl.where(is_high[None, :], hi2, lo2).to(tl.int32)
            w2 = _codebook_lookup(i2, b0, b1, b2, b3, b4, b5, b6, b7,
                                  b8, b9, b10, b11, b12, b13, b14, b15)
            acc2 += tl.dot(inp_tile, tl.trans(w2))

        n1 = tl.load(norms1_ptr + rn, mask=mask_n, other=1.0)
        n2 = tl.load(norms2_ptr + rn, mask=mask_n, other=1.0)
        result = acc1 * (n1[None, :] / SCALE) + acc2 * (n2[None, :] / SCALE)

        out_off = rb[:, None] * N + rn[None, :]
        out_mask = mask_b[:, None] & mask_n[None, :]
        tl.store(
            output_ptr + out_off,
            result.to(output_ptr.dtype.element_ty),
            mask=out_mask,
        )


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _pick_block_size(dim: int, candidates: list[int]) -> int:
    """Pick largest power-of-2 block size <= dim from candidates."""
    best = candidates[0]
    for p in candidates:
        if p <= dim:
            best = p
    return best


def triton_fused_matmul(
    x_rot: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    K: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused dequant + matmul via Triton. Expects pre-rotated input."""
    if not _HAS_TRITON:
        raise RuntimeError("Triton not available")

    B = x_rot.shape[0]
    N = indices_packed.shape[0]
    PACKED_K = indices_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    # AMD MI355X: wavefront=64, prefer larger blocks
    BLOCK_B = _pick_block_size(B, [1, 2, 4, 8, 16, 32, 64])
    BLOCK_N = _pick_block_size(N, [1, 2, 4, 8, 16, 32, 64, 128])
    BLOCK_K = _pick_block_size(K, [1, 2, 4, 8, 16, 32, 64, 128])

    grid = (
        (B + BLOCK_B - 1) // BLOCK_B,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    _turboquant_fused_matmul_kernel[grid](
        x_rot,
        indices_packed,
        codebook,
        norms,
        output,
        B,
        N,
        K,
        PACKED_K,
        SCALE=scale,
        BLOCK_B=BLOCK_B,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output


def triton_fused_dual_pass_matmul(
    x_rot: torch.Tensor,
    idx1_packed: torch.Tensor,
    cb1: torch.Tensor,
    norms1: torch.Tensor,
    idx2_packed: torch.Tensor,
    cb2: torch.Tensor,
    norms2: torch.Tensor,
    K: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused residual (dual-pass) dequant + matmul in one kernel launch."""
    if not _HAS_TRITON:
        raise RuntimeError("Triton not available")

    B = x_rot.shape[0]
    N = idx1_packed.shape[0]
    PACKED_K = idx1_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    BLOCK_B = _pick_block_size(B, [1, 2, 4, 8, 16, 32, 64])
    BLOCK_N = _pick_block_size(N, [1, 2, 4, 8, 16, 32, 64, 128])
    BLOCK_K = _pick_block_size(K, [1, 2, 4, 8, 16, 32, 64, 128])

    grid = (
        (B + BLOCK_B - 1) // BLOCK_B,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    _turboquant_fused_dual_pass_kernel[grid](
        x_rot,
        idx1_packed,
        cb1,
        norms1,
        idx2_packed,
        cb2,
        norms2,
        output,
        B,
        N,
        K,
        PACKED_K,
        SCALE=scale,
        BLOCK_B=BLOCK_B,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output
