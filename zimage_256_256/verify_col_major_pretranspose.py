#!/usr/bin/env python3
"""
Verify simple col-major transpose (no TMA padding) for sfb pre-transpose.

Approach: Bs.T.contiguous().T → stride=(1, N/128), NO padding.
This should:
  1. Pass SM90 check_sf_layout col-major check
  2. Be detected as already-aligned by get_mn_major_tma_aligned_tensor
  3. Eliminate the 139.9ms sfb transpose overhead

Test with ALL actual model shapes (including N/128 % 4 != 0).

Usage:
    python zimage_256_256/verify_col_major_pretranspose.py
"""

import torch

print("=" * 70)
print("Col-Major Pre-Transpose Verification")
print("=" * 70)

from sglang.srt.layers.deep_gemm_wrapper.configurer import (
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
)

assert ENABLE_JIT_DEEPGEMM

import deep_gemm
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

block_size = [128, 128]
M = 768

# All actual model GEMM shapes
gemm_shapes = [
    (3840, 3840, "qkvo/ffn_w2"),
    (20480, 3840, "ffn_w1w3"),
    (3840, 10240, "ffn_out"),
]

# ── Step 1: Correctness — does simple col-major Bs produce correct GEMM? ──
print(f"\n[Step 1: Correctness — Bs.T.contiguous().T on actual shapes]")

all_pass = True
for N, K, desc in gemm_shapes:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    B_fp8, Bs = per_block_cast_to_fp8(B_bf16)
    A_fp8, As = sglang_per_token_group_quant_fp8(
        A_bf16,
        block_size[1],
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
    )

    # Simple col-major: no TMA padding
    Bs_col = Bs.T.contiguous().T

    print(f"\n  {desc}: N={N}, K={K}")
    print(f"    Bs orig:    shape={Bs.shape} stride={Bs.stride()}")
    print(f"    Bs col-maj: shape={Bs_col.shape} stride={Bs_col.stride()}")
    print(
        f"    check_sf_layout col-major: "
        f"stride(-1)={Bs_col.stride(-1)} == size(-2)={Bs_col.size(-2)}: "
        f"{Bs_col.stride(-1) == Bs_col.size(-2)}"
    )

    # Run GEMM with original row-major Bs
    out_orig = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((A_fp8, As), (B_fp8, Bs), out_orig)

    # Run GEMM with col-major Bs
    try:
        out_col = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt((A_fp8, As), (B_fp8, Bs_col), out_col)

        diff = (out_orig - out_col).abs().max().item()
        passed = diff < 1e-3
        all_pass = all_pass and passed
        print(f"    GEMM: max_diff={diff:.6f} {'✅' if passed else '❌'}")
    except Exception as e:
        all_pass = False
        print(f"    GEMM: ❌ CRASHED: {e}")

print(f"\n  Overall: {'✅ All shapes pass' if all_pass else '❌ FAILURES'}")

if not all_pass:
    print(f"\n  Col-major pre-transpose does NOT work. Stopping.")
    exit(1)

# ── Step 2: Idempotency — does DeepGemm skip transpose for col-major? ────
print(f"\n[Step 2: Does get_mn_major_tma_aligned_tensor skip for col-major Bs?]")

for N, K, desc in gemm_shapes:
    B_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    _, Bs = per_block_cast_to_fp8(B_bf16)
    Bs_col = Bs.T.contiguous().T

    aligned = get_mn_major_tma_aligned_tensor(Bs_col)
    is_same = (
        aligned.data_ptr() == Bs_col.data_ptr()
        and aligned.shape == Bs_col.shape
        and aligned.stride() == Bs_col.stride()
    )
    print(
        f"  {desc} Bs_col {Bs_col.shape} stride={Bs_col.stride()} → "
        f"aligned stride={aligned.stride()} "
        f"same_ptr={aligned.data_ptr() == Bs_col.data_ptr()} "
        f"{'✅ SKIP (no copy)' if is_same else '❌ RE-TRANSPOSED'}"
    )

# ── Step 3: CUDA event timing — per-GEMM savings on actual shapes ────────
print(f"\n[Step 3: CUDA event timing — per-GEMM savings on actual shapes]")

start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)
n_iters = 5000

total_save_per_step = 0.0

for N, K, desc in gemm_shapes:
    A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    B_fp8, Bs = per_block_cast_to_fp8(B_bf16)
    A_fp8, As = sglang_per_token_group_quant_fp8(
        A_bf16,
        block_size[1],
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
    )
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    Bs_col = Bs.T.contiguous().T

    # Warmup
    for _ in range(300):
        deep_gemm.fp8_gemm_nt((A_fp8, As), (B_fp8, Bs), out)
        deep_gemm.fp8_gemm_nt((A_fp8, As), (B_fp8, Bs_col), out)
    torch.cuda.synchronize()

    # row-major Bs
    start_ev.record()
    for _ in range(n_iters):
        deep_gemm.fp8_gemm_nt((A_fp8, As), (B_fp8, Bs), out)
    end_ev.record()
    torch.cuda.synchronize()
    t_row = start_ev.elapsed_time(end_ev) / n_iters * 1000  # μs

    # col-major Bs
    start_ev.record()
    for _ in range(n_iters):
        deep_gemm.fp8_gemm_nt((A_fp8, As), (B_fp8, Bs_col), out)
    end_ev.record()
    torch.cuda.synchronize()
    t_col = start_ev.elapsed_time(end_ev) / n_iters * 1000  # μs

    save_us = t_row - t_col
    print(f"  {desc} (N={N}, K={K}):")
    print(f"    row-major Bs: {t_row:.1f} μs   col-major Bs: {t_col:.1f} μs")
    print(f"    saving: {save_us:.1f} μs/call")

    # Estimate calls per denoising step per layer for this shape
    # (from nsys instance counts / ~20 steps)
    total_save_per_step += save_us  # rough: 1 call per layer per shape

# ── Step 4: E2E projection ────────────────────────────────────────────────
print(f"\n[Step 4: E2E savings projection]")

# Model: ~40 layers, each layer has ~4 GEMM calls (qkv, out, w1w3, w2)
# ~20 denoising steps
# nsys: 1920 total GEMM calls for the full inference
# Approx: 1920 / 20 steps = 96 GEMMs per step

# Per-shape call counts (from nsys, divide by ~20 steps for per-step):
nsys_calls = {
    "qkvo/ffn_w2": 1280,  # N=3840, K=3840
    "ffn_w1w3": 320,  # N=20480, K=3840
    "ffn_out": 320,  # N=3840, K=10240
}

print(f"  nsys call counts (full inference):")
for shape_desc, count in nsys_calls.items():
    print(f"    {shape_desc}: {count} calls")

# Re-run to get per-shape savings (already measured above, but let's
# compute total projected savings
print(f"\n  To compute total savings, re-check Step 3 output above.")
print(f"  Multiply per-call saving × nsys call count for each shape.")

print(f"\n{'=' * 70}")
print(f"Done. If Step 1 all ✅ and Step 3 shows savings > 0,")
print(f"then Bs.T.contiguous().T is the correct pre-transpose approach.")
print(f"{'=' * 70}")
