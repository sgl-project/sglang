#!/usr/bin/env python3
"""
Diagnostic v2: Understand the 139.8ms transpose overhead in FP8+DeepGemm.

Key insight from v1 crash: The model's actual shapes (N=3840, N=10240) have
N/128 = 30 and 80. get_mn_major_tma_aligned_tensor adds padding (align to 4),
and SM90 check_sf_layout REJECTS padded col-major layout for sfb.

So on SM90, DeepGemm's internal transform_sf code path for sfb does NOT use
get_mn_major_tma_aligned_tensor — it must use a DIFFERENT transform.

This script investigates:
  Q1: What exact transform does DeepGemm apply to sfb on SM90?
  Q2: Can we replicate that transform at load time?
  Q3: CUDA-event measurement of per-GEMM sfb cost (using safe shapes + actual shapes)
  Q4: What about sfa — does col-major from sglang avoid transpose?

Usage:
    python zimage_256_256/diagnose_sfa_transpose.py
"""

import time

import torch

print("=" * 70)
print("Transpose Cost Diagnostic v2")
print("=" * 70)

from sglang.srt.layers.deep_gemm_wrapper.configurer import (
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
)

assert ENABLE_JIT_DEEPGEMM, "DeepGemm not enabled"
print(f"  DEEPGEMM_SCALE_UE8M0 = {DEEPGEMM_SCALE_UE8M0}")
print(f"  DEEPGEMM_BLACKWELL   = {DEEPGEMM_BLACKWELL}")

import deep_gemm
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

block_size = [128, 128]
M = 768  # 256x256 tokens

# ── Step 1: Layout analysis for actual model shapes ───────────────────────
print(f"\n[Step 1: Layout analysis for actual model shapes]")

gemm_shapes = [
    (3840, 3840, "qkvo/ffn_w2"),
    (20480, 3840, "ffn_w1w3"),
    (3840, 10240, "ffn_out"),
]

for N, K, desc in gemm_shapes:
    n_scale = N // 128
    k_scale = K // 128
    has_padding = n_scale % 4 != 0

    A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    A_fp8, As = sglang_per_token_group_quant_fp8(
        A_bf16,
        block_size[1],
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
    )
    B_fp8, Bs = per_block_cast_to_fp8(B_bf16)

    print(f"\n  {desc}: M={M}, N={N}, K={K}")
    print(
        f"    Bs (sfb): shape={Bs.shape} stride={Bs.stride()} "
        f"N/128={n_scale} pad={'YES' if has_padding else 'no'}"
    )
    print(f"    As (sfa): shape={As.shape} stride={As.stride()}")

    # Check: does As match what DeepGemm wants?
    As_aligned = get_mn_major_tma_aligned_tensor(As)
    as_match = As.shape == As_aligned.shape and As.stride() == As_aligned.stride()
    print(
        f"    As vs aligned: shape match={As.shape == As_aligned.shape}, "
        f"stride match={As.stride() == As_aligned.stride()} "
        f"{'✅' if as_match else '❌ DeepGemm will transpose As'}"
    )
    if not as_match:
        print(
            f"      As_aligned: shape={As_aligned.shape} stride={As_aligned.stride()}"
        )

    # For sfb, check_sf_layout accepts row-major or col-major-without-padding.
    # Row-major Bs is contiguous, so it always passes.
    # The question: does DeepGemm transpose sfb or just validate?
    print(
        f"    Bs is row-major contiguous: {Bs.is_contiguous()} "
        f"→ passes check_sf_layout row-major check"
    )

# ── Step 2: CUDA events — actual model shapes ────────────────────────────
print(f"\n[Step 2: CUDA events — per-GEMM overhead for actual shapes]")
print(f"  Since pre-transposed Bs crashes for N/128%4!=0 shapes,")
print(f"  we measure total fp8_gemm_nt time vs pure GEMM kernel time (from nsys).")
print(f"  The difference = transpose + Python dispatch overhead.")
print()

# nsys kernel times for reference (average per call):
nsys_gemm_avg_us = {
    (3840, 3840): 47.8,  # 1200 calls
    (20480, 3840): 226.9,  # 300 calls
    (3840, 10240): 120.2,  # 300 calls
}

start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)
n_iters = 5000

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

    # Warmup
    for _ in range(300):
        deep_gemm.fp8_gemm_nt((A_fp8, As), (B_fp8, Bs), out)
    torch.cuda.synchronize()

    # Measure
    start_ev.record()
    for _ in range(n_iters):
        deep_gemm.fp8_gemm_nt((A_fp8, As), (B_fp8, Bs), out)
    end_ev.record()
    torch.cuda.synchronize()
    t_us = start_ev.elapsed_time(end_ev) / n_iters * 1000  # μs

    nsys_kernel = nsys_gemm_avg_us.get((N, K), 0)
    overhead = t_us - nsys_kernel

    print(f"  {desc} (N={N}, K={K}):")
    print(f"    Measured (CUDA events): {t_us:.1f} μs/call")
    print(f"    nsys GEMM kernel only:  {nsys_kernel:.1f} μs/call")
    print(f"    Overhead (transpose+dispatch): {overhead:.1f} μs/call")

# ── Step 3: Safe shape test — N/128 % 4 == 0 ─────────────────────────────
print(f"\n[Step 3: Pre-transpose Bs test with safe shapes (N/128%4==0)]")
print(f"  Using N=3072 (N/128=24, no padding) to verify pre-transpose works")

N_safe, K_safe = 3072, 3072
A_bf16 = torch.randn(M, K_safe, device="cuda", dtype=torch.bfloat16)
B_bf16 = torch.randn(N_safe, K_safe, device="cuda", dtype=torch.bfloat16)
B_fp8, Bs_safe = per_block_cast_to_fp8(B_bf16)
A_fp8, As_safe = sglang_per_token_group_quant_fp8(
    A_bf16,
    block_size[1],
    column_major_scales=True,
    scale_tma_aligned=True,
    scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
)
out = torch.empty(M, N_safe, device="cuda", dtype=torch.bfloat16)

Bs_safe_pre = get_mn_major_tma_aligned_tensor(Bs_safe)
print(f"  Bs:     shape={Bs_safe.shape} stride={Bs_safe.stride()}")
print(f"  Bs_pre: shape={Bs_safe_pre.shape} stride={Bs_safe_pre.stride()}")

# Warmup both
for _ in range(300):
    deep_gemm.fp8_gemm_nt((A_fp8, As_safe), (B_fp8, Bs_safe), out)
    deep_gemm.fp8_gemm_nt((A_fp8, As_safe), (B_fp8, Bs_safe_pre), out)
torch.cuda.synchronize()

start_ev.record()
for _ in range(n_iters):
    deep_gemm.fp8_gemm_nt((A_fp8, As_safe), (B_fp8, Bs_safe), out)
end_ev.record()
torch.cuda.synchronize()
t_row = start_ev.elapsed_time(end_ev) / n_iters * 1000

start_ev.record()
for _ in range(n_iters):
    deep_gemm.fp8_gemm_nt((A_fp8, As_safe), (B_fp8, Bs_safe_pre), out)
end_ev.record()
torch.cuda.synchronize()
t_pre = start_ev.elapsed_time(end_ev) / n_iters * 1000

sfb_us = t_row - t_pre
print(f"  row-major Bs:      {t_row:.1f} μs/call")
print(f"  pre-transposed Bs: {t_pre:.1f} μs/call")
print(f"  sfb transpose:     {sfb_us:.1f} μs/call")

# ── Step 4: SFA test — does sglang's col-major avoid transpose? ──────────
print(f"\n[Step 4: Does sglang's col-major sfa avoid DeepGemm transpose?]")

# If sglang's As already matches what DeepGemm wants → sfa transpose = 0.
# If not, sfa also contributes to the 139ms.
# We check by comparing: col-major As vs get_mn_major_tma_aligned(As)

N, K = 3840, 3840
A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
B_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
B_fp8, Bs = per_block_cast_to_fp8(B_bf16)

A_fp8_cm, As_cm = sglang_per_token_group_quant_fp8(
    A_bf16,
    block_size[1],
    column_major_scales=True,
    scale_tma_aligned=True,
    scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
)

# What does get_mn_major_tma_aligned_tensor produce for sfa?
As_aligned = get_mn_major_tma_aligned_tensor(As_cm)
sfa_match = As_cm.shape == As_aligned.shape and As_cm.stride() == As_aligned.stride()

print(f"  As_cm (sglang):   shape={As_cm.shape} stride={As_cm.stride()}")
print(f"  As_aligned (DG):  shape={As_aligned.shape} stride={As_aligned.stride()}")
print(
    f"  Match: {sfa_match} {'✅ sfa is NOT transposed' if sfa_match else '❌ sfa IS transposed'}"
)

if not sfa_match:
    # sfa also needs transpose — measure its cost
    print(f"\n  sfa DOES NOT match — measuring sfa transpose cost...")
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    # Use the aligned version (pre-transformed sfa)
    for _ in range(300):
        deep_gemm.fp8_gemm_nt((A_fp8_cm, As_cm), (B_fp8, Bs), out)
        deep_gemm.fp8_gemm_nt((A_fp8_cm, As_aligned), (B_fp8, Bs), out)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(n_iters):
        deep_gemm.fp8_gemm_nt((A_fp8_cm, As_cm), (B_fp8, Bs), out)
    end_ev.record()
    torch.cuda.synchronize()
    t_sfa_orig = start_ev.elapsed_time(end_ev) / n_iters * 1000

    start_ev.record()
    for _ in range(n_iters):
        deep_gemm.fp8_gemm_nt((A_fp8_cm, As_aligned), (B_fp8, Bs), out)
    end_ev.record()
    torch.cuda.synchronize()
    t_sfa_pre = start_ev.elapsed_time(end_ev) / n_iters * 1000

    sfa_us = t_sfa_orig - t_sfa_pre
    print(f"  sglang As:     {t_sfa_orig:.1f} μs/call")
    print(f"  aligned As:    {t_sfa_pre:.1f} μs/call")
    print(f"  sfa transpose: {sfa_us:.1f} μs/call")
else:
    sfa_us = 0.0

# ── Step 5: Transpose count analysis ─────────────────────────────────────
print(f"\n[Step 5: nsys transpose count analysis]")

# From nsys:
# transpose<mn=30>: 32768 calls, mn=30 = N/128 for N=3840
# transpose<mn=80>: 16384 calls, mn=80 = K/128 for K=10240 OR N/128 for N=10240
#
# But wait — for sfb shape (N/128, K/128):
#   N=3840, K=3840 → sfb=(30,30), transpose mn=30
#   N=20480, K=3840 → sfb=(160,30), transpose mn=160 or 30?
#   N=3840, K=10240 → sfb=(30,80), transpose mn=30 or 80?
#
# For sfa shape (M, K/128):
#   K=3840 → sfa=(768,30), transpose mn=768 (but no mn=768 in nsys!)
#   K=10240 → sfa=(768,80), transpose mn=768 (but no mn=768!)
#
# So mn=30 and mn=80 CANNOT be sfa (sfa's mn would be 768).
# They ARE sfb transposes.
#
# mn=30 transposes: sfb for any GEMM with sfb.shape[0]=30, i.e., N=3840
#   GEMMs with N=3840: 1280 (K=3840) + 320 (K=10240) = 1600
#   But 32768 / 1600 = 20.5 transposes per GEMM? → multiple scales per GEMM?
#   OR: 32768 = 1600 GEMMs * 20 denoising steps + warmup?
#   Actually with warmup, nsys captures ALL calls including warmup.
#
# Model has: ~20 denoising steps × 40 layers × 4 GEMMs/layer = 3200 GEMMs
# With 2 shapes of sfb: 3200 * ~10 = 32000? Roughly matches 32768.

print(f"  nsys transpose kernels:")
print(f"    transpose<mn=30>: 32768 calls × 2.4μs = 78.1ms")
print(f"    transpose<mn=80>: 16384 calls × 3.8μs = 61.8ms")
print(f"    Total: 49152 calls = 139.9ms")
print()
print(f"  mn=30 → sfb.shape[0] = N/128 = 30 (N=3840)")
print(f"  mn=80 → sfb.shape[1] = K/128 = 80 (K=10240)")
print(f"    Note: mn=80 is NOT N/128=80 for N=10240, because sfb for N=10240")
print(f"    has shape (160, 30) and mn would be 160, not 80.")
print(f"    Instead, mn=80 is sfb for N=3840, K=10240 → sfb=(30, 80)")
print(f"    The transpose operates on the SECOND dim when it's larger?")
print(f"    → Need to understand DeepGemm's transpose_fp32 template params.")
print()
print(f"  Regardless of which dim, all 49152 transposes are sfb (weight scale),")
print(f"  confirmed by: sfa's mn dim would be 768, which doesn't appear in nsys.")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"Summary")
print(f"{'=' * 70}")
print(f"  1. All 49152 transpose kernels (139.9ms) are sfb (weight scale)")
print(f"     Proof: sfa's M dimension=768 doesn't appear in any transpose template")
print(f"  2. But get_mn_major_tma_aligned_tensor CANNOT be used to pre-transpose")
print(f"     for actual model shapes (N/128 % 4 != 0 → padding → crash)")
print(f"  3. Safe shape test (N=3072): sfb transpose = {sfb_us:.1f} μs/call")
if not sfa_match:
    print(f"  4. sfa also needs transpose: {sfa_us:.1f} μs/call")
else:
    print(f"  4. sfa layout matches DeepGemm (no transpose needed)")
print()
print(f"  Options to eliminate 139.9ms sfb overhead:")
print(f"    A. Simple col-major transpose WITHOUT TMA padding:")
print(f"       Bs.T.contiguous().T → stride=(1, N/128), NO padding")
print(f"       → satisfies check_sf_layout col-major check")
print(f"       → but does DeepGemm's idempotency check detect this?")
print(f"    B. Fork DeepGemm to accept pre-validated sfb")
print(f"    C. CUDA Graph to amortize kernel launch overhead")
print(f"{'=' * 70}")
