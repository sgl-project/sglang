#!/usr/bin/env python3
"""
Diagnostic script for DeepGemm transpose behavior on H20 SM90.

Run this on the GPU server to confirm:
1. UE8M0 configuration (expected False on H20 SM90)
2. get_mn_major_tma_aligned_tensor idempotency + TMA padding behavior
3. SM90 check_sf_layout: does pre-transposed Bs crash for padded shapes?
4. Whether pre-transposing weight scales produces correct GEMM results
5. nn.Parameter attribute survival after .data replacement
6. **KEY**: Where does the 70ms come from — sfb (weight) or sfa (activation)?
7. Benchmark per-GEMM savings from pre-transposing Bs

Usage:
    python zimage_256_256/diagnose_deepgemm_transpose.py
"""

import time

import torch

print("=" * 70)
print("DeepGemm Transpose Diagnostic (v3)")
print("=" * 70)

# ── Step 1: Check DeepGemm configuration ──────────────────────────────────
from sglang.srt.layers.deep_gemm_wrapper.configurer import (
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
)

print(f"\n[Step 1: Config]")
print(f"  ENABLE_JIT_DEEPGEMM  = {ENABLE_JIT_DEEPGEMM}")
print(f"  DEEPGEMM_SCALE_UE8M0 = {DEEPGEMM_SCALE_UE8M0}")
print(f"  DEEPGEMM_BLACKWELL   = {DEEPGEMM_BLACKWELL}")

if not ENABLE_JIT_DEEPGEMM:
    print("\n❌ DeepGemm not enabled. Cannot proceed.")
    exit(1)

import deep_gemm
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

# ── Step 2: TMA padding analysis ─────────────────────────────────────────
print(f"\n[Step 2: TMA padding analysis for check_sf_layout on SM90]")
print(f"  On SM90 with recipe (1,128,128), check_sf_layout for sfb requires:")
print(f"    row-major: stride(-1)==1, stride(-2)==size(-1)")
print(f"    col-major: stride(-1)==size(-2), stride(-2)==1  (NO padding!)")
print()

# Test which N values produce padding
for N in [384, 512, 640, 768, 896, 1024, 1536, 2048, 3072]:
    n_scale = N // 128
    test = torch.randn(n_scale, 24, dtype=torch.float32, device="cuda")
    aligned = get_mn_major_tma_aligned_tensor(test)

    has_padding = aligned.stride(-1) != aligned.size(-2)
    would_crash = has_padding  # SM90 check_sf_layout rejects padded col-major

    status = "❌ PADDING → SM90 crash" if would_crash else "✅ no padding"
    print(
        f"  N={N:5d}  N/128={n_scale:3d}  aligned.stride={aligned.stride()}  "
        f"size(-2)={aligned.size(-2)}  {status}"
    )

# ── Step 3: Idempotency of get_mn_major_tma_aligned_tensor ────────────────
print(f"\n[Step 3: get_mn_major_tma_aligned_tensor idempotency]")
for shape in [(6, 24), (24, 24), (8, 24)]:
    test_scale = torch.randn(*shape, dtype=torch.float32, device="cuda")
    aligned = get_mn_major_tma_aligned_tensor(test_scale)
    try:
        aligned_twice = get_mn_major_tma_aligned_tensor(aligned)
        idempotent_diff = (aligned - aligned_twice).abs().max().item()
        status = "✅" if idempotent_diff < 1e-6 else f"❌ diff={idempotent_diff}"
        print(f"  shape={shape} → {status}")
    except Exception as e:
        print(f"  shape={shape} → ❌ Exception: {e}")

# ── Step 4: Correctness — normal Bs vs pre-transposed Bs ─────────────────
print(f"\n[Step 4: Correctness — normal vs pre-transposed Bs]")

from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

torch.manual_seed(42)
torch.cuda.manual_seed(42)

block_size = [128, 128]
all_pass = True

# Only test shapes where N/128 % 4 == 0 (no padding) — the safe cases
test_shapes = [
    (768, 3072, 3072),  # N/128=24, no padding
    (64, 1024, 1024),  # N/128=8, no padding
    (3072, 2048, 3072),  # N/128=16, no padding
]

for M, N, K in test_shapes:
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

    out_normal = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((A_fp8, As), (B_fp8, Bs), out_normal)

    if not DEEPGEMM_SCALE_UE8M0:
        Bs_pre = get_mn_major_tma_aligned_tensor(Bs)
        out_pre = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt((A_fp8, As), (B_fp8, Bs_pre), out_pre)

        diff = (out_normal - out_pre).abs().max().item()
        passed = diff < 1e-3
        all_pass = all_pass and passed
        status = "✅" if passed else "❌"
        print(
            f"  M={M:4d}, N={N:4d}, K={K:4d}  N/128={N//128:2d}: "
            f"max_diff={diff:.6f} {status}"
        )

# Test a padded shape to see if it crashes
print(f"\n  Testing padded shape (N=768, N/128=6, padding expected):")
try:
    M_p, N_p, K_p = 768, 768, 3072
    A_bf16_p = torch.randn(M_p, K_p, device="cuda", dtype=torch.bfloat16)
    B_bf16_p = torch.randn(N_p, K_p, device="cuda", dtype=torch.bfloat16)
    B_fp8_p, Bs_p = per_block_cast_to_fp8(B_bf16_p)
    A_fp8_p, As_p = sglang_per_token_group_quant_fp8(
        A_bf16_p,
        block_size[1],
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
    )
    Bs_pre_p = get_mn_major_tma_aligned_tensor(Bs_p)
    print(f"    Bs_p shape={Bs_p.shape} stride={Bs_p.stride()}")
    print(f"    Bs_pre_p shape={Bs_pre_p.shape} stride={Bs_pre_p.stride()}")
    out_p = torch.empty(M_p, N_p, device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((A_fp8_p, As_p), (B_fp8_p, Bs_pre_p), out_p)
    print(f"    ✅ No crash with padded pre-transposed Bs (N=768)")
except Exception as e:
    print(f"    ❌ CRASHED with padded pre-transposed Bs: {type(e).__name__}: {e}")
    print(f"    → Confirms SM90 check_sf_layout rejects TMA-padded col-major layout")

if all_pass:
    print(f"\n  ✅ All non-padded shapes pass — pre-transposed Bs is correct!")
else:
    print(f"\n  ❌ SOME SHAPES FAILED")

# ── Step 5: nn.Parameter attribute survival ───────────────────────────────
print(f"\n[Step 5: nn.Parameter._pretransposed_for_deepgemm survival]")

param = torch.nn.Parameter(
    torch.randn(24, 24, dtype=torch.float32, device="cuda"),
    requires_grad=False,
)
param.data = get_mn_major_tma_aligned_tensor(param.data)
param._pretransposed_for_deepgemm = True


def check_attr(t):
    return getattr(t, "_pretransposed_for_deepgemm", False)


print(
    f"  On Parameter object: {check_attr(param)} {'✅' if check_attr(param) else '❌'}"
)
print(f"  On .data (Tensor):   {check_attr(param.data)} (expected False)")

# ── Step 6: KEY DIAGNOSTIC — Where does the transpose cost come from? ─────
print(f"\n[Step 6: KEY — sfb vs sfa transpose cost isolation]")
print(f"  This step isolates whether the runtime transpose overhead is from:")
print(f"    (a) sfb = weight scale (Bs) → pre-transpose helps")
print(f"    (b) sfa = activation scale (As) → pre-transpose Bs is useless")
print()

M, N, K = 768, 3072, 3072
A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
B_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
B_fp8, Bs = per_block_cast_to_fp8(B_bf16)
out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

# Prepare activation scales in different layouts
A_fp8_colmaj, As_colmaj = sglang_per_token_group_quant_fp8(
    A_bf16,
    block_size[1],
    column_major_scales=True,
    scale_tma_aligned=True,
    scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
)
A_fp8_rowmaj, As_rowmaj = sglang_per_token_group_quant_fp8(
    A_bf16,
    block_size[1],
    column_major_scales=False,
    scale_tma_aligned=False,
    scale_ue8m0=False,
)

print(
    f"  As_colmaj: shape={As_colmaj.shape}, stride={As_colmaj.stride()}, "
    f"contiguous={As_colmaj.is_contiguous()}"
)
print(
    f"  As_rowmaj: shape={As_rowmaj.shape}, stride={As_rowmaj.stride()}, "
    f"contiguous={As_rowmaj.is_contiguous()}"
)
print(
    f"  Bs:        shape={Bs.shape}, stride={Bs.stride()}, "
    f"contiguous={Bs.is_contiguous()}"
)

Bs_pre = get_mn_major_tma_aligned_tensor(Bs)
print(
    f"  Bs_pre:    shape={Bs_pre.shape}, stride={Bs_pre.stride()}, "
    f"contiguous={Bs_pre.is_contiguous()}"
)

n_iters = 500

# Warmup all configurations
for _ in range(50):
    deep_gemm.fp8_gemm_nt((A_fp8_colmaj, As_colmaj), (B_fp8, Bs), out)
    deep_gemm.fp8_gemm_nt((A_fp8_colmaj, As_colmaj), (B_fp8, Bs_pre), out)
torch.cuda.synchronize()

# Config A: row-major As + row-major Bs (both need transpose)
print(f"\n  Benchmarking {n_iters} iters each...")
try:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        deep_gemm.fp8_gemm_nt((A_fp8_rowmaj, As_rowmaj), (B_fp8, Bs), out)
    torch.cuda.synchronize()
    t_both_rowmaj = (time.perf_counter() - start) / n_iters * 1000
    print(f"  (a) row-maj As + row-maj Bs:  {t_both_rowmaj:.3f} ms/call")
except Exception as e:
    t_both_rowmaj = None
    print(f"  (a) row-maj As + row-maj Bs:  CRASHED: {e}")

# Config B: col-major As + row-major Bs (only Bs needs transpose)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    deep_gemm.fp8_gemm_nt((A_fp8_colmaj, As_colmaj), (B_fp8, Bs), out)
torch.cuda.synchronize()
t_colAs_rowBs = (time.perf_counter() - start) / n_iters * 1000
print(f"  (b) col-maj As + row-maj Bs:  {t_colAs_rowBs:.3f} ms/call")

# Config C: col-major As + pre-transposed Bs (nothing needs transpose)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    deep_gemm.fp8_gemm_nt((A_fp8_colmaj, As_colmaj), (B_fp8, Bs_pre), out)
torch.cuda.synchronize()
t_colAs_preBs = (time.perf_counter() - start) / n_iters * 1000
print(f"  (c) col-maj As + pre-trans Bs: {t_colAs_preBs:.3f} ms/call")

print(f"\n  Analysis:")
sfb_cost = t_colAs_rowBs - t_colAs_preBs
print(f"    sfb (Bs) transpose cost = (b)-(c) = {sfb_cost:.3f} ms/call")
if t_both_rowmaj is not None:
    sfa_cost = t_both_rowmaj - t_colAs_rowBs
    print(f"    sfa (As) transpose cost = (a)-(b) = {sfa_cost:.3f} ms/call")
    print(
        f"    Total transpose cost    = (a)-(c) = {t_both_rowmaj - t_colAs_preBs:.3f} ms/call"
    )
else:
    print(f"    sfa (As) transpose cost = could not measure (config (a) crashed)")

if sfb_cost > 0.01:
    print(f"\n    → sfb transpose IS significant ({sfb_cost:.3f} ms/call)")
    print(f"      Pre-transposing Bs is beneficial.")
else:
    print(f"\n    → sfb transpose is negligible ({sfb_cost:.3f} ms/call)")
    print(f"      Pre-transposing Bs will NOT help. The 70ms overhead")
    print(f"      likely comes from sfa (activation scale) transpose.")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"Diagnostic Summary")
print(f"{'=' * 70}")
print(
    f"  UE8M0 mode:        {'ON (Blackwell)' if DEEPGEMM_SCALE_UE8M0 else 'OFF (Hopper/SM90)'}"
)
print(f"  Correctness:       {'✅ All passed' if all_pass else '❌ FAILURES DETECTED'}")
print(f"  Attribute alive:   {'✅' if check_attr(param) else '❌'}")
print(
    f"  sfb transpose:     {sfb_cost:.3f} ms/call "
    f"({'significant' if sfb_cost > 0.01 else 'negligible'})"
)
print()
print(f"  NEXT STEPS:")
if sfb_cost > 0.01:
    print(f"    1. Pre-transpose optimization IS effective for Bs.")
    print(f"    2. Run benchmark_pretranspose.sh for E2E validation.")
    print(f"    3. Run nsys to confirm transpose_fp32 kernel reduction.")
else:
    print(f"    1. Pre-transpose Bs is NOT effective.")
    print(f"    2. Investigate sfa (activation scale) optimization instead.")
    print(f"    3. Or investigate if the 70ms comes from a different recipe.")
print(f"{'=' * 70}")
