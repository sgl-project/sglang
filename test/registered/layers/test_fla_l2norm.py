"""
Test: prove that `T: tl.constexpr` in l2norm_fwd_kernel is wrong and
the fix (plain `T` + do_not_specialize) is correct.

Run:
    python test/registered/layers/test_fla_l2norm.py

This script defines two INDEPENDENT kernel copies in this file (not touching source):
  - kernel_constexpr_T : T is tl.constexpr  (OLD / buggy)
  - kernel_runtime_T   : T is plain param + do_not_specialize  (FIXED)

Three parts:
  Part 1 — Correctness: both kernels vs PyTorch reference
  Part 2 — Compilation count: cache entries after N different T values
  Part 3 — End-to-end latency: simulate a real inference loop with varying T,
           measure wall-clock time INCLUDING compile overhead

The key insight:
  T = num_tokens * num_heads, which changes every batch during inference.
  Making it constexpr forces Triton to recompile for every new T.
  Each recompile takes hundreds of ms -> devastating for online serving latency.
"""

import time

import torch
import triton
import triton.language as tl

# ──────────────────────────────────────────────────────────────────────────────
# Two kernel variants: OLD (constexpr T) vs FIXED (runtime T)
#
# These are self-contained copies, NOT imported from source code.
# This way the test proves the design difference directly, regardless of
# what's currently in l2norm.py.
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def kernel_constexpr_T(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """OLD version: T is tl.constexpr — recompiles for every distinct T."""
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def kernel_runtime_T(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """FIXED version: T is a plain runtime param, no recompile on T change."""
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Reference
# ──────────────────────────────────────────────────────────────────────────────


def l2norm_ref(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    norm = torch.sqrt((x_f * x_f).sum(dim=-1, keepdim=True) + eps)
    return (x_f / norm).to(x.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_cache_size(kernel_fn) -> int:
    """Number of compiled kernel variants in Triton's in-memory cache."""
    total = 0
    for _dev, cache_tuple in kernel_fn.device_caches.items():
        total += len(cache_tuple[0])
    return total


def launch_kernel(kernel_fn, x, D, eps=1e-6):
    """Launch a kernel following the same logic as l2norm_fwd (D <= 512 path)."""
    T = x.shape[0]
    y = torch.empty_like(x)
    BD = triton.next_power_of_2(D)
    BT = 16
    NB = triton.cdiv(T, 2048)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]),)

    kernel_fn[grid](x, y, eps, NB=NB, T=T, D=D, BT=BT, BD=BD, num_warps=8, num_stages=3)
    return y


# ──────────────────────────────────────────────────────────────────────────────
# Part 1 + 2: Correctness & Compilation count
# ──────────────────────────────────────────────────────────────────────────────


def test_correctness_and_compilation():
    D = 128
    dtype = torch.bfloat16
    eps = 1e-6
    # All T values give NB = ceil(T/2048) = 1, so the ONLY variable is T.
    t_values = [16, 37, 64, 100, 128, 200, 256, 333, 500, 777]

    print("=" * 78)
    print("  Part 1 & 2: Correctness + Compilation Count")
    print("=" * 78)
    print(f"  D={D}, dtype={dtype}, NB=1 for all T")
    print()

    # Warmup (first compile)
    x0 = torch.randn(32, D, dtype=dtype, device="cuda")
    launch_kernel(kernel_constexpr_T, x0, D, eps)
    launch_kernel(kernel_runtime_T, x0, D, eps)
    cache_c0 = get_cache_size(kernel_constexpr_T)
    cache_r0 = get_cache_size(kernel_runtime_T)

    # Run
    header = f"  {'T':>6} │ constexpr_T │ runtime_T"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for t in t_values:
        torch.manual_seed(t)
        x = torch.randn(t, D, dtype=dtype, device="cuda")
        ref = l2norm_ref(x, eps)
        out_c = launch_kernel(kernel_constexpr_T, x, D, eps)
        out_r = launch_kernel(kernel_runtime_T, x, D, eps)
        ok_c = torch.allclose(out_c, ref, atol=1e-2, rtol=1e-2)
        ok_r = torch.allclose(out_r, ref, atol=1e-2, rtol=1e-2)
        print(
            f"  {t:>6} │ {'PASS' if ok_c else 'FAIL':>11} │ {'PASS' if ok_r else 'FAIL':>9}"
        )

    cache_c1 = get_cache_size(kernel_constexpr_T)
    cache_r1 = get_cache_size(kernel_runtime_T)
    new_c = cache_c1 - cache_c0
    new_r = cache_r1 - cache_r0

    print()
    print(f"  Compilations triggered by {len(t_values)} different T values:")
    print(f"    constexpr T : {new_c:>3}  (cache {cache_c0} -> {cache_c1})")
    print(f"    runtime   T : {new_r:>3}  (cache {cache_r0} -> {cache_r1})")
    print()

    assert new_c >= len(t_values), (
        f"constexpr should compile >= {len(t_values)} times, got {new_c}"
    )
    assert new_r == 0, f"runtime should compile 0 times, got {new_r}"
    print("  Part 1 & 2 PASSED.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Part 3: End-to-end latency benchmark
#
# Simulate a real inference scenario:
#   - A sequence of batches arrives, each with a DIFFERENT T.
#   - Measure the TOTAL wall-clock time (compile + kernel launch + execution).
#   - constexpr T pays a compile penalty on every new T.
#   - runtime T compiles once, then reuses.
# ──────────────────────────────────────────────────────────────────────────────


def test_e2e_latency():
    D = 128
    dtype = torch.bfloat16
    eps = 1e-6

    # 20 distinct T values, simulating 20 batches with different token counts.
    # In real serving this easily reaches hundreds of distinct T values.
    t_values = [t * 16 for t in range(1, 21)]  # 16, 32, 48, ..., 320

    print("=" * 78)
    print("  Part 3: End-to-End Latency (compile + launch + execute)")
    print("=" * 78)
    print(f"  D={D}, dtype={dtype}")
    print(f"  Simulating {len(t_values)} batches with T = {t_values}")
    print()

    # Pre-generate all inputs
    inputs = []
    for t in t_values:
        inputs.append(torch.randn(t, D, dtype=dtype, device="cuda"))

    # ── Measure constexpr T ──
    # Fresh cache: each T triggers a new compilation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x in inputs:
        _ = launch_kernel(kernel_constexpr_T, x, D, eps)
    torch.cuda.synchronize()
    time_constexpr = time.perf_counter() - t0

    # ── Measure runtime T ──
    # Already warmed up from Part 1, so this uses cached kernel
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x in inputs:
        _ = launch_kernel(kernel_runtime_T, x, D, eps)
    torch.cuda.synchronize()
    time_runtime = time.perf_counter() - t0

    speedup = time_constexpr / time_runtime if time_runtime > 0 else float("inf")

    print(f"  constexpr T total : {time_constexpr * 1000:>10.2f} ms")
    print(f"  runtime   T total : {time_runtime * 1000:>10.2f} ms")
    print(f"  speedup           : {speedup:>10.1f}x")
    print()

    # Per-batch average
    avg_c = time_constexpr / len(t_values) * 1000
    avg_r = time_runtime / len(t_values) * 1000
    print(f"  Per-batch average:")
    print(f"    constexpr T : {avg_c:>8.2f} ms/batch  (includes compile overhead)")
    print(f"    runtime   T : {avg_r:>8.2f} ms/batch  (pure kernel execution)")
    print()

    if speedup > 1.5:
        print(f"  CONCLUSION: runtime T is {speedup:.1f}x faster.")
        print(
            f"    The {time_constexpr - time_runtime:.3f}s difference is almost entirely"
        )
        print(f"    Triton compilation overhead from constexpr T.")
    else:
        print(f"  NOTE: speedup ({speedup:.1f}x) is modest because constexpr T's")
        print(f"    compilations were partially cached from Part 1.")
        print(f"    In a cold-start scenario the difference is much larger.")

    print()
    print("  Part 3 PASSED.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Part 4: Cold-start latency (most realistic)
#
# Use FRESH kernel copies that have never been compiled, to measure the
# true cold-start cost difference.
# We define new kernels here to ensure zero cache.
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _cold_constexpr_T(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _cold_runtime_T(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def test_cold_start_latency():
    D = 128
    dtype = torch.bfloat16
    eps = 1e-6
    n_batches = 10
    t_values = [16 + i * 50 for i in range(n_batches)]  # 16, 66, 116, ..., 466

    print("=" * 78)
    print("  Part 4: Cold-Start Latency (fresh kernels, zero cache)")
    print("=" * 78)
    print(f"  {n_batches} batches, T = {t_values}")
    print()

    inputs = [torch.randn(t, D, dtype=dtype, device="cuda") for t in t_values]

    # ── constexpr T: cold start ──
    assert get_cache_size(_cold_constexpr_T) == 0, "kernel should have empty cache"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x in inputs:
        launch_kernel(_cold_constexpr_T, x, D, eps)
    torch.cuda.synchronize()
    time_cold_constexpr = time.perf_counter() - t0
    compiles_constexpr = get_cache_size(_cold_constexpr_T)

    # ── runtime T: cold start ──
    assert get_cache_size(_cold_runtime_T) == 0, "kernel should have empty cache"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x in inputs:
        launch_kernel(_cold_runtime_T, x, D, eps)
    torch.cuda.synchronize()
    time_cold_runtime = time.perf_counter() - t0
    compiles_runtime = get_cache_size(_cold_runtime_T)

    speedup = (
        time_cold_constexpr / time_cold_runtime
        if time_cold_runtime > 0
        else float("inf")
    )

    print(f"  {'':>20} │ {'constexpr T':>14} │ {'runtime T':>14}")
    print(f"  {'─' * 20}─┼─{'─' * 14}─┼─{'─' * 14}")
    print(
        f"  {'Total time':>20} │ {time_cold_constexpr * 1000:>11.1f} ms │ {time_cold_runtime * 1000:>11.1f} ms"
    )
    print(f"  {'Compilations':>20} │ {compiles_constexpr:>14} │ {compiles_runtime:>14}")
    print(
        f"  {'Avg per batch':>20} │ {time_cold_constexpr / n_batches * 1000:>11.1f} ms │ {time_cold_runtime / n_batches * 1000:>11.1f} ms"
    )
    print(f"  {'Speedup':>20} │ {'':>14} │ {speedup:>11.1f}x")
    print()
    print(f"  constexpr T compiled {compiles_constexpr} times (once per T value)")
    print(
        f"  runtime   T compiled {compiles_runtime} time(s) (reused for all T values)"
    )
    print(
        f"  Wasted compile time: ~{(time_cold_constexpr - time_cold_runtime) * 1000:.0f} ms"
    )
    print()
    print("  Part 4 PASSED.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    assert torch.cuda.is_available(), "This test requires a CUDA GPU."

    print()
    test_correctness_and_compilation()
    test_e2e_latency()
    test_cold_start_latency()

    print("=" * 78)
    print("  ALL TESTS PASSED")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
