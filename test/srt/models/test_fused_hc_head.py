"""Standalone correctness + microbenchmark for the fused hc_head triton kernel.

Run on a single GPU. Compares the fused triton path against the eager-torch
reference from DeepseekV4Model.hc_head and reports max abs/rel error plus
microbenchmark timings across the shapes that DSV4-Pro prefill actually uses.

Usage (on the cluster, from inside the container or with cuda available):
    python test/srt/models/test_fused_hc_head.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def hc_head_reference(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    """Byte-for-byte the original DeepseekV4Model.hc_head."""
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def run_one(T: int, hc_mult: int, hidden_size: int, dtype=torch.bfloat16):
    from sglang.srt.layers.mhc_head import fused_hc_head

    device = "cuda"
    torch.manual_seed(0)

    x = torch.randn(T, hc_mult, hidden_size, dtype=dtype, device=device) * 0.5
    hc_fn = torch.randn(hc_mult, hc_mult * hidden_size, dtype=torch.float32, device=device) * (1.0 / (hc_mult * hidden_size) ** 0.5)
    hc_scale = torch.tensor([0.7], dtype=torch.float32, device=device)
    hc_base = torch.randn(hc_mult, dtype=torch.float32, device=device) * 0.1
    norm_eps = 1e-6
    hc_eps = 1e-6

    y_ref = hc_head_reference(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)
    y_fused = fused_hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)

    abs_err = (y_fused.float() - y_ref.float()).abs()
    rel_err = abs_err / (y_ref.float().abs() + 1e-6)
    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()
    print(
        f"  shape T={T:>5d} hc_mult={hc_mult} hidden={hidden_size}  "
        f"max_abs={max_abs:.4e}  max_rel={max_rel:.4e}"
    )

    # bf16 has ~7-bit mantissa; 1 ULP near magnitude 1 is ~7.8e-3. Allow 2 ULPs
    # to absorb summation-order differences between torch and triton tree reductions.
    abs_tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    rel_tol = 1e-1 if dtype == torch.bfloat16 else 1e-5
    # Both must fail to call it broken; small outputs make rel error noisy by itself.
    if max_abs >= abs_tol and max_rel >= rel_tol:
        raise AssertionError(
            f"max_abs {max_abs:.4e} >= abs_tol {abs_tol:.4e} AND "
            f"max_rel {max_rel:.4e} >= rel_tol {rel_tol:.4e} for shape "
            f"T={T} hc_mult={hc_mult} hidden={hidden_size}"
        )

    n_warmup = 5
    n_iter = 50
    for _ in range(n_warmup):
        hc_head_reference(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)
        fused_hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        hc_head_reference(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) * 1000 / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        fused_hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) * 1000 / n_iter

    speedup = ref_ms / fused_ms
    print(
        f"           ref={ref_ms:6.3f} ms  fused={fused_ms:6.3f} ms  "
        f"speedup={speedup:5.2f}x"
    )


def main():
    print("=" * 70)
    print("Fused hc_head correctness + microbench  (DSV4-Pro shapes)")
    print("=" * 70)
    hc_mult = 4
    for hidden_size in (4096, 7168):
        for T in (1, 256, 1024, 8192, 32768):
            run_one(T, hc_mult, hidden_size)
        print()
    print("All shapes passed correctness.")


if __name__ == "__main__":
    main()
