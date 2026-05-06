"""Correctness + microbench for the fused mhc_pre + RMSNorm TileLang kernel.

Compares:
  reference path:  mhc_pre()  →  RMSNorm()
  fused path:      mhc_pre(norm_weight=, norm_eps=)

Across DSV4-Pro shapes:
  hc_mult=4
  hidden_size in {4096, 7168}
  T in {1, 256, 1024, 8192, 32768}

Tolerance: bf16 outputs are compared with abs<2e-2 (≈2 ULPs near magnitude 1)
OR rel<10%. Both must fail to count as broken — small magnitudes have noisy rel
errors by themselves.

Usage on the cluster:
  python /workspace/sglang/test/srt/models/test_fused_mhc_pre_norm.py
"""

from __future__ import annotations

import time

import torch


def _make_inputs(T: int, hc_mult: int, hidden_size: int, device="cuda"):
    torch.manual_seed(0)
    hc_dim = hc_mult * hidden_size
    hc_mult3 = hc_mult * (2 + hc_mult)
    x = torch.randn(T, hc_mult, hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    fn = torch.randn(hc_mult3, hc_dim, dtype=torch.float32, device=device) * (
        1.0 / hc_dim**0.5
    )
    hc_scale = torch.tensor([0.7, 0.5, 0.3], dtype=torch.float32, device=device)
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device=device) * 0.1
    norm_weight = (
        torch.randn(hidden_size, dtype=torch.float32, device=device) * 0.1 + 1.0
    )
    return x, fn, hc_scale, hc_base, norm_weight


def reference(x, fn, hc_scale, hc_base, norm_weight, rms_eps, hc_eps, norm_eps):
    """mhc_pre then RMSNorm — separate kernels."""
    from sglang.srt.layers.mhc import mhc_pre

    post, comb, y = mhc_pre(
        residual=x,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    )
    # y is (T, hidden_size) bf16. Apply RMSNorm:
    y_f = y.float()
    var = y_f.square().mean(dim=-1, keepdim=True)
    y_normed = y_f * torch.rsqrt(var + norm_eps) * norm_weight
    return post, comb, y_normed.to(y.dtype)


def fused(x, fn, hc_scale, hc_base, norm_weight, rms_eps, hc_eps, norm_eps):
    from sglang.srt.layers.mhc import mhc_pre

    post, comb, y = mhc_pre(
        residual=x,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return post, comb, y


def run_one(T: int, hc_mult: int, hidden_size: int):
    x, fn, hc_scale, hc_base, norm_weight = _make_inputs(T, hc_mult, hidden_size)
    rms_eps = 1e-6
    hc_eps = 1e-6
    norm_eps = 1e-6

    post_r, comb_r, y_r = reference(
        x, fn, hc_scale, hc_base, norm_weight, rms_eps, hc_eps, norm_eps
    )
    post_f, comb_f, y_f = fused(
        x, fn, hc_scale, hc_base, norm_weight, rms_eps, hc_eps, norm_eps
    )

    # post and comb come from the same code path on both sides — should be byte-identical.
    assert torch.equal(post_r, post_f), "post_mix divergence!"
    assert torch.equal(comb_r, comb_f), "comb_mix divergence!"

    abs_err = (y_f.float() - y_r.float()).abs()
    rel_err = abs_err / (y_r.float().abs() + 1e-6)
    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()
    print(
        f"  T={T:>5d}  hc_mult={hc_mult}  hidden={hidden_size}  "
        f"max_abs={max_abs:.4e}  max_rel={max_rel:.4e}"
    )

    abs_tol = 2e-2  # ~2 ULPs of bf16 near magnitude 1
    rel_tol = 1e-1
    if max_abs >= abs_tol and max_rel >= rel_tol:
        raise AssertionError(
            f"max_abs {max_abs:.4e} AND max_rel {max_rel:.4e} both above tolerance "
            f"({abs_tol}, {rel_tol}) for shape T={T} hidden={hidden_size}"
        )

    # Microbench
    n_warmup = 5
    n_iter = 30
    for _ in range(n_warmup):
        reference(x, fn, hc_scale, hc_base, norm_weight, rms_eps, hc_eps, norm_eps)
        fused(x, fn, hc_scale, hc_base, norm_weight, rms_eps, hc_eps, norm_eps)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        reference(x, fn, hc_scale, hc_base, norm_weight, rms_eps, hc_eps, norm_eps)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) * 1000 / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        fused(x, fn, hc_scale, hc_base, norm_weight, rms_eps, hc_eps, norm_eps)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) * 1000 / n_iter

    speedup = ref_ms / fused_ms
    print(
        f"           ref={ref_ms:7.3f} ms  fused={fused_ms:7.3f} ms  "
        f"speedup={speedup:5.2f}x  saved={(ref_ms-fused_ms):+.3f} ms"
    )


def main():
    print("=" * 78)
    print("Fused mhc_pre + RMSNorm correctness + microbench (DSV4-Pro shapes)")
    print("=" * 78)
    hc_mult = 4
    for hidden_size in (4096, 7168):
        print(f"\n--- hidden_size={hidden_size} ---")
        for T in (1, 256, 1024, 8192, 32768):
            run_one(T, hc_mult, hidden_size)
    print("\nAll shapes passed correctness.")


if __name__ == "__main__":
    main()
