"""Realistic microbench: compares
   ref:    mhc_pre()  →  RMSNorm.forward_cuda()  (the actual production path)
   fused:  mhc_pre(norm_weight=, norm_eps=)
across DSV4-Pro shapes.
"""

from __future__ import annotations
import time
import torch
from sglang.srt.layers.layernorm import RMSNorm


def _make_inputs(T, hc_mult, hidden_size, device="cuda"):
    torch.manual_seed(0)
    hc_dim = hc_mult * hidden_size
    hc_mult3 = hc_mult * (2 + hc_mult)
    x = torch.randn(T, hc_mult, hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    fn = torch.randn(hc_mult3, hc_dim, dtype=torch.float32, device=device) * (1.0 / hc_dim**0.5)
    hc_scale = torch.tensor([0.7, 0.5, 0.3], dtype=torch.float32, device=device)
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device=device) * 0.1
    rms = RMSNorm(hidden_size, eps=1e-6).to(device).to(torch.bfloat16)
    rms.weight.data.copy_(torch.randn(hidden_size, device=device) * 0.1 + 1.0)
    return x, fn, hc_scale, hc_base, rms


def reference(x, fn, hc_scale, hc_base, rms):
    from sglang.srt.layers.mhc import mhc_pre
    post, comb, y = mhc_pre(
        residual=x, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20,
    )
    y = rms(y)
    return post, comb, y


def fused(x, fn, hc_scale, hc_base, rms):
    from sglang.srt.layers.mhc import mhc_pre
    return mhc_pre(
        residual=x, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20,
        norm_weight=rms.weight.data,
        norm_eps=rms.variance_epsilon,
    )


def mhc_pre_only(x, fn, hc_scale, hc_base):
    """Just mhc_pre — to measure the baseline kernel without any norm."""
    from sglang.srt.layers.mhc import mhc_pre
    return mhc_pre(
        residual=x, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20,
    )


def run(T, hc_mult, hidden_size):
    x, fn, hc_scale, hc_base, rms = _make_inputs(T, hc_mult, hidden_size)

    post_r, comb_r, y_r = reference(x, fn, hc_scale, hc_base, rms)
    post_f, comb_f, y_f = fused(x, fn, hc_scale, hc_base, rms)
    err = (y_f.float() - y_r.float()).abs().max().item()
    print(f"  T={T:>5d} hidden={hidden_size}  max_abs_err={err:.4e}")

    n_warm = 10
    n_iter = 50
    for _ in range(n_warm):
        reference(x, fn, hc_scale, hc_base, rms)
        fused(x, fn, hc_scale, hc_base, rms)
        mhc_pre_only(x, fn, hc_scale, hc_base)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        mhc_pre_only(x, fn, hc_scale, hc_base)
    torch.cuda.synchronize()
    mhc_only_ms = (time.perf_counter() - t0) * 1000 / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        reference(x, fn, hc_scale, hc_base, rms)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) * 1000 / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        fused(x, fn, hc_scale, hc_base, rms)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) * 1000 / n_iter

    rms_only_ms = ref_ms - mhc_only_ms
    saved = ref_ms - fused_ms
    pct = 100 * saved / ref_ms if ref_ms > 0 else 0
    print(
        f"      mhc_pre_only={mhc_only_ms:6.3f} ms  "
        f"sgl_RMSNorm≈{rms_only_ms:6.3f} ms  "
        f"ref(mhc+rms)={ref_ms:6.3f} ms  "
        f"fused={fused_ms:6.3f} ms  saved={saved:+.3f} ({pct:+.1f}%)"
    )


def main():
    print("Realistic ref = mhc_pre + sgl_kernel RMSNorm")
    print("=" * 78)
    for hidden in (4096, 7168):
        print(f"\n--- hidden={hidden} ---")
        for T in (1, 256, 1024, 8192, 32768):
            run(T, 4, hidden)


if __name__ == "__main__":
    main()
