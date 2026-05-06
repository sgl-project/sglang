"""Production-realistic benchmark for hc_head and mhc_pre+RMSNorm fusions.

Both reference and fused paths are captured in CUDA graphs and replay-timed.
The decode hot path runs under cuda graphs (cuda-graph-max-bs=1024 in the
production yaml), so launch overhead doesn't exist in production. My earlier
microbenchmarks compared eager-torch reference launches against a single
triton/tilelang kernel call and counted launch overhead as part of the win,
which doesn't translate to E2E.

This bench captures each path in a graph and times the replay, which is
what actually runs in production.
"""

from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from sglang.srt.layers.layernorm import RMSNorm


def bench_graph(func, n_iter=200, n_warm=10):
    for _ in range(n_warm):
        func()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        func()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n_iter  # microseconds


# ============================== hc_head ==============================

def _hc_head_inputs(T, hc_mult, hidden, device="cuda"):
    torch.manual_seed(0)
    x = torch.randn(T, hc_mult, hidden, dtype=torch.bfloat16, device=device) * 0.5
    fn = torch.randn(hc_mult, hc_mult * hidden, dtype=torch.float32, device=device) * (
        1.0 / (hc_mult * hidden) ** 0.5
    )
    scale = torch.tensor([0.7], dtype=torch.float32, device=device)
    base = torch.randn(hc_mult, dtype=torch.float32, device=device) * 0.1
    return x, fn, scale, base


def hc_head_eager(x, fn, scale, base, norm_eps=1e-6, hc_eps=1e-6):
    """Original eager-torch impl."""
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def hc_head_fused(x, fn, scale, base, norm_eps=1e-6, hc_eps=1e-6):
    from sglang.srt.layers.mhc_head import fused_hc_head
    return fused_hc_head(x.contiguous(), fn, scale, base, norm_eps, hc_eps)


# ============================== mhc_pre + RMSNorm ==============================

def _mhc_inputs(T, hc_mult, hidden, device="cuda"):
    torch.manual_seed(0)
    hc_dim = hc_mult * hidden
    hc_mult3 = hc_mult * (2 + hc_mult)
    x = torch.randn(T, hc_mult, hidden, dtype=torch.bfloat16, device=device) * 0.5
    fn = torch.randn(hc_mult3, hc_dim, dtype=torch.float32, device=device) * (1.0 / hc_dim**0.5)
    hc_scale = torch.tensor([0.7, 0.5, 0.3], dtype=torch.float32, device=device)
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device=device) * 0.1
    rms = RMSNorm(hidden, eps=1e-6).to(device).to(torch.bfloat16)
    rms.weight.data.copy_(torch.randn(hidden, device=device) * 0.1 + 1.0)
    return x, fn, hc_scale, hc_base, rms


def mhc_ref(x, fn, hc_scale, hc_base, rms):
    """Production reference: TileLang mhc_pre + sgl_kernel RMSNorm."""
    from sglang.srt.layers.mhc import mhc_pre
    post, comb, y = mhc_pre(
        residual=x, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20,
    )
    y = rms(y)
    return post, comb, y


def mhc_fused(x, fn, hc_scale, hc_base, rms):
    from sglang.srt.layers.mhc import mhc_pre
    return mhc_pre(
        residual=x, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20,
        norm_weight=rms.weight.data, norm_eps=rms.variance_epsilon,
    )


# ============================== drivers ==============================

def run_hc_head():
    print("\n========== hc_head (cuda graph) ==========")
    print(f"{'T':>6} {'hidden':>6} {'eager_us':>10} {'fused_us':>10} {'delta':>10}")
    for hidden in (4096, 7168):
        for T in (1, 32, 256, 1024, 8192, 32768):
            x, fn, scale, base = _hc_head_inputs(T, 4, hidden)
            t_eager = bench_graph(lambda: hc_head_eager(x, fn, scale, base))
            t_fused = bench_graph(lambda: hc_head_fused(x, fn, scale, base))
            delta = t_eager - t_fused
            pct = 100 * delta / t_eager if t_eager > 0 else 0
            print(f"{T:>6} {hidden:>6} {t_eager:>10.2f} {t_fused:>10.2f} {delta:+9.2f}us ({pct:+5.1f}%)")


def run_mhc():
    print("\n========== mhc_pre + RMSNorm (cuda graph) ==========")
    print(f"{'T':>6} {'hidden':>6} {'ref_us':>10} {'fused_us':>10} {'delta':>10}")
    for hidden in (4096, 7168):
        for T in (1, 32, 256, 1024, 8192, 32768):
            x, fn, hc_scale, hc_base, rms = _mhc_inputs(T, 4, hidden)
            t_ref = bench_graph(lambda: mhc_ref(x, fn, hc_scale, hc_base, rms))
            t_fused = bench_graph(lambda: mhc_fused(x, fn, hc_scale, hc_base, rms))
            delta = t_ref - t_fused
            pct = 100 * delta / t_ref if t_ref > 0 else 0
            print(f"{T:>6} {hidden:>6} {t_ref:>10.2f} {t_fused:>10.2f} {delta:+9.2f}us ({pct:+5.1f}%)")


def main():
    print("Production-realistic bench: both paths captured in cuda graphs")
    print("(decode hot path runs under cuda-graph-max-bs in the production yaml)")
    run_hc_head()
    run_mhc()


if __name__ == "__main__":
    main()
