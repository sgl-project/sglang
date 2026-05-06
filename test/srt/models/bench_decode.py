"""Decode-batch-size bench (T = 400, 512, 768, 1024) at hidden=7168."""
import time, torch
from sglang.srt.layers.layernorm import RMSNorm

HIDDEN, HC = 7168, 4
HC3 = HC * (2 + HC)


def make(T):
    torch.manual_seed(0)
    x = torch.randn(T, HC, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
    fn = torch.randn(HC3, HC * HIDDEN, dtype=torch.float32, device="cuda") * (1.0 / (HC * HIDDEN) ** 0.5)
    sc = torch.tensor([0.7, 0.5, 0.3], dtype=torch.float32, device="cuda")
    bs = torch.randn(HC3, dtype=torch.float32, device="cuda") * 0.1
    rms = RMSNorm(HIDDEN, eps=1e-6).to("cuda").to(torch.bfloat16)
    rms.weight.data.copy_(torch.randn(HIDDEN, device="cuda") * 0.1 + 1.0)
    return x, fn, sc, bs, rms


def ref(x, fn, sc, bs, rms):
    from sglang.srt.layers.mhc import mhc_pre
    p, c, y = mhc_pre(residual=x, fn=fn, hc_scale=sc, hc_base=bs,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20)
    return p, c, rms(y)


def fused(x, fn, sc, bs, rms):
    from sglang.srt.layers.mhc import mhc_pre
    return mhc_pre(residual=x, fn=fn, hc_scale=sc, hc_base=bs,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0, sinkhorn_repeat=20,
        norm_weight=rms.weight.data, norm_eps=rms.variance_epsilon)


def bench_graph(func, n_iter=300, n_warm=20):
    for _ in range(n_warm):
        func()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        func()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            g.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6 / n_iter)
    return min(times)


def main():
    print(f"{'T':>5} {'ref_us':>8} {'fused_us':>8} {'saved':>8} {'pct':>7}")
    for T in (400, 512, 640, 768, 1024):
        x, fn, sc, bs, rms = make(T)
        # Also: per-forward saving in a 60-layer model with 2 mhc_pre+norm
        # calls per layer (input_layernorm + post_attention_layernorm).
        r = bench_graph(lambda: ref(x, fn, sc, bs, rms))
        f = bench_graph(lambda: fused(x, fn, sc, bs, rms))
        saved = r - f
        pct = 100 * saved / r
        print(f"{T:>5} {r:>8.2f} {f:>8.2f} {saved:>+8.2f} {pct:>+6.1f}%")
    print()
    print("Per-forward decode saving = saved_per_call × 120 (60 layers × 2 calls/layer)")
    print(f"{'T':>5} {'saved/call_us':>14} {'saved/forward_us':>18}")
    for T in (400, 512, 640, 768, 1024):
        x, fn, sc, bs, rms = make(T)
        r = bench_graph(lambda: ref(x, fn, sc, bs, rms))
        f = bench_graph(lambda: fused(x, fn, sc, bs, rms))
        saved = r - f
        print(f"{T:>5} {saved:>14.2f} {saved*120:>18.0f}")


if __name__ == "__main__":
    main()
