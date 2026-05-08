"""Focused bench at T=8192, hidden=7168 (the production-critical shape).
Compares mhc_pre + sgl_kernel RMSNorm (ref) vs my fused kernel under cuda graphs.
"""
import time, torch
from sglang.srt.layers.layernorm import RMSNorm

T, HIDDEN, HC = 8192, 7168, 4
HC3 = HC * (2 + HC)


def make():
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
    return min(times), sum(times) / len(times)


def main():
    x, fn, sc, bs, rms = make()
    p_r, c_r, y_r = ref(x, fn, sc, bs, rms)
    p_f, c_f, y_f = fused(x, fn, sc, bs, rms)
    err = (y_f.float() - y_r.float()).abs().max().item()
    print(f"correctness max_abs_err: {err:.4e}")

    r_min, r_avg = bench_graph(lambda: ref(x, fn, sc, bs, rms))
    f_min, f_avg = bench_graph(lambda: fused(x, fn, sc, bs, rms))
    print(f"T={T} hidden={HIDDEN}")
    print(f"  ref:   min={r_min:7.2f} us  avg={r_avg:7.2f} us")
    print(f"  fused: min={f_min:7.2f} us  avg={f_avg:7.2f} us")
    print(f"  delta(min): {r_min-f_min:+7.2f} us  ({100*(r_min-f_min)/r_min:+.1f}%)")
    print(f"  delta(avg): {r_avg-f_avg:+7.2f} us  ({100*(r_avg-f_avg)/r_avg:+.1f}%)")


if __name__ == "__main__":
    main()
