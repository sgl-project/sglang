"""Before/after benchmark for the DeepSeek-V4 mHC `hc_split_sinkhorn` fallback.

On backends without TileLang (e.g. ROCm/AMD), `hc_split_sinkhorn` has no tuned
CUDA path, so the alternative is a vectorized-torch implementation. This script
compares that torch baseline (BEFORE) against the fused Triton fallback (AFTER)
that ships in sglang.srt.layers.mhc.

Run:
    python3 benchmark/kernels/bench_hc_split_sinkhorn.py
"""

import argparse
import statistics

import torch

from sglang.srt.layers.mhc import _hc_split_sinkhorn_triton  # AFTER (fused Triton)

HC, ITERS, EPS = 4, 20, 1e-6  # DeepSeek-V4: hc_mult=4, 20 Sinkhorn iterations


def sinkhorn_torch_baseline(mixes, hc_scale, hc_base, hc, iters, eps):
    """BEFORE: vectorized-torch fallback (the alternative when TileLang is absent).

    mixes: [b, s, (2 + hc) * hc]  ->  pre/post: [b, s, hc], comb: [b, s, hc, hc]
    """
    b, s, _ = mixes.shape
    m = mixes.reshape(b * s, -1)
    pre = torch.sigmoid(m[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(-1, hc, hc)

    row_max = comb.max(dim=2, keepdim=True).values
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=2, keepdim=True) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=2, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    return (
        pre.view(b, s, hc),
        post.view(b, s, hc),
        comb.view(b, s, hc, hc),
    )


def time_ms(fn, trials=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(trials):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        fn()
        en.record()
        torch.cuda.synchronize()
        out.append(st.elapsed_time(en))
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify before==after first")
    args = ap.parse_args()

    dev = "cuda"
    mix_hc = (2 + HC) * HC
    print(f"device: {torch.cuda.get_device_name(0)}")

    if args.check:
        torch.manual_seed(0)
        mix = torch.randn(1, 4096, mix_hc, device=dev)
        sc = torch.rand(3, device=dev) + 0.5
        ba = torch.randn(mix_hc, device=dev)
        a = sinkhorn_torch_baseline(mix, sc, ba, HC, ITERS, EPS)
        b = _hc_split_sinkhorn_triton(mix, sc, ba, HC, ITERS, EPS)
        md = max((a[i] - b[i]).abs().max().item() for i in range(3))
        print(f"max|before-after| = {md:.2e}  ({'OK' if md < 1e-4 else 'MISMATCH'})\n")

    print(f"{'tokens':>8} | {'BEFORE torch (ms)':>18} | {'AFTER triton (ms)':>18} | {'speedup':>8}")
    print("-" * 62)
    for n in [1, 8, 32, 128, 512, 2048, 8192, 32768]:
        torch.manual_seed(0)
        mix = torch.randn(1, n, mix_hc, device=dev)
        sc = torch.rand(3, device=dev) + 0.5
        ba = torch.randn(mix_hc, device=dev)
        t_before = time_ms(lambda: sinkhorn_torch_baseline(mix, sc, ba, HC, ITERS, EPS))
        t_after = time_ms(lambda: _hc_split_sinkhorn_triton(mix, sc, ba, HC, ITERS, EPS))
        print(f"{n:>8} | {t_before:>18.4f} | {t_after:>18.4f} | {t_before / t_after:>7.2f}x")


if __name__ == "__main__":
    main()
