"""Benchmark: fused_temperature_softmax vs separate div_ + softmax vs flashinfer.sampling.softmax.

Each path clones logits every iteration so timing is not skewed by in-place reuse.
Uses torch.cuda.Event timing; default 50 warmup, 200 timed iterations.

Columns tri/base and fi/base are speedup vs PyTorch baseline; tri/fi is t_flashinfer/t_triton
(>1 means Triton is faster).
"""

import argparse

import torch


def benchmark_fn(fn, warmup=50, iters=200):
    """Time a zero-arg callable using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # microseconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    from flashinfer.sampling import softmax as flashinfer_softmax

    from sglang.srt.layers.fused_sampling import (
        fused_temperature_softmax,
        fused_temperature_softmax_inplace,
    )

    configs = [
        # (batch_size, vocab_size, dtype)
        (1, 32000, torch.bfloat16),
        (1, 128256, torch.bfloat16),
        (32, 32000, torch.bfloat16),
        (32, 128256, torch.bfloat16),
        (128, 32000, torch.bfloat16),
        (128, 128256, torch.bfloat16),
        (512, 32000, torch.bfloat16),
        (512, 128256, torch.bfloat16),
    ]

    header = (
        f"{'bs':>5}  {'vocab':>7}  {'dtype':>8}  "
        f"{'baseline (us)':>14}  {'triton (us)':>12}  {'inplace (us)':>13}  {'flashinfer (us)':>16}  "
        f"{'tri/base':>9}  {'fi/base':>8}  {'tri/fi':>7}"
    )
    print(header)
    print("-" * len(header))

    for bs, vocab, dtype in configs:
        temps = torch.rand(bs, 1, dtype=torch.float32, device="cuda") * 1.5 + 0.1
        temps_1d = temps.view(-1)
        logits_src = torch.randn(bs, vocab, dtype=dtype, device="cuda")

        # --- Baseline: div_ + softmax ---
        def run_baseline(src=logits_src, t=temps):
            l = src.clone()
            l.div_(t)
            l[:] = torch.softmax(l, dim=-1)

        t_base = benchmark_fn(run_baseline, args.warmup, args.iters)

        # --- Triton fused (out-of-place) ---
        def run_triton(src=logits_src, t=temps):
            fused_temperature_softmax(src.clone(), t)

        t_triton = benchmark_fn(run_triton, args.warmup, args.iters)

        # --- Triton fused (in-place) ---
        def run_inplace(src=logits_src, t=temps):
            l = src.clone()
            fused_temperature_softmax_inplace(l, t)

        t_ip = benchmark_fn(run_inplace, args.warmup, args.iters)

        # --- FlashInfer (clone each iter, same as other paths) ---
        def run_flashinfer(src=logits_src, t=temps_1d):
            l = src.clone()
            flashinfer_softmax(l, temperature=t)

        t_fi = benchmark_fn(run_flashinfer, args.warmup, args.iters)

        sp_triton = t_base / t_triton
        sp_fi = t_base / t_fi
        tri_vs_fi = t_fi / t_triton
        print(
            f"{bs:>5}  {vocab:>7}  {str(dtype):>8}  "
            f"{t_base:>14.1f}  {t_triton:>12.1f}  {t_ip:>13.1f}  {t_fi:>16.1f}  "
            f"{sp_triton:>8.2f}x  {sp_fi:>7.2f}x  {tri_vs_fi:>6.2f}x"
        )


if __name__ == "__main__":
    main()
