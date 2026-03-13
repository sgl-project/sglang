"""Benchmark: fused_temperature_softmax vs separate div_ + softmax.

Measures wall-clock time with torch.cuda.Event timing, 200 iterations
after 50 warmup. Reports per-call latency and speedup.
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


def reference_temperature_softmax(logits, temperatures):
    """Original two-kernel path."""
    logits.div_(temperatures)
    logits[:] = torch.softmax(logits, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

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

    print(f"{'bs':>5}  {'vocab':>7}  {'dtype':>8}  {'original (us)':>14}  "
          f"{'fused (us)':>11}  {'inplace (us)':>13}  {'speedup':>8}  {'speedup_ip':>11}")
    print("-" * 100)

    for bs, vocab, dtype in configs:
        temps = (torch.rand(bs, 1, dtype=torch.float32, device="cuda") * 1.5 + 0.1)

        # --- Original ---
        logits_orig = torch.randn(bs, vocab, dtype=dtype, device="cuda")

        def run_original():
            l = logits_orig.clone()
            l.div_(temps)
            l[:] = torch.softmax(l, dim=-1)

        t_orig = benchmark_fn(run_original, args.warmup, args.iters)

        # --- Fused (out-of-place) ---
        logits_fused = torch.randn(bs, vocab, dtype=dtype, device="cuda")

        def run_fused():
            fused_temperature_softmax(logits_fused, temps)

        t_fused = benchmark_fn(run_fused, args.warmup, args.iters)

        # --- Fused (in-place) ---
        logits_ip = torch.randn(bs, vocab, dtype=dtype, device="cuda")

        def run_inplace():
            fused_temperature_softmax_inplace(logits_ip, temps)

        t_ip = benchmark_fn(run_inplace, args.warmup, args.iters)

        speedup = t_orig / t_fused
        speedup_ip = t_orig / t_ip
        print(
            f"{bs:>5}  {vocab:>7}  {str(dtype):>8}  {t_orig:>14.1f}  "
            f"{t_fused:>11.1f}  {t_ip:>13.1f}  {speedup:>7.2f}x  {speedup_ip:>10.2f}x"
        )


if __name__ == "__main__":
    main()
