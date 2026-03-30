"""Benchmark: Triton fused kernel vs flashinfer.sampling.softmax vs PyTorch baseline.

Fair comparison: all variants clone logits each iteration to avoid
measuring on already-softmaxed data.
Uses torch.cuda.Event timing, 200 iterations after 50 warmup.
"""

import argparse
import torch


def benchmark_fn(fn, warmup=50, iters=200):
    """Time a zero-arg callable using CUDA events. Returns microseconds."""
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
    return start.elapsed_time(end) / iters * 1000  # ms -> us


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
        f"{'pytorch(us)':>12}  {'triton(us)':>11}  {'triton_ip(us)':>14}  "
        f"{'flashinfer(us)':>15}  "
        f"{'tri/py':>7}  {'fi/py':>7}  {'tri/fi':>7}"
    )
    print(header)
    print("-" * len(header))

    for bs, vocab, dtype in configs:
        temps = torch.rand(bs, 1, dtype=torch.float32, device="cuda") * 1.5 + 0.1
        temps_flat = temps.view(-1)
        logits_base = torch.randn(bs, vocab, dtype=dtype, device="cuda")

        # --- PyTorch baseline (clone each iter for fairness) ---
        def run_pytorch(logits_base=logits_base, temps=temps):
            l = logits_base.clone()
            l.div_(temps)
            l[:] = torch.softmax(l, dim=-1)

        t_pytorch = benchmark_fn(run_pytorch, args.warmup, args.iters)

        # --- Triton out-of-place (clone each iter) ---
        def run_triton(logits_base=logits_base, temps=temps):
            l = logits_base.clone()
            fused_temperature_softmax(l, temps)

        t_triton = benchmark_fn(run_triton, args.warmup, args.iters)

        # --- Triton in-place (clone each iter) ---
        def run_triton_ip(logits_base=logits_base, temps=temps):
            l = logits_base.clone()
            fused_temperature_softmax_inplace(l, temps)

        t_triton_ip = benchmark_fn(run_triton_ip, args.warmup, args.iters)

        # --- flashinfer (clone each iter) ---
        def run_flashinfer(logits_base=logits_base, temps_flat=temps_flat):
            l = logits_base.clone()
            flashinfer_softmax(l, temperature=temps_flat)

        t_fi = benchmark_fn(run_flashinfer, args.warmup, args.iters)

        speedup_triton = t_pytorch / t_triton
        speedup_fi = t_pytorch / t_fi
        triton_vs_fi = t_fi / t_triton  # >1 means triton faster

        print(
            f"{bs:>5}  {vocab:>7}  {str(dtype):>8}  "
            f"{t_pytorch:>12.1f}  {t_triton:>11.1f}  {t_triton_ip:>14.1f}  "
            f"{t_fi:>15.1f}  "
            f"{speedup_triton:>6.2f}x  {speedup_fi:>6.2f}x  {triton_vs_fi:>6.2f}x"
        )


if __name__ == "__main__":
    main()
