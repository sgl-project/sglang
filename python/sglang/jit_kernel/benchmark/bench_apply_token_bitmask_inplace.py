"""Benchmark: JIT CUDA vs Triton apply_token_bitmask_inplace.

Measures latency across representative vocab sizes and batch sizes
for grammar-constrained decoding workloads.
"""

import itertools

import pandas as pd
import torch

from sglang.jit_kernel.apply_token_bitmask_inplace import (
    apply_token_bitmask_inplace_jit,
)
from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.srt.constrained.triton_ops.bitmask_ops import (
    apply_token_bitmask_inplace_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(est_time=20, suite="stage-b-kernel-benchmark-1-gpu-large")

IS_CI = is_in_ci()
BITS_PER_BLOCK = 32

BATCH_SIZES = get_benchmark_range(
    full_range=[1, 4, 16, 64, 256],
    ci_range=[1, 16],
)
VOCAB_SIZES = get_benchmark_range(
    full_range=[32000, 65536, 128256, 151936],
    ci_range=[32000, 128256],
)
DTYPES = get_benchmark_range(
    full_range=[torch.float32, torch.float16, torch.bfloat16],
    ci_range=[torch.bfloat16],
)
MASK_DENSITIES = get_benchmark_range(
    full_range=[0.01, 0.5, 0.99],
    ci_range=[0.5],
)


def _make_bitmask(batch_size, vocab_size, density, device="cuda"):
    """Create a bitmask where `density` fraction of tokens are allowed (bit=1)."""
    bm_width = (vocab_size + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK
    bits = (torch.rand(batch_size, bm_width * BITS_PER_BLOCK) < density).int()
    packed = torch.zeros(batch_size, bm_width, dtype=torch.int32)
    for w in range(bm_width):
        for b in range(BITS_PER_BLOCK):
            idx = w * BITS_PER_BLOCK + b
            if idx < vocab_size:
                packed[:, w] |= bits[:, idx] << b
    return packed.to(device)


def bench():
    results = []

    configs = list(
        itertools.product(BATCH_SIZES, VOCAB_SIZES, DTYPES, MASK_DENSITIES)
    )

    for batch_size, vocab_size, dtype, density in configs:
        bm_width = (vocab_size + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK
        bitmask = _make_bitmask(batch_size, vocab_size, density)

        # --- JIT CUDA ---
        logits_jit = torch.randn(
            batch_size, vocab_size, dtype=dtype, device="cuda"
        )
        jit_us, _, _ = run_benchmark(
            lambda: apply_token_bitmask_inplace_jit(logits_jit, bitmask)
        )

        # --- Triton ---
        logits_triton = torch.randn(
            batch_size, vocab_size, dtype=dtype, device="cuda"
        )
        triton_us, _, _ = run_benchmark(
            lambda: apply_token_bitmask_inplace_triton(logits_triton, bitmask)
        )

        # --- AOT sgl-kernel (optional) ---
        aot_us = None
        try:
            from sgl_kernel import apply_token_bitmask_inplace_cuda

            logits_aot = torch.randn(
                batch_size, vocab_size, dtype=dtype, device="cuda"
            )
            aot_us, _, _ = run_benchmark(
                lambda: apply_token_bitmask_inplace_cuda(logits_aot, bitmask)
            )
        except ImportError:
            pass

        row = {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "dtype": str(dtype).split(".")[-1],
            "mask_density": density,
            "JIT CUDA (us)": f"{jit_us:.1f}",
            "Triton (us)": f"{triton_us:.1f}",
            "speedup": f"{triton_us / jit_us:.2f}x",
        }
        if aot_us is not None:
            row["AOT (us)"] = f"{aot_us:.1f}"
        results.append(row)

    df = pd.DataFrame(results)
    print("\napply-token-bitmask-inplace-performance:")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    bench()
