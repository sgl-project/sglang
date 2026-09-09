"""Benchmark Triton and CUDA quantization with an automatic CTA budget.

All providers start from the same BF16 NoPE/RoPE values and finish with the
same two staging tensors: uint8 ``[N, 1, 528]`` NoPE data/scales and uint8
``[N, 1, 128]`` RoPE bytes. Cache scatter is outside this benchmark.

CUDA derives its CTA budget from the device SM count and kernel occupancy.
"""

from __future__ import annotations

import itertools
from typing import Tuple

import torch
import triton.testing

from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_QUANTILES,
    get_benchmark_range,
)
from sglang.kernels.ops.attention.dsa.quant_k_cache import (
    _quantize_k_cache_fast_separate_cuda,
    _quantize_k_cache_fast_separate_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

NOPE_DIM = 512
ROPE_DIM = 64
SM_COUNT = torch.cuda.get_device_properties(DEFAULT_DEVICE).multi_processor_count
BS_RANGE = get_benchmark_range(
    full_range=[
        1,
        2,
        4,
        8,
        10,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        4096,
        8192,
        16384,
        32768,
    ],
    ci_range=[1, 4, 10, 64, 512, 8192, 16384],
)

LINE_VALS = ["triton", "cuda"]
LINE_NAMES = ["Triton", "CUDA"]
STYLES = [("red", "-."), ("blue", "-")]
CONFIGS = list(itertools.product(BS_RANGE))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="mla-k-cache-quantize-performance",
        args={},
    )
)
def benchmark(batch_size: int, provider: str) -> Tuple[float, float, float]:
    # Match OperatorsForge: two independently allocated, row-contiguous inputs.
    torch.manual_seed(0)
    k_nope = torch.randn(
        (batch_size, NOPE_DIM),
        dtype=torch.bfloat16,
        device=DEFAULT_DEVICE,
    )
    k_rope = torch.randn(
        (batch_size, ROPE_DIM),
        dtype=torch.bfloat16,
        device=DEFAULT_DEVICE,
    )

    if provider == "triton":

        def fn():
            _quantize_k_cache_fast_separate_triton(k_nope, k_rope)

    elif provider == "cuda":

        def fn():
            _quantize_k_cache_fast_separate_cuda(k_nope, k_rope)

    else:
        raise ValueError(f"Unknown benchmark provider: {provider}")

    # Compile all lazy kernels before CUDA-graph timing.
    fn()
    torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=DEFAULT_QUANTILES
    )
    return (
        1000 * ms,
        1000 * max_ms,
        1000 * min_ms,
    )


def run_console_benchmark() -> None:
    """Print median latency in microseconds and speedups of CUDA."""

    print(
        f"SMs={SM_COUNT}; latencies in us",
        flush=True,
    )
    print(
        f"{'N':>8} {'triton':>12} {'cuda':>12} {'triton/cuda':>12}",
        flush=True,
    )
    for batch_size in BS_RANGE:
        medians = {}
        for provider in LINE_VALS:
            median_us, _, _ = benchmark.fn(
                batch_size=batch_size,
                provider=provider,
            )
            medians[provider] = median_us

        triton_speedup = medians["triton"] / medians["cuda"]
        print(
            f"{batch_size:8d} {medians['triton']:12.3f} "
            f"{medians['cuda']:12.3f} {triton_speedup:11.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    run_console_benchmark()
