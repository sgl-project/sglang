"""Benchmark for softmax sampling kernel vs torch.softmax and flashinfer.softmax."""

import itertools

import torch
import triton
import triton.testing
from flashinfer.sampling import softmax as fi_softmax

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.softmax import softmax_sampling
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="stage-b-kernel-benchmark-1-gpu-large")

DTYPE = torch.bfloat16

VOCAB_SIZES = get_benchmark_range(
    full_range=[32000, 65536, 128256, 151936, 262144],
    ci_range=[32000, 128256],
)
BATCH_SIZES = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    ci_range=[1, 64, 512],
)
NUM_REPEAT = 4

configs = list(itertools.product(VOCAB_SIZES, BATCH_SIZES))


@torch.compile
def _torch_softmax(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    scaled = logits / temperatures.unsqueeze(1)
    return torch.softmax(scaled, dim=-1)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["vocab_size", "batch_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["jit", "flashinfer", "torch"],
        line_names=["SGL JIT Softmax", "FlashInfer", "PyTorch"],
        styles=[("blue", "-"), ("green", "-."), ("red", "--")],
        ylabel="us",
        plot_name="softmax-sampling-performance",
        args={},
    )
)
def benchmark(vocab_size: int, batch_size: int, provider: str):
    logits = torch.randn(
        (NUM_REPEAT, batch_size, vocab_size), dtype=DTYPE, device=DEFAULT_DEVICE
    )
    temperatures = torch.ones(
        (NUM_REPEAT, batch_size), dtype=torch.float32, device=DEFAULT_DEVICE
    )
    FN_MAP = {
        "jit": softmax_sampling,
        "flashinfer": fi_softmax,
        "torch": _torch_softmax,
    }

    def f() -> None:
        fn = FN_MAP[provider]
        for i in range(NUM_REPEAT):
            fn(logits[i], temperatures[i])

    return run_benchmark(f, scale=NUM_REPEAT)


if __name__ == "__main__":
    benchmark.run(print_data=True)
