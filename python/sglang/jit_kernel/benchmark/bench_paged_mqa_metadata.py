"""Benchmark paged_mqa_metadata JIT kernel.

Reports per-shape median latency in µs via CUDA-graph timing
(``triton.testing.do_bench_cudagraph`` through
``sglang.jit_kernel.benchmark.utils.run_benchmark``).

Shape axes:
  - ``bs``: dense sweep, 15 values from single-request decode (1) to large
    multi-block batch (32768). Covers the three internal dispatch paths
    (tiny ``bs<=64`` / small ``bs<=2048`` / multi-block ``bs>2048``).
  - ``max_ctx``: 2 extreme values (2048, 32768) — the algorithm's compute
    cost is O(bs) regardless of seq_lens values; including both bookends
    makes the (small, expected) seq_lens-value variance empirically visible.

Constants:
  - ``num_sm = 132`` (H200; matches the kernel's tuned per-shape grid)
  - ``page_size = 64`` (fixed by sglang public API)

Local run:
    python python/sglang/jit_kernel/benchmark/bench_paged_mqa_metadata.py
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.deepseek_v4 import get_paged_mqa_logits_metadata
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="base-b-kernel-benchmark-1-gpu-large")


NUM_SM = 132
PAGE_SIZE = 64
DEVICE = "cuda"

BS_LIST = get_benchmark_range(
    full_range=[
        1,
        8,
        16,
        32,
        64,
        128,
        256,
        384,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    ],
    ci_range=[128, 2048],
)
MAX_CTX_LIST = get_benchmark_range(
    full_range=[2048, 32768],
    ci_range=[8192],
)
configs = list(itertools.product(BS_LIST, MAX_CTX_LIST))


def _make_seq_lens(bs: int, max_ctx: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randint(
        1, max_ctx + 1, (bs,), dtype=torch.int32, device=DEVICE, generator=g
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "max_ctx"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["jit"],
        line_names=["SGL JIT Kernel"],
        styles=[("green", "-")],
        ylabel="us (median)",
        plot_name=f"paged_mqa_metadata-perf (num_sm={NUM_SM}, page_size={PAGE_SIZE})",
        args={"num_sm": NUM_SM, "page_size": PAGE_SIZE},
    )
)
def benchmark(bs: int, max_ctx: int, num_sm: int, page_size: int, provider: str):
    seq_lens = _make_seq_lens(bs, max_ctx)
    fn = lambda: get_paged_mqa_logits_metadata(seq_lens, page_size, num_sm)
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
