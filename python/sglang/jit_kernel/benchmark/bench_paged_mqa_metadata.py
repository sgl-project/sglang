"""Benchmark paged_mqa_metadata JIT kernel.

Reports per-shape median latency in µs via ``marker.do_bench`` (CUDA-graph
timing).

Shape axes:
  - ``bs``: dense sweep from single-request decode (1) to large multi-block
    batch (32768). Covers the three internal dispatch paths
    (tiny ``bs<=64`` / small ``bs<=2048`` / multi-block ``bs>2048``).
  - ``max_ctx``: two extremes (2048, 32768). The kernel is value-invariant
    (cost is O(bs) regardless of seq_lens values); sweeping both bookends
    makes that empirically visible.

Constants: ``num_sm`` queried from the active GPU; ``page_size = 64``.

Local run:
    python python/sglang/jit_kernel/benchmark/bench_paged_mqa_metadata.py
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.dsv4 import get_paged_mqa_logits_metadata
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="base-b-kernel-benchmark-1-gpu-large")


NUM_SM = (
    torch.cuda.get_device_properties(0).multi_processor_count
    if torch.cuda.is_available()
    else 132
)
PAGE_SIZE = 64
DEVICE = "cuda"


def _make_seq_lens(bs: int, max_ctx: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randint(
        1, max_ctx + 1, (bs,), dtype=torch.int32, device=DEVICE, generator=g
    )


@marker.parametrize(
    "bs",
    [1, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    [128, 2048],
)
@marker.parametrize("max_ctx", [2048, 32768], [8192])
@marker.benchmark("impl", ["jit"])
def benchmark(bs: int, max_ctx: int, impl: str):
    seq_lens = _make_seq_lens(bs, max_ctx)
    return marker.do_bench(
        get_paged_mqa_logits_metadata,
        input_args=(seq_lens, PAGE_SIZE, NUM_SM),
    )


if __name__ == "__main__":
    benchmark.run()
