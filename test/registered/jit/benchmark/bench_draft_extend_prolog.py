"""Benchmark EAGLE draft-extend metadata construction outside CUDA graphs."""

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark_no_cudagraph,
)
from sglang.kernels.ops.attention.position import compute_position_triton
from sglang.kernels.ops.speculative.draft_extend import fused_draft_extend_prolog
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-small"
)

BATCH_SIZE_RANGE = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128],
    ci_range=[1, 16, 128],
)
NUM_DRAFT_TOKENS = 6
FRONT_OFFSET = 0
WINDOW_SIZE = NUM_DRAFT_TOKENS + FRONT_OFFSET


def eager_draft_extend_prolog(seq_lens: torch.Tensor):
    batch_size = seq_lens.numel()
    prefix_lens = (seq_lens - FRONT_OFFSET).clamp(min=0).to(torch.int32)
    extend_seq_lens = torch.full(
        (batch_size,),
        WINDOW_SIZE,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    positions, extend_start_loc = compute_position_triton(
        prefix_lens,
        extend_seq_lens,
        batch_size * WINDOW_SIZE,
    )
    output_seq_lens = seq_lens + NUM_DRAFT_TOKENS
    return (
        prefix_lens,
        extend_seq_lens,
        positions,
        extend_start_loc,
        output_seq_lens,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=BATCH_SIZE_RANGE,
        line_arg="provider",
        line_vals=["eager", "fused"],
        line_names=["PyTorch + position kernel", "Fused prolog"],
        styles=[("green", "--"), ("blue", "-")],
        ylabel="us",
        plot_name="draft-extend-prolog",
        args={},
    )
)
def benchmark(batch_size: int, provider: str):
    seq_lens = torch.randint(
        1, 8192, (batch_size,), dtype=torch.int64, device=DEFAULT_DEVICE
    )
    if provider == "eager":
        fn = lambda: eager_draft_extend_prolog(seq_lens)
    else:
        fn = lambda: fused_draft_extend_prolog(
            seq_lens, NUM_DRAFT_TOKENS, front_offset=FRONT_OFFSET
        )
    return run_benchmark_no_cudagraph(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
