import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.kernels.ops.moe.shuffle_rows_with_scales import shuffle_rows_with_scales
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

GROUP_SIZE = 128

# (hidden, tokens). The cutlass fp8 blockwise MoE gathers tokens * topk rows out
# of a tokens-row source, so the small end is bs=1 decode and the large end is a
# prefill-sized batch. 7168 is a DeepSeek-class hidden size.
SHAPES = get_benchmark_range(
    full_range=[(7168, 1), (2048, 1), (7168, 8), (7168, 64), (7168, 1024)],
    ci_range=[(7168, 1), (7168, 1024)],
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["k", "tokens"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=["fused", "two_launches"],
        line_names=["Fused gather", "Two shuffle_rows"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="shuffle-rows-with-scales-performance",
        args={"topk": 8},
    )
)
def benchmark(k: int, tokens: int, topk: int, provider: str):
    from sgl_kernel import shuffle_rows

    rows = tokens * topk
    q = torch.randint(
        0, 256, (tokens, k), dtype=torch.uint8, device=DEFAULT_DEVICE
    ).view(torch.float8_e4m3fn)
    scale = torch.randn(
        (tokens, k // GROUP_SIZE), dtype=torch.float32, device=DEFAULT_DEVICE
    )
    # Duplicate source rows are the normal case: a token is replicated once per
    # expert it routes to.
    dst2src = torch.randint(
        0, tokens, (rows,), dtype=torch.int32, device=DEFAULT_DEVICE
    )

    if provider == "fused":
        fn = lambda: shuffle_rows_with_scales(q, scale, dst2src, rows)
    else:
        fn = lambda: (
            shuffle_rows(q, dst2src, (rows, k)),
            shuffle_rows(scale, dst2src, (rows, k // GROUP_SIZE)),
        )

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
