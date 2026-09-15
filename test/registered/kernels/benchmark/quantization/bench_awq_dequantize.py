import itertools

import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import run_benchmark
from sglang.kernels.ops.quantization.awq_dequantize import (
    awq_dequantize as jit_awq_dequantize,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

IS_CI = is_in_ci()

if IS_CI:
    qweight_row_range = [128]
    qweight_cols_range = [16]
else:
    qweight_row_range = [128, 256, 512, 1024, 3584]
    qweight_cols_range = [16, 32, 64, 128, 448]

configs = list(itertools.product(qweight_row_range, qweight_cols_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["qweight_row", "qweight_col"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["jit"],
        line_names=["JIT Kernel"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="awq-dequantize-jit",
        args={},
    )
)
def benchmark(qweight_row, qweight_col, provider):
    device = torch.device("cuda")
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (qweight_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )
    group_size = qweight_row
    scales_row = qweight_row // group_size
    scales_col = qweight_col * 8
    scales = torch.rand(scales_row, scales_col, dtype=torch.float16, device=device)
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (scales_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )

    if provider != "jit":
        raise ValueError(f"Unknown provider: {provider}")
    fn = lambda: jit_awq_dequantize(qweight, scales, qzeros)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
