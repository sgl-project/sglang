import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.awq_dequantize import awq_dequantize as jit_awq_dequantize
from sglang.jit_kernel.benchmark.utils import is_in_ci, run_benchmark

try:
    from sgl_kernel import awq_dequantize as aot_awq_dequantize

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

IS_CI = is_in_ci()

if IS_CI:
    qweight_row_range = [128]
    qweight_cols_range = [16]
else:
    qweight_row_range = [128, 256, 512, 1024, 3584]
    qweight_cols_range = [16, 32, 64, 128, 448]

configs = list(itertools.product(qweight_row_range, qweight_cols_range))


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return

    qweight_row, qweight_col = 128, 16
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

    jit_out = jit_awq_dequantize(qweight, scales, qzeros)
    aot_out = aot_awq_dequantize(qweight, scales, qzeros)
    torch.cuda.synchronize()
    torch.testing.assert_close(jit_out, aot_out, rtol=0, atol=0)
    print("Correctness check passed (JIT vs AOT)")


if AOT_AVAILABLE:
    line_vals = ["jit", "aot"]
    line_names = ["JIT Kernel", "AOT Kernel"]
    styles = [("blue", "-"), ("green", "-")]
else:
    line_vals = ["jit"]
    line_names = ["JIT Kernel"]
    styles = [("blue", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["qweight_row", "qweight_col"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="awq-dequantize-jit-vs-aot",
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

    if provider == "jit":
        fn = lambda: jit_awq_dequantize(qweight, scales, qzeros)
    elif provider == "aot":
        fn = lambda: aot_awq_dequantize(qweight, scales, qzeros)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
