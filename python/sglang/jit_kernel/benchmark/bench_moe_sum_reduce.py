import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.moe_sum_reduce import moe_sum_reduce as jit_moe_sum_reduce

try:
    from sgl_kernel import moe_sum_reduce as aot_moe_sum_reduce

    AOT_AVAILABLE = True
except ImportError:
    aot_moe_sum_reduce = None
    AOT_AVAILABLE = False


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return
    m, topk, k = 16, 4, 2048
    scale = 1.0
    input_tensor = torch.randn(m, topk, k, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    output_jit = torch.empty(m, k, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    output_aot = torch.empty(m, k, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    jit_moe_sum_reduce(input_tensor, output_jit, scale)
    aot_moe_sum_reduce(input_tensor, output_aot, scale)
    torch.testing.assert_close(output_jit, output_aot, rtol=0, atol=0)
    print("Correctness check passed (JIT vs AOT)")


M_LIST = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    ci_range=[16, 128],
)

TOPK_LIST = get_benchmark_range(
    full_range=[2, 4, 8],
    ci_range=[2, 4],
)

K_LIST = get_benchmark_range(
    full_range=[2048, 4096, 7168],
    ci_range=[2048, 4096],
)

configs = list(itertools.product(M_LIST, TOPK_LIST, K_LIST))

line_vals = ["jit", "pytorch"]
line_names = ["SGL JIT Kernel", "PyTorch"]
styles = [("blue", "-"), ("red", "--")]

if AOT_AVAILABLE:
    line_vals.insert(1, "aot")
    line_names.insert(1, "SGL AOT Kernel")
    styles.insert(1, ("green", "-."))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "topk", "k"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="moe-sum-reduce-performance",
        args={},
    )
)
def benchmark(m, topk, k, provider):
    scale = 1.0
    input_tensor = torch.randn(m, topk, k, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    output_tensor = torch.empty(m, k, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)

    if provider == "jit":

        def fn():
            jit_moe_sum_reduce(input_tensor, output_tensor, scale)

    elif provider == "aot":

        def fn():
            aot_moe_sum_reduce(input_tensor, output_tensor, scale)

    else:  # pytorch

        def fn():
            torch.sum(input_tensor, dim=1, out=output_tensor)
            output_tensor.mul_(scale)

    return run_benchmark(fn)


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
