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
from sglang.jit_kernel.moe_sum import moe_sum as jit_moe_sum

try:
    from sgl_kernel import moe_sum as aot_moe_sum

    AOT_AVAILABLE = True
except ImportError:
    aot_moe_sum = None
    AOT_AVAILABLE = False

M_LIST = get_benchmark_range(
    full_range=[1, 4, 16, 64, 256, 1024],
    ci_range=[64],
)
TOPK_LIST = get_benchmark_range(
    full_range=[2, 3, 4],
    ci_range=[2],
)
K_LIST = get_benchmark_range(
    full_range=[128, 512, 1024, 4096],
    ci_range=[1024],
)

configs = list(itertools.product(M_LIST, TOPK_LIST, K_LIST))


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return
    for topk in [2, 3, 4]:
        inp = torch.randn((64, topk, 1024), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
        out_jit = torch.empty((64, 1024), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
        out_aot = torch.empty((64, 1024), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
        jit_moe_sum(inp, out_jit)
        aot_moe_sum(inp, out_aot)
        torch.cuda.synchronize()
        torch.testing.assert_close(out_jit, out_aot, rtol=0, atol=0)
    print("Correctness check passed (JIT vs AOT)")


if AOT_AVAILABLE:
    line_vals = ["jit", "aot", "torch"]
    line_names = ["SGL JIT Kernel", "SGL AOT Kernel", "PyTorch"]
    styles = [("blue", "-"), ("green", "-"), ("red", "--")]
else:
    line_vals = ["jit", "torch"]
    line_names = ["SGL JIT Kernel", "PyTorch"]
    styles = [("blue", "-"), ("red", "--")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "topk", "k"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="moe-sum-performance",
        args={},
    )
)
def benchmark(m: int, topk: int, k: int, provider: str):
    inp = torch.randn((m, topk, k), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    output = torch.empty((m, k), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)

    if provider == "jit":
        fn = lambda: jit_moe_sum(inp, output)
    elif provider == "aot":
        fn = lambda: aot_moe_sum(inp, output)
    else:
        fn = lambda: torch.sum(inp, dim=1, out=output)

    return run_benchmark(fn)


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
