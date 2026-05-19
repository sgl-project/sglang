import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.fast_topk import fast_topk_v2 as jit_fast_topk_v2

try:
    from sgl_kernel import fast_topk_v2 as aot_fast_topk_v2

    AOT_AVAILABLE = True
except ImportError:
    aot_fast_topk_v2 = None
    AOT_AVAILABLE = False

TOPK = 2048

B_LIST = get_benchmark_range(
    full_range=[1, 4, 16, 64, 128],
    ci_range=[4, 16],
)

LENGTH_LIST = get_benchmark_range(
    full_range=[4096, 8192, 16384, 32768],
    ci_range=[4096, 8192],
)

configs = list(itertools.product(B_LIST, LENGTH_LIST))

line_vals = ["jit", "pytorch"]
line_names = ["SGL JIT Kernel", "PyTorch"]
styles = [("blue", "-"), ("red", "--")]

if AOT_AVAILABLE:
    line_vals.insert(1, "aot")
    line_names.insert(1, "SGL AOT Kernel")
    styles.insert(1, ("green", "-."))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "length"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="fast-topk-performance",
        args={},
    )
)
def benchmark(batch_size, length, provider):
    device = DEFAULT_DEVICE
    input_stride = length
    score = torch.randn(
        batch_size, input_stride, dtype=torch.float32, device=device
    )
    lengths = torch.full(
        (batch_size,), length, dtype=torch.int32, device=device
    )

    if provider == "jit":

        def fn():
            return jit_fast_topk_v2(score, lengths, TOPK)

    elif provider == "aot":

        def fn():
            return aot_fast_topk_v2(score, lengths, TOPK)

    else:

        def fn():
            return torch.topk(score, TOPK, dim=-1)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
