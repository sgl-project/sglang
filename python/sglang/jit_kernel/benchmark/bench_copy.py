import itertools

import torch
import triton
import triton.testing
from sgl_kernel.elementwise import copy_to_gpu_no_ce as aot_copy_to_gpu_no_ce

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.jit_kernel.copy import copy_to_gpu_no_ce as jit_copy_to_gpu_no_ce

IS_CI = is_in_ci()

DEVICE = "cuda"

if IS_CI:
    SIZE_LIST = [64, 72]
else:
    SIZE_LIST = [64, 72]

LINE_VALS = ["jit", "aot"]
LINE_NAMES = ["SGL JIT Kernel", "SGL AOT Kernel"]
STYLES = [("orange", "-"), ("blue", "--")]

configs = list(itertools.product(SIZE_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=SIZE_LIST,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="copy-to-gpu-no-ce-performance",
        args={},
    )
)
def benchmark(size: int, provider: str):
    tensor_cpu = torch.randint(0, 1_000_000, (size,), dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty(size, dtype=torch.int32, device=DEVICE)
    FN_MAP = {
        "jit": jit_copy_to_gpu_no_ce,
        "aot": aot_copy_to_gpu_no_ce,
    }
    fn = lambda: FN_MAP[provider](tensor_cpu, tensor_gpu)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)  # type: ignore
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
