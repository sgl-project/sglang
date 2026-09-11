import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark_no_cudagraph,
)
from sglang.kernels.ops.elementwise.add_constant import (
    _jit_add_constant_module,
    add_constant,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=15, stage="jit-kernel-benchmark", runner_config="amd")

CONSTANT = 7
SIZE_LIST = get_benchmark_range(
    full_range=[128, 1024, 1025, 4096, 4097, 65536, 2**20, 2**22, 2**24],
    ci_range=[4096, 2**20],
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=SIZE_LIST,
        line_arg="provider",
        line_vals=["jit_module", "jit_wrapper", "torch"],
        line_names=["JIT module", "JIT wrapper", "PyTorch"],
        styles=[("blue", "-"), ("orange", "-"), ("green", "--")],
        ylabel="us",
        plot_name="add-constant-performance",
        args={},
    )
)
def benchmark(size: int, provider: str):
    src = torch.arange(size, dtype=torch.int32, device=DEFAULT_DEVICE)

    if provider == "jit_module":
        dst = torch.empty_like(src)
        module = _jit_add_constant_module(CONSTANT)

        def fn():
            module.add_constant(dst, src)

    elif provider == "jit_wrapper":

        def fn():
            add_constant(src, CONSTANT)

    else:

        def fn():
            src + CONSTANT

    return run_benchmark_no_cudagraph(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
