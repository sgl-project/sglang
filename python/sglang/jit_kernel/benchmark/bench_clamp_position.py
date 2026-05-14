import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.clamp_position import clamp_position_cuda
from sglang.srt.utils import get_compiler_backend
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=13, suite="stage-b-kernel-benchmark-1-gpu-large")
register_amd_ci(est_time=16, suite="jit-kernel-unit-test-amd")

SIZE_LIST = get_benchmark_range(
    full_range=[2**n for n in range(4, 16)],
    ci_range=[256, 4096],
)

configs = list(itertools.product(SIZE_LIST))


def _torch_clamp_position(seq_lens):
    return torch.clamp(seq_lens - 1, min=0).to(torch.int64)


_compiled_clamp_position = torch.compile(
    _torch_clamp_position, dynamic=True, backend=get_compiler_backend()
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["jit", "torch_compile", "torch"],
        line_names=["SGL JIT Kernel", "torch.compile", "PyTorch"],
        styles=[("blue", "-"), ("green", "-."), ("red", "--")],
        ylabel="us",
        plot_name="clamp-position-performance",
        args={},
    )
)
def benchmark(size: int, provider: str):
    seq_lens = torch.randint(
        0, 10000, (size,), dtype=torch.int64, device=DEFAULT_DEVICE
    )

    if provider == "jit":
        fn = lambda: clamp_position_cuda(seq_lens)
    elif provider == "torch_compile":
        fn = lambda: _compiled_clamp_position(seq_lens)
    else:
        fn = lambda: _torch_clamp_position(seq_lens)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
