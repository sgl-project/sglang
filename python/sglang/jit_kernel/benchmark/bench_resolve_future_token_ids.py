import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.resolve_future_token_ids import resolve_future_token_ids_cuda
from sglang.srt.utils import get_compiler_backend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-benchmark-1-gpu-large")

SIZE_LIST = get_benchmark_range(
    full_range=[2**n for n in range(4, 16)] + [2**20, 2**22, 2**24],
    ci_range=[256, 4096],
)
DTYPE_LIST = get_benchmark_range(
    full_range=[torch.int32, torch.int64],
    ci_range=[torch.int32],
)

configs = list(itertools.product(SIZE_LIST, DTYPE_LIST))


def _torch_resolve(input_ids, future_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


_compiled_resolve = torch.compile(
    _torch_resolve, dynamic=True, backend=get_compiler_backend()
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size", "dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["jit", "torch_compile", "torch"],
        line_names=["SGL JIT Kernel", "torch.compile", "PyTorch"],
        styles=[("blue", "-"), ("green", "-."), ("red", "--")],
        ylabel="us",
        plot_name="resolve-future-token-ids-performance",
        args={},
    )
)
def benchmark(size: int, dtype: torch.dtype, provider: str):
    map_size = 8192
    future_map = torch.randint(
        0, 50000, (map_size,), dtype=dtype, device=DEFAULT_DEVICE
    )
    input_ids = torch.randint(
        -map_size + 1, 50000, (size,), dtype=dtype, device=DEFAULT_DEVICE
    )

    if provider == "jit":
        fn = lambda: resolve_future_token_ids_cuda(input_ids.clone(), future_map)
    elif provider == "torch_compile":
        fn = lambda: _compiled_resolve(input_ids.clone(), future_map)
    else:
        fn = lambda: _torch_resolve(input_ids.clone(), future_map)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
