import torch
from flashinfer.norm import fused_add_rmsnorm as fi_fused_add_rmsnorm
from flashinfer.norm import rmsnorm as fi_rmsnorm

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm
from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


HIDDEN_SIZE_LIST = sorted([1536, *range(1024, 8192 + 1, 1024)])


def torch_rmsnorm(input: torch.Tensor, weight: torch.Tensor) -> None:
    fi_rmsnorm(input, weight, out=input)


FN_MAP_RMSNORM = {
    "jit": lambda input, weight: jit_rmsnorm(input, weight, out=input),
    "flashinfer": torch_rmsnorm,
}

FN_MAP_FUSED = {
    "jit": jit_fused_add_rmsnorm,
    "flashinfer": fi_fused_add_rmsnorm,
}


@marker.parametrize("hidden_size", HIDDEN_SIZE_LIST + [16384], [512, 2048])
@marker.parametrize("batch_size", [2**n for n in range(0, 14)], [16, 32])
@marker.benchmark("impl", ["flashinfer", "jit"])
def benchmark_rmsnorm(hidden_size: int, batch_size: int, impl: str):
    input = create_random(batch_size, hidden_size)
    weight = create_random(hidden_size)
    fn = FN_MAP_RMSNORM[impl]
    try:
        return marker.do_bench(
            fn,
            input_args=(input, weight),
            graph_clone_args=(0, 1),
            memory_output=(input,),  # inplace write to input
        )
    except RuntimeError as e:
        marker.skip(str(e))


@marker.parametrize("hidden_size", HIDDEN_SIZE_LIST, [512, 2048])
@marker.parametrize("batch_size", [2**n for n in range(0, 14)], [16, 32])
@marker.benchmark("impl", ["flashinfer", "jit"])
def benchmark_fused_add_rmsnorm(hidden_size: int, batch_size: int, impl: str):
    input = create_random(batch_size, hidden_size)
    residual = create_random(batch_size, hidden_size)
    weight = create_random(hidden_size)
    fn = FN_MAP_FUSED[impl]
    try:
        return marker.do_bench(
            fn,
            input_args=(input, residual, weight),
            graph_clone_args=(0, 1, 2),
            memory_output=(input, residual),  # inplace write to input, residual
        )
    except RuntimeError as e:
        marker.skip(str(e))


if __name__ == "__main__":
    print("Benchmarking rmsnorm...")
    benchmark_rmsnorm.run()

    print("Benchmarking fused_add_rmsnorm...")
    benchmark_fused_add_rmsnorm.run()
