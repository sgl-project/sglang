from flashinfer.norm import fused_add_rmsnorm as fi_fused_add_rmsnorm
from flashinfer.norm import rmsnorm as fi_rmsnorm

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.layernorm.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm
from sglang.kernels.ops.layernorm.norm import rmsnorm as jit_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


NORM_DIMS = [1024 * n for n in range(1, 9)] + [512, 768, 1536, 12288, 16384]
NORM_DIMS.sort()

BS_LIST = [2**n for n in range(14)]
BS_LIST += [int(x * 1.5) for x in BS_LIST] + [2**14]
BS_LIST.sort()


@marker.parametrize("hidden_size", NORM_DIMS, [7168])
@marker.parametrize("batch_size", BS_LIST, [1, 64, 4096])
@marker.benchmark("impl", ["jit", "flashinfer"])
def benchmark_rmsnorm(hidden_size: int, batch_size: int, impl: str):
    input = create_random(batch_size, hidden_size)
    weight = create_random(hidden_size)
    out = create_random(batch_size, hidden_size)
    FN_MAP = {"jit": jit_rmsnorm, "flashinfer": fi_rmsnorm}
    fn = FN_MAP[impl]
    return marker.do_bench(
        fn,
        input_args=(input, weight),
        input_kwargs={"out": out},
        memory_args=(input, weight),
        memory_output=(input,),
    )


@marker.parametrize("hidden_size", NORM_DIMS, [7168])
@marker.parametrize("batch_size", BS_LIST, [1, 64, 4096])
@marker.benchmark("impl", ["jit", "flashinfer"])
def benchmark_fused_add_rmsnorm(hidden_size: int, batch_size: int, impl: str):
    input = create_random(batch_size, hidden_size)
    residual = create_random(batch_size, hidden_size)
    weight = create_random(hidden_size)
    FN_MAP = {"jit": jit_fused_add_rmsnorm, "flashinfer": fi_fused_add_rmsnorm}
    fn = FN_MAP[impl]
    return marker.do_bench(
        fn,
        input_args=(input, residual, weight),
        memory_output=(input, residual),
    )


if __name__ == "__main__":
    print("Benchmarking rmsnorm...")
    benchmark_rmsnorm.run()

    print("Benchmarking fused_add_rmsnorm...")
    benchmark_fused_add_rmsnorm.run()
