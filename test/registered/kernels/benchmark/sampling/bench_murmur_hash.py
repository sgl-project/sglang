import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.sampling.murmur_hash import (
    _murmur_hash32_jit,
    _murmur_hash32_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

FN_MAP = {
    "jit": _murmur_hash32_jit,
    "triton": _murmur_hash32_triton,
}


# `m` spans typical vocab sizes, `n` typical sampling batch sizes.
@marker.parametrize("n", [1, 4, 16, 64, 256], [16])
@marker.parametrize("m", [2**14, 2**16, 2**17], [2**14])
@marker.benchmark("impl", ["jit", "triton"])
def benchmark(n: int, m: int, impl: str):
    seed = torch.randint(
        0, torch.iinfo(torch.int64).max, (n,), dtype=torch.uint64, device="cuda"
    )
    positions = torch.arange(n, dtype=torch.int64, device="cuda")
    col_indices = torch.arange(m, dtype=torch.int64, device="cuda")
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(seed, positions, col_indices),
        # All three tensors are read every call; clone them per iteration so
        # L2 reuse does not skew the timings.
        graph_clone_args=(0, 1, 2),
        # Defaults already report bandwidth: memory_args="all" counts the three
        # inputs, memory_output="out" counts the returned (n, m) uint32 tensor.
    )


if __name__ == "__main__":
    benchmark.run()
