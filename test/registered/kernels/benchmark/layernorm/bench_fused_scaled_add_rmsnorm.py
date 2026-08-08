import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random, get_benchmark_range
from sglang.kernels.ops.layernorm.norm import (
    fused_add_rmsnorm,
    fused_scaled_add_rmsnorm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

EPS = 1e-6
INPUT_SCALE = 0.22


def baseline(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    from sgl_kernel import rmsnorm

    scaled = input * INPUT_SCALE
    summed = residual + scaled
    return rmsnorm(summed, weight, EPS)


def prescaled_fused(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> None:
    scaled = input * INPUT_SCALE
    fused_add_rmsnorm(scaled, residual, weight, EPS)


def scaled_fused(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> None:
    fused_scaled_add_rmsnorm(input, residual, weight, EPS, INPUT_SCALE)


FN_MAP = {
    "baseline": baseline,
    "prescaled_fused": prescaled_fused,
    "scaled_fused": scaled_fused,
}


@marker.parametrize(
    "hidden_size",
    get_benchmark_range([1536, 4096, 8192], [1536]),
)
@marker.parametrize(
    "batch_size",
    get_benchmark_range([1, 16, 128, 4096], [1, 128]),
)
@marker.benchmark("impl", ["baseline", "prescaled_fused", "scaled_fused"])
def benchmark(hidden_size: int, batch_size: int, impl: str):
    input = create_random(batch_size, hidden_size)
    residual = create_random(batch_size, hidden_size)
    weight = create_random(hidden_size)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(input, residual, weight),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
