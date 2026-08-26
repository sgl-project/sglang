import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
    grouped_gemma_rmsnorm as jit_grouped_gemma_rmsnorm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


def torch_impl_grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float
) -> torch.Tensor:
    """Eager baseline (the production unfused chain)."""
    x_float = x.float()
    hidden = x_float.shape[-1]
    x_grouped = x_float.reshape(*x_float.shape[:-1], hidden // group_size, group_size)
    variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x_grouped * torch.rsqrt(variance + eps)).flatten(-2)
    return (x_norm * (1.0 + weight.float())).to(x.dtype)


FN_MAP = {
    "jit": jit_grouped_gemma_rmsnorm,
    "torch_eager": torch_impl_grouped_gemma_rmsnorm,
}


@marker.parametrize("num_tokens", [1, 8, 64, 512, 2048, 8192, 16384], [1, 512, 8192])
@marker.benchmark("impl", ["jit", "torch_eager"])
def benchmark(num_tokens: int, impl: str):
    hidden_size, group_size, eps = 10240, 2560, 1e-6
    x = create_random(num_tokens, hidden_size)
    weight = create_random(hidden_size) * 0.2
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x, weight, group_size, eps),
        # x is read -> clone it per iter to avoid L2 reuse; weight is tiny.
        graph_clone_args=(0,),
    )


if __name__ == "__main__":
    benchmark.run()
