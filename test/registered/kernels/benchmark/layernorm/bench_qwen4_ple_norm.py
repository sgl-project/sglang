import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.srt.models.qwen4_exp import Qwen4ExpPLEGroupedNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


def torch_impl_ple_norm(
    norm: Qwen4ExpPLEGroupedNorm, x: torch.Tensor
) -> torch.Tensor:
    """Eager baseline (the production unfused fp32 chain)."""
    x_float = x.float()
    group_shape = x_float.shape[:-1] + (-1, norm.group_size)
    variance = x_float.reshape(group_shape).pow(2).mean(dim=-1, keepdim=True)
    variance = variance.expand(group_shape).reshape_as(x_float)
    x_norm = x_float * torch.rsqrt(variance + norm.eps)
    weight = norm.weight.float() + 1.0
    return (x_norm * weight).to(x.dtype)


FN_MAP = {
    "jit": lambda norm, x: norm(x),
    "torch_eager": torch_impl_ple_norm,
}


@marker.parametrize("num_tokens", [1, 8, 64, 512, 2048, 8192], [1, 8192])
@marker.benchmark("impl", ["jit", "torch_eager"])
def benchmark(num_tokens: int, impl: str):
    # Real Qwen4-Exp PLE shape: HC 4 x hidden 2560, grouped per hidden_size.
    hidden_size, group_size, eps = 10240, 2560, 1e-6
    norm = Qwen4ExpPLEGroupedNorm(hidden_size, eps=eps, group_size=group_size)
    norm = norm.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(create_random(hidden_size) * 0.2)
    x = create_random(num_tokens, hidden_size)
    return marker.do_bench(
        lambda x_: FN_MAP[impl](norm, x_),
        input_args=(x,),
        # x is read -> clone it per iter to avoid L2 reuse; weight is tiny.
        graph_clone_args=(0,),
    )


if __name__ == "__main__":
    benchmark.run()
