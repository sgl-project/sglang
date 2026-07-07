import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate, moe_fused_gate_jit
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=20, stage="jit-kernel-benchmark", runner_config="amd")


TOPK = 8
SCALE = 2.5


@torch.compile
def torch_router(scores, bias, topk, scoring_func):
    """Reference PyTorch router: scoring + bias + top-k + renorm + scale."""
    if scoring_func == "sigmoid":
        activated = scores.sigmoid()
    else:
        activated = torch.nn.functional.softplus(scores).sqrt()
    biased = activated + bias.unsqueeze(0)
    _, ids = torch.topk(biased, k=topk, dim=-1)
    weights = activated.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * SCALE, ids.to(torch.int32)


@marker.parametrize("scoring_func", ["sigmoid", "sqrtsoftplus"])
@marker.parametrize("num_experts", [128, 256, 384, 512], [256, 384])
@marker.parametrize("num_tokens", [1, 4, 16, 64, 512, 1024, 8192], [16, 1024])
@marker.benchmark("provider", ["triton", "jit", "torch"])
def benchmark(num_tokens: int, num_experts: int, scoring_func: str, provider: str):
    torch.manual_seed(0)
    scores = create_random(num_tokens, num_experts, dtype=torch.float32)
    bias = create_random(num_experts, dtype=torch.float32)

    common = dict(
        topk=TOPK,
        scoring_func=scoring_func,
        renormalize=True,
        routed_scaling_factor=SCALE,
        apply_routed_scaling_factor_on_output=True,
    )
    if provider == "triton":
        return marker.do_bench(
            moe_fused_gate, input_args=(scores, bias), input_kwargs=common
        )
    if provider == "jit":
        return marker.do_bench(
            moe_fused_gate_jit, input_args=(scores, bias), input_kwargs=common
        )
    if provider == "torch":
        return marker.do_bench(
            torch_router, input_args=(scores, bias, TOPK, scoring_func)
        )
    raise ValueError(f"unknown provider: {provider}")


if __name__ == "__main__":
    benchmark.run()
