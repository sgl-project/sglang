from unittest.mock import patch

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe import moe_route_warp as route_warp_module
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.kernels.ops.moe.moe_route_warp import moe_route_warp
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# name -> (experts, topk, scoring, groups, topk_groups, logits dtype)
CONFIGS = {
    "deepseek_v3": (256, 8, "sigmoid", 8, 4, torch.float32),
    "kimi_k2": (384, 8, "sigmoid", 1, 1, torch.float32),
    "glm4_5": (160, 8, "sigmoid", 1, 1, torch.float32),
    "qwen3_moe": (128, 8, "softmax", 1, 1, torch.bfloat16),
    "qwen3_5_moe": (512, 10, "softmax", 1, 1, torch.bfloat16),
}


def _triton(*args, **kwargs):
    with patch.object(route_warp_module, "covered", lambda *args: False):
        return moe_fused_gate(*args, **kwargs)


@marker.parametrize("config", list(CONFIGS), ["deepseek_v3", "kimi_k2", "qwen3_moe"])
@marker.parametrize("num_tokens", [1, 16, 64, 256, 1024, 4096, 16384], [16, 4096])
@marker.benchmark("provider", ["triton", "warp"])
def benchmark(num_tokens: int, config: str, provider: str):
    num_experts, topk, scoring, groups, topk_groups, dtype = CONFIGS[config]
    torch.manual_seed(0)
    scores = (torch.randn(num_tokens, num_experts, device="cuda") * 2).to(dtype)
    bias = (
        torch.randn(num_experts, device="cuda") * 0.1 if scoring == "sigmoid" else None
    )
    kwargs = dict(
        topk=topk,
        scoring_func=scoring,
        renormalize=True,
        routed_scaling_factor=2.5,
        num_expert_group=groups,
        topk_group=topk_groups,
    )
    fn = _triton if provider == "triton" else moe_route_warp
    return marker.do_bench(fn, input_args=(scores, bias), input_kwargs=kwargs)


if __name__ == "__main__":
    benchmark.run()
