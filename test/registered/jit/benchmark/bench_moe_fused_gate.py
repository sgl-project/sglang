import torch
from sgl_kernel import kimi_k2_moe_fused_gate as aot_kimi_k2_gate
from sgl_kernel import moe_fused_gate as aot_moe_fused_gate

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate, moe_fused_gate_jit
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


TOPK = 8
SCALE = 2.5
# AOT moe_fused_gate requires experts_per_group <= 32, so split experts into
# groups of 32 and select every group (topk_group == num_expert_group) to get a
# flat top-k. The 384-expert (3x128) layout uses the dedicated Kimi-K2 kernel.
AOT_GROUP_SIZE = 32


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
@marker.benchmark("provider", ["triton", "jit", "aot", "torch"])
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
    if provider == "aot":
        # The AOT CUDA kernels only implement sigmoid scoring.
        if scoring_func != "sigmoid":
            marker.skip("AOT kernel supports sigmoid only")
        if num_experts == 384:  # 3 groups of 128 -> dedicated Kimi-K2 kernel
            return marker.do_bench(
                aot_kimi_k2_gate,
                input_args=(scores, bias),
                input_kwargs=dict(
                    topk=TOPK,
                    renormalize=True,
                    routed_scaling_factor=SCALE,
                    apply_routed_scaling_factor_on_output=True,
                ),
            )
        num_group = max(num_experts // AOT_GROUP_SIZE, 1)
        return marker.do_bench(
            aot_moe_fused_gate,
            input_args=(
                scores,
                bias,
                num_group,
                num_group,
                TOPK,
                0,  # num_fused_shared_experts
                SCALE,
                True,  # apply_routed_scaling_factor_on_output
            ),
        )
    raise ValueError(f"unknown provider: {provider}")


if __name__ == "__main__":
    benchmark.run()
