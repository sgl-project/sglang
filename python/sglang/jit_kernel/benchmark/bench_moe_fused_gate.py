import itertools

import torch
import triton
import triton.testing
from sgl_kernel import kimi_k2_moe_fused_gate as aot_kimi_k2_gate
from sgl_kernel import moe_fused_gate as aot_moe_fused_gate

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate, moe_fused_gate_jit
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="base-b-kernel-benchmark-1-gpu-large")


DTYPE = torch.float32
DEVICE = "cuda"
TOPK = 8
SCALE = 2.5

# AOT moe_fused_gate requires experts_per_group <= 32, so split experts into
# groups of 32 and select every group (topk_group == num_expert_group) to get a
# flat top-k. The 384-expert (3x128) layout uses the dedicated Kimi-K2 kernel.
AOT_GROUP_SIZE = 32

NUM_TOKENS_LIST = get_benchmark_range(
    full_range=[1, 4, 16, 64, 512, 1024, 8192],
    ci_range=[16, 1024],
)
NUM_EXPERTS_LIST = get_benchmark_range(
    full_range=[128, 256, 384, 512],
    ci_range=[256, 384],
)

# The AOT CUDA kernels only implement sigmoid scoring.
LINE_VALS = ["triton", "jit", "aot", "torch"]
LINE_NAMES = ["Triton", "JIT CUDA", "AOT CUDA", "PyTorch"]
STYLES = [("red", "-"), ("blue", "--"), ("orange", "-."), ("green", ":")]

configs = list(itertools.product(NUM_TOKENS_LIST, NUM_EXPERTS_LIST))


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


def _aot_fn(scores, bias, num_experts):
    if num_experts == 384:  # 3 groups of 128 -> dedicated Kimi-K2 kernel
        return lambda: aot_kimi_k2_gate(
            scores,
            bias,
            topk=TOPK,
            renormalize=True,
            routed_scaling_factor=SCALE,
            apply_routed_scaling_factor_on_output=True,
        )
    num_group = max(num_experts // AOT_GROUP_SIZE, 1)
    return lambda: aot_moe_fused_gate(
        scores,
        bias,
        num_group,
        num_group,
        TOPK,
        0,  # num_fused_shared_experts
        SCALE,
        True,  # apply_routed_scaling_factor_on_output
    )


def _benchmark(num_tokens, num_experts, provider, scoring_func):
    torch.manual_seed(0)
    scores = torch.randn(num_tokens, num_experts, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(num_experts, dtype=DTYPE, device=DEVICE)

    FN_MAP = {
        "triton": lambda: moe_fused_gate(
            scores,
            bias,
            topk=TOPK,
            scoring_func=scoring_func,
            renormalize=True,
            routed_scaling_factor=SCALE,
            apply_routed_scaling_factor_on_output=True,
        ),
        "jit": lambda: moe_fused_gate_jit(
            scores,
            bias,
            topk=TOPK,
            scoring_func=scoring_func,
            renormalize=True,
            routed_scaling_factor=SCALE,
            apply_routed_scaling_factor_on_output=True,
        ),
        "torch": lambda: torch_router(scores, bias, TOPK, scoring_func),
    }
    if scoring_func == "sigmoid":
        FN_MAP["aot"] = _aot_fn(scores, bias, num_experts)

    if provider not in FN_MAP:
        return float("nan"), float("nan"), float("nan")
    torch.cuda.synchronize()
    return run_benchmark(FN_MAP[provider])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="moe-fused-gate-sigmoid",
        args={"scoring_func": "sigmoid"},
    )
)
def benchmark_sigmoid(num_tokens, num_experts, provider, scoring_func):
    return _benchmark(num_tokens, num_experts, provider, scoring_func)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="moe-fused-gate-sqrtsoftplus",
        args={"scoring_func": "sqrtsoftplus"},
    )
)
def benchmark_sqrtsoftplus(num_tokens, num_experts, provider, scoring_func):
    return _benchmark(num_tokens, num_experts, provider, scoring_func)


if __name__ == "__main__":
    benchmark_sigmoid.run(print_data=True)
    benchmark_sqrtsoftplus.run(print_data=True)
