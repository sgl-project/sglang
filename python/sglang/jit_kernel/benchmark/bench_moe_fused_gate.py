from __future__ import annotations

import itertools
from typing import Callable

import torch
import triton
import triton.testing
from sgl_kernel import kimi_k2_moe_fused_gate as aot_kimi_k2_gate
from sgl_kernel import moe_fused_gate as aot_moe_fused_gate

from sglang.jit_kernel.benchmark.utils import run_benchmark
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate, moe_fused_gate_jit
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(est_time=20, suite="base-b-kernel-benchmark-1-gpu-large")


DEVICE = "cuda"
SCALE = 2.5
TOPK = 8


@torch.compile
def torch_router(scores, bias, topk, scoring_func):
    """Reference PyTorch implementation: scoring + bias + topk + renorm."""
    if scoring_func == "sigmoid":
        activated = scores.sigmoid()
    else:
        activated = torch.nn.functional.softplus(scores).sqrt()
    biased = activated + bias.unsqueeze(0)
    _, ids = torch.topk(biased, k=topk, dim=-1)
    w = activated.gather(1, ids)
    w = w / w.sum(dim=-1, keepdim=True)
    return w * SCALE, ids.to(torch.int32)


# --- Configurations -----------------------------------------------------------

if is_in_ci():
    NUM_TOKENS_VALS = [16, 1024]
    NUM_EXPERTS_VALS = [256, 384]
else:
    NUM_TOKENS_VALS = [1, 4, 16, 64, 512, 1024, 8192]
    NUM_EXPERTS_VALS = [128, 256, 384, 512]


def _line_vals_for(scoring_func: str):
    """Pick which baselines are valid for this (num_experts, scoring_func)."""
    providers = ["triton", "torch"]
    providers.append("jit")
    if scoring_func == "sigmoid":
        providers.append("aot")
    return providers


_ALL_PROVIDERS = [
    "triton",
    "jit",
    "aot",
    "torch",
]
_PROVIDER_NAMES = {
    "triton": "Triton",
    "jit": "JIT CUDA",
    "aot": "AOT CUDA",
    "torch": "PyTorch",
}
_STYLES = {
    "triton": ("red", "-"),
    "jit": ("blue", "--"),
    "aot": ("orange", "-."),
    "torch": ("green", ":"),
}


def _run_provider(
    provider: str,
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str,
    num_experts: int,
) -> Callable:
    if provider == "triton":
        return lambda: moe_fused_gate(
            scores,
            bias,
            topk=topk,
            scoring_func=scoring_func,
            renormalize=True,
            routed_scaling_factor=SCALE,
            apply_routed_scaling_factor_on_output=True,
        )
    if provider == "jit":
        return lambda: moe_fused_gate_jit(
            scores,
            bias,
            topk=topk,
            scoring_func=scoring_func,
            renormalize=True,
            routed_scaling_factor=SCALE,
            apply_routed_scaling_factor_on_output=True,
        )
    if provider == "aot":
        if num_experts != 384:
            group = max(num_experts // 32, 1)
            return lambda: aot_moe_fused_gate(
                scores,
                bias,
                group,
                group,
                topk,
                0,
                SCALE,  # type: ignore[arg-type]
                True,  # apply_routed_scaling_factor_on_output
            )
        else:  # use special kimi kernel for 384 experts (3 groups of 128)
            return lambda: aot_kimi_k2_gate(
                scores,
                bias,
                topk=topk,
                renormalize=True,
                routed_scaling_factor=SCALE,
                apply_routed_scaling_factor_on_output=True,
            )
    if provider == "torch":
        return lambda: torch_router(scores, bias, topk, scoring_func)
    raise ValueError(provider)


# --- Benchmarks ---------------------------------------------------------------

_configs = list(itertools.product(NUM_TOKENS_VALS, NUM_EXPERTS_VALS))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts"],
        x_vals=_configs,
        line_arg="provider",
        line_vals=_ALL_PROVIDERS,
        line_names=[_PROVIDER_NAMES[p] for p in _ALL_PROVIDERS],
        styles=[_STYLES[p] for p in _ALL_PROVIDERS],
        ylabel="us",
        plot_name="router-triton-sigmoid",
        args={"scoring_func": "sigmoid"},
    )
)
def bench_router_sigmoid(num_tokens, num_experts, provider, scoring_func):
    if provider not in _line_vals_for(scoring_func):
        return float("nan"), float("nan"), float("nan")
    torch.manual_seed(0)
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(num_experts, dtype=torch.float32, device=DEVICE)
    fn = _run_provider(provider, scores, bias, TOPK, scoring_func, num_experts)
    torch.cuda.synchronize()
    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts"],
        x_vals=_configs,
        line_arg="provider",
        line_vals=_ALL_PROVIDERS,
        line_names=[_PROVIDER_NAMES[p] for p in _ALL_PROVIDERS],
        styles=[_STYLES[p] for p in _ALL_PROVIDERS],
        ylabel="us",
        plot_name="router-triton-sqrtsoftplus",
        args={"scoring_func": "sqrtsoftplus"},
    )
)
def bench_router_sqrtsoftplus(num_tokens, num_experts, provider, scoring_func):
    if provider not in _line_vals_for(scoring_func):
        return float("nan"), float("nan"), float("nan")
    torch.manual_seed(0)
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(num_experts, dtype=torch.float32, device=DEVICE)
    fn = _run_provider(provider, scores, bias, TOPK, scoring_func, num_experts)
    torch.cuda.synchronize()
    return run_benchmark(fn)


if __name__ == "__main__":
    print("--- scoring_func=sigmoid ---")
    bench_router_sigmoid.run(print_data=True)
    print("\n--- scoring_func=sqrtsoftplus ---")
    bench_router_sqrtsoftplus.run(print_data=True)
