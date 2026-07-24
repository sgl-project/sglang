import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate
from sglang.jit_kernel.moe_route_radix import route_radix
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

NUM_EXPERTS = 896
TOPK = 16
SCALE = 2.5


def _radix_sorted(scores, bias):
    return route_radix(scores, bias, TOPK, True, SCALE, True, sorted=True)


def _radix_unsorted(scores, bias):
    return route_radix(scores, bias, TOPK, True, SCALE, True, sorted=False)


def _triton(scores, bias):
    # fp32 scores = the triton router's production input (topk.py upcasts,
    # cost not counted here); fp32 also fails the radix covered() checks, so
    # this stays on the Triton fallback path.
    return moe_fused_gate(
        scores,
        bias,
        topk=TOPK,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=SCALE,
        apply_routed_scaling_factor_on_output=True,
    )


FN_MAP = {
    "triton": _triton,
    "radix": _radix_sorted,
    "radix_unsorted": _radix_unsorted,
}


@marker.parametrize("num_tokens", [2**n for n in range(0, 14)], [1, 64, 1024])
@marker.benchmark("provider", ["triton", "radix", "radix_unsorted"])
def benchmark(num_tokens: int, provider: str):
    torch.manual_seed(42)
    dtype = torch.float32 if provider == "triton" else torch.bfloat16
    scores = create_random(num_tokens, NUM_EXPERTS, dtype=dtype)
    bias = create_random(NUM_EXPERTS, dtype=torch.float32)
    return marker.do_bench(FN_MAP[provider], input_args=(scores, bias))


if __name__ == "__main__":
    benchmark.run()
