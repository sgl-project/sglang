import sys

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe import moe_route_radix4
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="base-a-jit-kernel-benchmark", runner_config="amd")

NUM_EXPERTS = 896
TOPK = 16
# Kimi-K3's routed latent width, the row the fused launch also zeroes.
MODEL_DIM = 3584
SCALING = 2.5

if not moe_route_radix4.supported_hardware():
    print("moe_route_radix4 targets gfx942/gfx950; nothing to benchmark")
    sys.exit(0)

from aiter import biased_grouped_topk as aiter_biased_grouped_topk  # noqa: E402
from aiter.fused_moe import moe_sorting  # noqa: E402

SORT_WS = moe_route_radix4.new_sort_workspace(torch.device("cuda"))


# The sorts AITER's fused_moe consumes: block_m and whether it is the aux sort (which also
# writes the aux tensors); every one zeroes moe_buf.
SORTS = {"bm16_aux": (16, True), "bm16": (16, False), "bm32": (32, False)}


def _aiter_sort(ids, weights, moe_buf, sort):
    block_m, aux = SORTS[sort]
    return moe_sorting(
        ids,
        weights,
        NUM_EXPERTS,
        MODEL_DIM,
        torch.bfloat16,
        block_size=block_m,
        output_aux=aux,
        output=moe_buf,
    )


def _aiter_route_then_sort(scores, bias, moe_buf, sort):
    weights = torch.empty((scores.shape[0], TOPK), dtype=torch.float32, device="cuda")
    ids = torch.empty((scores.shape[0], TOPK), dtype=torch.int32, device="cuda")
    aiter_biased_grouped_topk(scores, bias, weights, ids, 1, 1, True, SCALING)
    return _aiter_sort(ids, weights, moe_buf, sort)


def _radix4_then_aiter_sort(scores, bias, moe_buf, sort):
    weights, ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, SCALING)
    return _aiter_sort(ids, weights, moe_buf, sort)


def _radix4_fused_sort(scores, bias, moe_buf, sort):
    return moe_route_radix4.route_radix4_with_sort(
        scores,
        bias,
        TOPK,
        True,
        SCALING,
        sort_ws=SORT_WS,
        block_m=SORTS[sort][0],
        moe_buf=moe_buf,
    )


PROVIDERS = {
    "aiter_route_then_sort": _aiter_route_then_sort,
    "radix4_then_aiter_sort": _radix4_then_aiter_sort,
    "radix4_fused_sort": _radix4_fused_sort,
}


# fp32 scores and bias are what Kimi-K3's router dispatches. marker.do_bench clones
# the inputs per replay, so every launch reads cold L2.
@marker.parametrize("sort", list(SORTS))
@marker.parametrize("num_tokens", [1, 16, 64, 128, 256, 512, 1024], [16, 256])
@marker.benchmark("provider", list(PROVIDERS))
def benchmark(num_tokens: int, sort: str, provider: str):
    torch.manual_seed(0)
    scores = torch.randn(num_tokens, NUM_EXPERTS, device="cuda")
    bias = torch.randn(NUM_EXPERTS, device="cuda") * 0.1
    moe_buf = torch.empty(num_tokens, MODEL_DIM, dtype=torch.bfloat16, device="cuda")
    return marker.do_bench(
        PROVIDERS[provider],
        input_args=(scores, bias, moe_buf),
        input_kwargs={"sort": sort},
    )


if __name__ == "__main__":
    benchmark.run()
