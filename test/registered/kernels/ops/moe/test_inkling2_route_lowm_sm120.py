"""SM12x low-M route kernels are bitwise equal to route_topk + route_metadata."""

import sys

import pytest
import torch

from sglang.kernels.ops.moe import inkling2_moe_lowm_sm120 as lowm
from sglang.kernels.ops.moe.inkling2_route import route_metadata, route_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="Requires an SM12x GPU",
)

EXPERTS, TOP_K, ROUTE_SCALE = 128, 8, 8.0


def _bitwise(actual, expected):
    return torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


# Three split-K groups pin the order the partials are summed in.
@pytest.mark.parametrize("groups,with_bias", [(1, True), (3, False)])
def test_low_m_route_matches_route_metadata(groups, with_bias):
    generator = torch.Generator(device="cuda").manual_seed(groups)
    one = torch.ones((), device="cuda")
    bias = (
        torch.randn(EXPERTS, device="cuda", generator=generator) * 0.1
        if with_bias
        else None
    )
    for tokens in range(1, lowm.LOWM_MAX_TOKENS + 1):
        partials = torch.randn(
            groups, tokens, EXPERTS, device="cuda", generator=generator
        )
        # Equal logits on experts e and e + 64: without a bias every selection
        # is a tie, which must break toward the lower expert id on both sides.
        partials[:, :, 64:] = partials[:, :, :64]
        _, ids, weights = route_topk(partials, bias, TOP_K, ROUTE_SCALE, one)
        src2dst, dst2src, gammas, _, offsets = route_metadata(
            ids.flatten(), weights.flatten(), 0, EXPERTS, EXPERTS
        )[:5]
        maps = (src2dst, dst2src, gammas, offsets)
        sorted_only = lowm.route_metadata_lowm(ids, weights, EXPERTS)
        for actual, expected in zip(sorted_only, maps, strict=True):
            assert _bitwise(actual, expected), tokens
        if tokens <= lowm.LOWM_FUSED_ROUTE_MAX_TOKENS:
            fused = lowm.route_topk_meta_lowm(partials, bias, ROUTE_SCALE, one)
            for actual, expected in zip(fused, (ids, weights, *maps), strict=True):
                assert _bitwise(actual, expected), tokens


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
