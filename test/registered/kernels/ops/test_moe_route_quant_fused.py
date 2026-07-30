"""Fused route+pack+quant vs the unfused chain, bit-exact.

The fused kernel (kernels/ops/moe/moe_route_quant_fused.py) must reproduce all
three of its constituents exactly: route_radix (weights, ids), the triton
(id << 16 | bf16(weight)) pack, and per_token_group_quant (fp8 rows + packed
UE8M0 group-32 scales) — including on row-strided activation views (the K3
fused-front split) and the route kernel's tie/NaN adversarial cases.
"""

import pytest
import torch

from sglang.kernels.ops.moe import moe_route_quant_fused
from sglang.kernels.ops.moe.moe_route_radix import covered as route_covered
from sglang.kernels.ops.moe.moe_route_radix import route_radix
from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds
from sglang.kernels.ops.quantization.per_token_group_quant import (
    per_token_group_quant,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896
TOPK = 16
HIDDEN = 3584


def _make_scores(case: str, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.randn(num_tokens, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    if case == "random":
        pass
    elif case == "all_equal":
        # Full 896-way key tie: min-id tie-break must pick experts 0..15.
        scores.fill_(0.5)
        bias.zero_()
    elif case == "nan_mixed":
        scores[:, ::3] = float("nan")
    else:
        raise AssertionError(case)
    return scores, bias


def _make_x(num_tokens: int, strided: bool) -> torch.Tensor:
    if not strided:
        return torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    # Row-strided view with 32B-aligned rows, like the K3 fused-front split
    # ([gate_up | router | latent] slices of one GEMM output).
    full = torch.randn(num_tokens, HIDDEN + 1024, dtype=torch.bfloat16, device="cuda")
    x = full[:, 512 : 512 + HIDDEN]
    assert x.data_ptr() % 32 == 0 and (x.stride(0) * x.element_size()) % 32 == 0
    return x


@pytest.mark.parametrize("num_tokens", [1, 2, 8, 64])
@pytest.mark.parametrize("case", ["random", "all_equal", "nan_mixed"])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_matches_unfused_chain(num_tokens: int, case: str, strided: bool):
    if not moe_route_quant_fused.available():
        pytest.skip("JIT fused route+quant kernel unavailable")
    torch.manual_seed(0x5EED + num_tokens)
    scores, bias = _make_scores(case, num_tokens)
    x = _make_x(num_tokens, strided)
    routed_scaling_factor = 2.446
    assert route_covered(scores, bias, TOPK)
    assert moe_route_quant_fused.covered(scores, bias, TOPK, x)

    ref_w, ref_i = route_radix(
        scores,
        bias,
        TOPK,
        renormalize=True,
        routed_scaling_factor=routed_scaling_factor,
        apply_scale=True,
        sorted=False,
    )
    ref_packed = PackTopkIds.execute(ref_i, ref_w)
    ref_q, ref_s = per_token_group_quant(x, group_size=32, scale_ue8m0=True)

    w, i, packed, x_q, x_s = moe_route_quant_fused.route_quant_fused(
        scores,
        bias,
        x,
        TOPK,
        renormalize=True,
        routed_scaling_factor=routed_scaling_factor,
        apply_scale=True,
    )

    # NaN-selected weights (raw sigmoid of NaN) compare by bit pattern.
    torch.testing.assert_close(i, ref_i, rtol=0, atol=0)
    assert torch.equal(w.view(torch.int32), ref_w.view(torch.int32))
    torch.testing.assert_close(packed, ref_packed, rtol=0, atol=0)
    assert torch.equal(x_q.view(torch.uint8), ref_q.view(torch.uint8))
    torch.testing.assert_close(x_s, ref_s, rtol=0, atol=0)


def test_covered_rejects():
    if not moe_route_quant_fused.available():
        pytest.skip("JIT fused route+quant kernel unavailable")
    scores = torch.randn(2, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    ok = torch.randn(2, HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert moe_route_quant_fused.covered(scores, bias, TOPK, ok)
    # wrong hidden width
    assert not moe_route_quant_fused.covered(
        scores, bias, TOPK, torch.randn(2, 4096, dtype=torch.bfloat16, device="cuda")
    )
    # token-count mismatch
    assert not moe_route_quant_fused.covered(
        scores, bias, TOPK, torch.randn(3, HIDDEN, dtype=torch.bfloat16, device="cuda")
    )
    # misaligned row start (offset 1 element = 2B)
    full = torch.randn(2, HIDDEN + 32, dtype=torch.bfloat16, device="cuda")
    assert not moe_route_quant_fused.covered(
        scores, bias, TOPK, full[:, 1 : 1 + HIDDEN]
    )
    # over the batch cap
    big_scores = torch.randn(65, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    assert not moe_route_quant_fused.covered(
        big_scores,
        bias,
        TOPK,
        torch.randn(65, HIDDEN, dtype=torch.bfloat16, device="cuda"),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
