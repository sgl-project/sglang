"""Representative parity coverage for the lightweight Kimi-K3 prerequisites."""

import unittest

import torch

from sglang.kernels.ops.moe import moe_route_quant_fused
from sglang.kernels.ops.moe.moe_route_radix import route_radix
from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
from sglang.kernels.ops.quantization.per_token_group_quant import (
    per_token_group_quant,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896

TOPK = 16

NOPE_DIM = 512

ROPE_DIM = 64

MLA_DIM = NOPE_DIM + ROPE_DIM

MLA_PAGES = 256


def _pack_topk_oracle(topk_ids, topk_weights):
    """Pure-torch reference for the packed ids the fused route+quant kernel emits.

    FlashInfer's routed MoE reads one int32 per entry: the expert id in the high
    half and the bf16 weight bits in the low half.
    """
    weight_bits = (
        topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    )
    return (topk_ids.to(torch.int32) << 16) | weight_bits


def _route_oracle(
    scores, bias, topk, renormalize, routed_scaling_factor, apply_scale, sorted
):
    """Pure-torch fp32 reference for route_radix.

    Deliberately independent of moe_fused_gate: that entry dispatches back to
    route_radix whenever scoring is sigmoid with no shared experts, no expert
    groups and no softcapping (moe_fused_gate.py, the covered() fast path), which
    is exactly the configuration under test.

    Contract, from route_radix.cuh: bias participates in RANKING only and the
    emitted weight stays bias-free; NaN is floored so it can never win; ties go to
    the lower expert id; renormalize divides by the winners' sum (guarded to 1 when
    that sum is non-positive) and only then is routed scaling applied; sorted=True
    emits (biased desc, id asc) while sorted=False emits ascending expert id.
    """
    s = torch.sigmoid(scores.float())
    biased = s + bias.float()
    biased = torch.where(torch.isnan(biased), torch.full_like(biased, -1e30), biased)
    # stable + descending: equal biased values keep ascending-id order
    ranked = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :topk]
    w = s.gather(1, ranked)
    total = w.sum(-1, keepdim=True)
    norm = torch.where(total > 0, total, torch.ones_like(total))
    if renormalize:
        w = w / norm
    if apply_scale:
        w = w * routed_scaling_factor
    if sorted:
        return w, ranked.to(torch.int32)
    by_id = ranked.argsort(dim=-1)
    return w.gather(1, by_id), ranked.gather(1, by_id).to(torch.int32)


class TestKimiK3PrerequisiteOps(CustomTestCase):
    def test_moe_auxiliary_kernels(self):
        x = torch.randn(2, TOPK, 7168, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(2, 7168, device="cuda", dtype=torch.bfloat16)
        self.assertIs(moe_topk_sum(x, out), out)
        self.assertTrue(torch.equal(out, x.float().sum(1).to(torch.bfloat16)))

    def test_moe_route_and_quant(self):
        torch.manual_seed(1)
        scores = torch.randn(8, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.float32)
        args = (scores, bias, TOPK, True, 2.5, True)
        weights, ids = route_radix(*args, sorted=True)
        # Oracle, NOT moe_fused_gate: for this exact configuration (sigmoid, no
        # shared experts, no expert groups, no softcapping) moe_fused_gate
        # dispatches straight back to route_radix, so using it as the reference
        # compares the kernel with itself and cannot see a selection, tie-break,
        # NaN, renormalize or scaling error.
        ref_weights, ref_ids = _route_oracle(*args, sorted=True)
        self.assertTrue(torch.equal(ids, ref_ids))
        # rtol is not 1e-6: the kernel computes sigmoid with __fdividef/__expf,
        # whose last bits differ from torch's. The old self-comparison could
        # afford atol=0; a real oracle cannot.
        torch.testing.assert_close(weights, ref_weights, rtol=1e-5, atol=1e-6)

        if not moe_route_quant_fused.available():
            self.skipTest("fused route+quant kernel unavailable")
        hidden = torch.randn(8, 3584, device="cuda", dtype=torch.bfloat16)
        ref_weights, ref_ids = route_radix(*args, sorted=False)
        ref_packed = _pack_topk_oracle(ref_ids, ref_weights)
        ref_q, ref_scale = per_token_group_quant(
            hidden, group_size=32, scale_ue8m0=True
        )
        actual = moe_route_quant_fused.route_quant_fused(
            scores,
            bias,
            hidden,
            TOPK,
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_scale=True,
        )
        weights, ids, packed, quantized, scale = actual
        self.assertTrue(torch.equal(ids, ref_ids))
        self.assertTrue(
            torch.equal(weights.view(torch.int32), ref_weights.view(torch.int32))
        )
        self.assertTrue(torch.equal(packed, ref_packed))
        self.assertTrue(
            torch.equal(quantized.view(torch.uint8), ref_q.view(torch.uint8))
        )
        torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)

    def test_route_radix_ties_and_nan(self):
        """The cases the self-comparison could not see.

        Exact ties: many experts share one biased value, so the winner set is only
        determined by the lowest-id rule. NaN: floored, so a NaN expert must never
        be selected while enough finite ones exist. Both run with renormalize and
        scaling on and off, since those are applied in a fixed order.
        """
        bias = torch.zeros(NUM_EXPERTS, device="cuda", dtype=torch.float32)

        tied = torch.full((4, NUM_EXPERTS), 0.25, device="cuda", dtype=torch.bfloat16)
        # a handful of strict winners above the tied plateau, the rest exactly equal
        tied[:, 300] = 2.0
        tied[:, 7] = 2.0
        tied[:, 800] = 1.5

        nan_scores = torch.randn(4, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
        nan_scores[:, 100] = float("nan")
        nan_scores[:, 500] = float("nan")
        # make the NaN experts the ones that would otherwise win outright
        nan_scores[:, 101] = 5.0

        for name, scores in (("ties", tied), ("nan", nan_scores)):
            for renormalize in (False, True):
                for apply_scale in (False, True):
                    for sorted_ in (False, True):
                        args = (scores, bias, TOPK, renormalize, 2.5, apply_scale)
                        ids = route_radix(*args, sorted=sorted_)[1]
                        ref_ids = _route_oracle(*args, sorted=sorted_)[1]
                        tag = (
                            f"{name} renorm={renormalize} "
                            f"scale={apply_scale} sorted={sorted_}"
                        )
                        self.assertTrue(torch.equal(ids, ref_ids), msg=tag)
                        if name == "nan":
                            self.assertFalse(
                                bool(((ids == 100) | (ids == 500)).any()),
                                msg=f"{tag}: a NaN expert was selected",
                            )


if __name__ == "__main__":
    unittest.main()
