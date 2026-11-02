"""Expert routing, packed output, and quantization parity."""

import unittest

import torch

from sglang.kernels.ops.moe import moe_route_quant_fused
from sglang.kernels.ops.moe.moe_route_radix import route_radix
from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
from sglang.kernels.ops.quantization.per_token_group_quant import (
    per_token_group_quant,
)
from sglang.srt.layers.moe import route_quant_handoff
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896

TOPK = 16


def _pack_topk_oracle(topk_ids, topk_weights):
    """Pure-torch reference for the packed ids the fused route+quant kernel emits.

    FlashInfer's routed MoE reads one int32 per entry: the expert id in the high
    half and the bf16 weight bits in the low half.
    """
    weight_bits = (
        topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    )
    return (topk_ids.to(torch.int32) << 16) | weight_bits


def _rank_local_oracle(packed, offset, count):
    """Rank-local packed ids: experts [offset, offset + count) become 0..count-1,
    any other expert in top-k slot k becomes count + k; weight bits unchanged."""
    ids = packed >> 16
    slot = torch.arange(packed.shape[1], device=packed.device, dtype=torch.int32)
    is_local = (ids >= offset) & (ids < offset + count)
    local_ids = torch.where(is_local, ids - offset, count + slot)
    return (local_ids << 16) | (packed & 0xFFFF)


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


class TestRouteQuant(CustomTestCase):
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

    def test_rank_local_packed_ids(self):
        """Under EP the fused launch can write rank-local packed ids so FlashInfer
        routes count + top_k experts instead of 896; only the packed ids change,
        and they must stay unique per row and below count + top_k."""
        if not moe_route_quant_fused.available():
            self.skipTest("fused route+quant kernel unavailable")
        for ep_size in (2, 8, 16):
            count = NUM_EXPERTS // ep_size
            for rank in sorted({0, ep_size // 2, ep_size - 1}):
                # 64 is the fused launch's token cap
                for num_tokens in (1, 5, 17, 64):
                    torch.manual_seed(num_tokens + rank)
                    scores = torch.randn(
                        num_tokens, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16
                    )
                    bias = torch.randn(NUM_EXPERTS, device="cuda")
                    hidden = torch.randn(
                        num_tokens, 3584, device="cuda", dtype=torch.bfloat16
                    )
                    args = (scores, bias, hidden, TOPK)
                    kwargs = dict(
                        renormalize=True, routed_scaling_factor=2.5, apply_scale=True
                    )
                    ref = moe_route_quant_fused.route_quant_fused(*args, **kwargs)
                    out = moe_route_quant_fused.route_quant_fused(
                        *args,
                        **kwargs,
                        local_expert_offset=rank * count,
                        num_local_experts=count,
                    )
                    tag = f"ep={ep_size} rank={rank} m={num_tokens}"
                    for i in (0, 1, 3, 4):
                        self.assertTrue(
                            torch.equal(
                                out[i].view(torch.uint8), ref[i].view(torch.uint8)
                            ),
                            msg=tag,
                        )
                    self.assertTrue(
                        torch.equal(
                            out[2], _rank_local_oracle(ref[2], rank * count, count)
                        ),
                        msg=tag,
                    )
                    local_ids = (out[2] >> 16).sort(dim=-1).values
                    self.assertTrue(bool((local_ids[:, 1:] != local_ids[:, :-1]).all()))
                    self.assertLess(int(local_ids.max()), count + TOPK, msg=tag)

    def test_handoff_rank_local_layout(self):
        """The runner must learn which range the packed ids it took were made
        rank-local for (it changes num_experts and the offset), exactly once and
        only for that tensor; a range that cannot shrink routing keeps global ids."""
        if not moe_route_quant_fused.available():
            self.skipTest("fused route+quant kernel unavailable")
        torch.manual_seed(2)
        scores = torch.randn(8, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(NUM_EXPERTS, device="cuda")
        hidden = torch.randn(8, 3584, device="cuda", dtype=torch.bfloat16)

        def route_and_take(local_experts):
            route_quant_handoff.stage(hidden, local_experts=local_experts)
            hit = route_quant_handoff.try_route_quant_fused(
                scores,
                bias,
                TOPK,
                num_fused_shared_experts=0,
                renormalize=True,
                routed_scaling_factor=2.5,
                apply_routed_scaling_factor_on_output=False,
            )
            self.assertIsNotNone(hit)
            return route_quant_handoff.take(hidden)[0]

        try:
            ref_packed = route_and_take(None)
            self.assertIsNone(route_quant_handoff.take_local_experts(ref_packed))
            for staged, expected in (
                ((336, 112), (336, 112)),
                ((0, NUM_EXPERTS), None),  # every expert local: nothing to shrink
                ((840, 112), None),  # past the routed experts
            ):
                packed = route_and_take(staged)
                self.assertIsNone(
                    route_quant_handoff.take_local_experts(packed.clone())
                )
                packed = route_and_take(staged)
                self.assertEqual(
                    route_quant_handoff.take_local_experts(packed), expected
                )
                self.assertIsNone(route_quant_handoff.take_local_experts(packed))
                want = (
                    ref_packed
                    if expected is None
                    else (_rank_local_oracle(ref_packed, *expected))
                )
                self.assertTrue(torch.equal(packed, want), msg=str(staged))
        finally:
            route_quant_handoff.clear()
        # FlashInfer's routing requires num_experts % 4 == 0
        rank_local = route_quant_handoff.rank_local_num_experts
        self.assertEqual(rank_local(336, 112, NUM_EXPERTS, TOPK), 128)
        self.assertEqual(rank_local(0, 7, 64, 6), 16)
        self.assertIsNone(rank_local(0, 56, 72, 16))  # 72 does not shrink

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
