"""Rank-local packed ids give FlashInfer's routed MXFP4 MoE the same output.

Under EP the fused Kimi-K3 route+quant launch can write the rank's experts as
0..count-1 and any other expert as count + slot, and the runner then passes
num_experts = count + top_k with offset 0. That only changes which routing-kernel
tier FlashInfer runs; the (token, local expert, weight) entries reaching the
GEMMs, and so the output, must stay bitwise equal to the global-id call. Needs
SM100 for the trtllm-gen MoE; weights are random bytes, which is enough to
compare two routings of the same layer.
"""

import unittest

import torch

from sglang.kernels.ops.moe import moe_route_quant_fused
from sglang.srt.layers.moe.route_quant_handoff import rank_local_num_experts
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

NUM_EXPERTS = 896
TOPK = 16
HIDDEN = 3584
INTERMEDIATE = 384


class TestRankLocalRoutedMoe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("trtllm-gen routed MoE requires SM100+")
        if not moe_route_quant_fused.available():
            raise unittest.SkipTest("fused route+quant kernel unavailable")

    def test_routed_moe_output_unchanged(self):
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer.tllm_enums import RoutingMethodType

        local_entries = 0
        for ep_size in (2, 4, 8, 16):
            count = NUM_EXPERTS // ep_size
            g = torch.Generator(device="cuda").manual_seed(ep_size)

            def rand_bytes(*shape, low=0, high=256):
                return torch.randint(
                    low, high, shape, device="cuda", dtype=torch.uint8, generator=g
                )

            # e2m1 weights are any byte; UE8M0 scales 2^-10..2^-6 keep it finite
            w13 = rand_bytes(count, 2 * INTERMEDIATE, HIDDEN // 2)
            w13_scale = rand_bytes(
                count, 2 * INTERMEDIATE, HIDDEN // 32, low=117, high=122
            )
            w2 = rand_bytes(count, HIDDEN, INTERMEDIATE // 2)
            w2_scale = rand_bytes(count, HIDDEN, INTERMEDIATE // 32, low=117, high=122)

            def moe(packed, x_q, x_s, num_experts, local_expert_offset):
                out = torch.empty(
                    packed.shape[0], HIDDEN, device="cuda", dtype=torch.bfloat16
                )
                trtllm_fp4_block_scale_routed_moe(
                    topk_ids=packed,
                    routing_bias=None,
                    hidden_states=x_q,
                    hidden_states_scale=x_s.view(torch.float8_e4m3fn),
                    gemm1_weights=w13,
                    gemm1_weights_scale=w13_scale.view(torch.float8_e4m3fn),
                    gemm1_bias=None,
                    gemm1_alpha=None,
                    gemm1_beta=None,
                    gemm1_clamp_limit=None,
                    gemm2_weights=w2,
                    gemm2_weights_scale=w2_scale.view(torch.float8_e4m3fn),
                    gemm2_bias=None,
                    output1_scale_scalar=None,
                    output1_scale_gate_scalar=None,
                    output2_scale_scalar=None,
                    num_experts=num_experts,
                    top_k=TOPK,
                    n_group=None,
                    topk_group=None,
                    intermediate_size=INTERMEDIATE,
                    local_expert_offset=local_expert_offset,
                    local_num_experts=count,
                    routed_scaling_factor=None,
                    routing_method_type=RoutingMethodType.TopK.value,
                    output=out,
                )
                return out

            for rank in sorted({0, ep_size // 2, ep_size - 1}):
                offset = rank * count
                # FlashInfer routing: block kernel up to 4 tokens, dyn-block or
                # cluster up to 16, cluster above; 64 is the fused launch's cap
                for num_tokens in (1, 4, 5, 8, 16, 17, 24, 64):
                    torch.manual_seed(num_tokens + rank)
                    scores = torch.randn(
                        num_tokens, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16
                    )
                    bias = torch.randn(NUM_EXPERTS, device="cuda")
                    x = torch.randn(
                        num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16
                    )
                    args = (scores, bias, x, TOPK)
                    kwargs = dict(
                        renormalize=True, routed_scaling_factor=2.5, apply_scale=False
                    )
                    _, ids, packed, x_q, x_s = moe_route_quant_fused.route_quant_fused(
                        *args, **kwargs
                    )
                    local_packed = moe_route_quant_fused.route_quant_fused(
                        *args,
                        **kwargs,
                        local_expert_offset=offset,
                        num_local_experts=count,
                    )[2]
                    ref = moe(packed, x_q, x_s, NUM_EXPERTS, offset)
                    out = moe(
                        local_packed,
                        x_q,
                        x_s,
                        rank_local_num_experts(offset, count, NUM_EXPERTS, TOPK),
                        0,
                    )
                    tag = f"ep={ep_size} rank={rank} m={num_tokens}"
                    self.assertTrue(bool(torch.isfinite(ref.float()).all()), msg=tag)
                    self.assertTrue(torch.equal(out, ref), msg=tag)
                    local_entries += int(
                        ((ids >= offset) & (ids < offset + count)).sum()
                    )
        # not vacuous: rows did reach the rank's experts
        self.assertGreater(local_entries, 0)


if __name__ == "__main__":
    unittest.main()
