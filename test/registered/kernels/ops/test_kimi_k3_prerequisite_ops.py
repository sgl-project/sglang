"""Representative parity coverage for the lightweight Kimi-K3 prerequisites."""

import unittest

import torch

from sglang.kernels.ops.attention.vision_rope import (
    apply_fused_qk_complex_rope,
)
from sglang.kernels.ops.elementwise import add3
from sglang.kernels.ops.gemm.tiny_gemm import (
    tiny_k_gemm_bf16,
    tiny_n_gemm_bf16,
)
from sglang.kernels.ops.mm.process.image import (
    _normalize_and_patchify_torch,
    normalize_and_patchify,
)
from sglang.kernels.ops.moe import moe_route_quant_fused
from sglang.kernels.ops.moe.moe_align_single_token import (
    moe_align_single_token,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.kernels.ops.moe.moe_route_radix import route_radix
from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds
from sglang.kernels.ops.quantization.per_token_group_quant import (
    per_token_group_quant,
)
from sglang.kernels.ops.sampling.top_p_renorm_triton import (
    top_p_renorm_probs_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896
TOPK = 16


class TestKimiK3PrerequisiteOps(CustomTestCase):
    def test_add3_bit_exact(self):
        torch.manual_seed(0)
        tensors = [
            torch.randn(9, 112, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        actual = add3.add3(*tensors, prefetch_bc=True)
        expected = (tensors[0] + tensors[1]) + tensors[2]
        self.assertTrue(torch.equal(actual, expected))

    def test_moe_auxiliary_kernels(self):
        expert_ids = torch.randperm(NUM_EXPERTS, device="cuda")[:TOPK].to(torch.int32)
        sorted_ids, sorted_experts, num_post = moe_align_single_token(
            expert_ids.unsqueeze(0), 64
        )
        order = torch.argsort(expert_ids)
        expected_ids = torch.full((TOPK * 64,), TOPK, dtype=torch.int32, device="cuda")
        expected_ids[::64] = order.to(torch.int32)
        self.assertTrue(torch.equal(sorted_experts, expert_ids[order]))
        self.assertTrue(torch.equal(sorted_ids, expected_ids))
        self.assertEqual(num_post.item(), TOPK * 64)

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
        ref_weights, ref_ids = moe_fused_gate(
            scores.float(),
            bias,
            topk=TOPK,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=True,
        )
        order = ids.argsort(dim=-1)
        ref_order = ref_ids.argsort(dim=-1)
        self.assertTrue(
            torch.equal(
                ids.gather(1, order),
                ref_ids.to(torch.int32).gather(1, ref_order),
            )
        )
        torch.testing.assert_close(
            weights.gather(1, order),
            ref_weights.gather(1, ref_order),
            rtol=1e-6,
            atol=0,
        )

        if not moe_route_quant_fused.available():
            self.skipTest("fused route+quant kernel unavailable")
        hidden = torch.randn(8, 3584, device="cuda", dtype=torch.bfloat16)
        ref_weights, ref_ids = route_radix(*args, sorted=False)
        ref_packed = PackTopkIds.execute(ref_ids, ref_weights)
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

    def test_tiny_gemm_variants(self):
        torch.manual_seed(2)
        x = torch.randn(2, 7168, device="cuda", dtype=torch.bfloat16) / 8
        weight = torch.randn(144, 7168, device="cuda", dtype=torch.bfloat16) / 8
        actual = tiny_n_gemm_bf16(x, weight, out_dtype=torch.float32)
        torch.testing.assert_close(
            actual.double(), x.double() @ weight.double().t(), rtol=1e-3, atol=1e-3
        )

        x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16) / 4
        weight = torch.randn(1536, 128, device="cuda", dtype=torch.bfloat16) / 4
        actual = tiny_k_gemm_bf16(x, weight)
        torch.testing.assert_close(
            actual.double(), x.double() @ weight.double().t(), rtol=2e-2, atol=2e-2
        )

    def test_top_p_renorm(self):
        torch.manual_seed(3)
        probs = torch.randn(3, 1024, device="cuda").softmax(-1)
        top_p = torch.tensor([0.5, 0.8, 0.95], device="cuda")
        sorted_probs = probs.sort(-1).values
        cutoff = torch.searchsorted(
            sorted_probs.cumsum(-1), (1 - top_p).unsqueeze(1)
        ).squeeze(1)
        cutoff.clamp_(max=probs.shape[1] - 1)
        pivot = sorted_probs.gather(1, cutoff[:, None])
        expected = torch.where(probs >= pivot, probs, 0)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(
            top_p_renorm_probs_triton(probs, top_p),
            expected,
            rtol=2e-6,
            atol=1e-8,
        )

    def test_vision_rope(self):
        torch.manual_seed(4)
        qkv = torch.randn(480, 3, 12, 128, device="cuda", dtype=torch.bfloat16)
        q, k, _ = qkv.unbind(1)
        angles = torch.randn(480, 64, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        freqs_expanded = freqs.unsqueeze(-2)

        def reference(x):
            value = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
            return torch.view_as_real(value * freqs_expanded).flatten(-2).type_as(x)

        actual_q, actual_k = apply_fused_qk_complex_rope(q, k, freqs)
        atol = 2 * torch.finfo(torch.bfloat16).eps
        torch.testing.assert_close(actual_q, reference(q), rtol=0, atol=atol)
        torch.testing.assert_close(actual_k, reference(k), rtol=0, atol=atol)

    def test_normalize_and_patchify(self):
        torch.manual_seed(5)
        image = torch.randn(2, 3, 17, 19, device="cuda")
        scale = torch.randn(1, 3, 1, 1, device="cuda")
        bias = torch.randn(1, 3, 1, 1, device="cuda")
        args = (image, scale, bias, 4, 20, 20)
        actual = normalize_and_patchify(
            args[0],
            args[1],
            args[2],
            patch_size=args[3],
            padded_height=args[4],
            padded_width=args[5],
        )
        expected = _normalize_and_patchify_torch(
            args[0],
            args[1],
            args[2],
            patch_size=args[3],
            padded_height=args[4],
            padded_width=args[5],
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
