"""Validate the opt-in SM90 block-FP8 down-GEMM 128+32 K decomposition."""

import unittest
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe import fused_moe_triton_kernels as kernels
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(is_sm90_supported(), "The K=160 specialization targets SM90")
class TestMoEFp8K160Tail(CustomTestCase):
    def _case(
        self,
        m,
        topk,
        bn,
        *,
        k=160,
        n=257,
        skewed=False,
        filtered=False,
        tail_only=False,
    ):
        torch.manual_seed(314)
        e = 17
        ids = torch.rand((m, e), device="cuda").topk(topk, dim=-1).indices.int()
        if skewed:
            ids[:] = torch.arange(topk, device="cuda", dtype=torch.int32)
        if filtered:
            ids[:, 0] = -1
        weights = torch.rand((m, topk), device="cuda")
        weights /= weights.sum(dim=-1, keepdim=True)
        a = torch.randn((m * topk, k), device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        b = torch.randn((e, n, k), device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        if tail_only:
            # Only the final 32 lanes contribute, including the very last K
            # element. This catches a dropped/shifted tail group or scale.
            a_float = a.float()
            a_float[:, :128] = 0
            a_float[:, -1] = 448
            a = a_float.to(torch.float8_e4m3fn)
        a_scale = (
            torch.rand((m * topk, triton.cdiv(k, 128)), device="cuda") * 0.1 + 0.01
        )
        b_scale = (
            torch.rand((e, triton.cdiv(n, 128), triton.cdiv(k, 128)), device="cuda")
            * 0.1
            + 0.01
        )
        output = torch.full(
            (m, topk, n), float("nan"), device="cuda", dtype=torch.bfloat16
        )
        sorted_ids, expert_ids, padded = moe_align_block_size(ids, 64, e)
        base = dict(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=32,
            num_warps=4,
            num_stages=2,
        )
        common = dict(
            A=a,
            B=b,
            bias=None,
            C=output,
            A_scale=a_scale,
            B_scale=b_scale,
            B_zp=None,
            topk_weights=weights,
            topk_ids=ids,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=padded,
            mul_routed_weight=True,
            top_k=1,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=[128, 128],
            filter_expert=filtered,
        )
        kernels.invoke_fused_moe_kernel(config=base, **common)
        reference = output.clone()
        candidate = dict(base, USE_FP8_K160_TAIL=True)
        saved_config = dict(candidate)
        output.fill_(float("nan"))
        kernels.invoke_fused_moe_kernel(config=candidate, **common)
        self.assertEqual(candidate, saved_config, "Do not mutate cached tuning configs")
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(
            torch.equal(output.view(torch.int16), reference.view(torch.int16))
        )
        with patch.object(kernels, "fused_moe_kernel") as launch:
            kernels.invoke_fused_moe_kernel(config=candidate, **common)
            kwargs = launch.__getitem__.return_value.call_args.kwargs
            self.assertEqual(kwargs["USE_FP8_K160_TAIL"], k == 160 and bn <= 128)

        # Independent sampled FP32 reference, with the original groupwise
        # rounding/scaling boundary and a ragged output N scale group.
        rows = torch.linspace(0, m * topk - 1, min(16, m * topk), device="cuda").long()
        rid = ids.flatten()[rows].long()
        valid = rid >= 0
        rid = rid.clamp_min(0)
        aq = a.float()[rows]
        bq = b[rid].float()
        expected = torch.zeros((rows.numel(), n), device="cuda")
        tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            for group, left in enumerate(range(0, k, 128)):
                right = min(left + 128, k)
                dot = torch.bmm(bq[:, :, left:right], aq[:, left:right, None]).squeeze(
                    -1
                )
                scale = (
                    a_scale[rows, group, None]
                    * b_scale[rid, :, group].repeat_interleave(128, dim=-1)[:, :n]
                )
                expected += dot * scale
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        expected *= weights.flatten()[rows, None]
        expected[~valid] = 0
        torch.testing.assert_close(
            output.view(-1, n)[rows], expected.to(torch.bfloat16), rtol=0.016, atol=2e-4
        )

    def test_tail_mask_routing_and_scale_boundaries(self):
        for m, topk, bn, skewed, filtered in (
            (1, 8, 32, False, False),
            (63, 9, 64, False, False),
            (64, 8, 128, False, False),
            (65, 9, 64, True, False),
            (129, 8, 64, True, True),
        ):
            with self.subTest(m=m, topk=topk, bn=bn, skewed=skewed, filtered=filtered):
                self._case(m, topk, bn, skewed=skewed, filtered=filtered)

    def test_ineligible_shapes_retain_generic_kernel(self):
        for k, bn in ((128, 64), (192, 64), (160, 256)):
            with self.subTest(k=k, bn=bn):
                self._case(65, 8, bn, k=k)

    def test_tail_only_and_large_last_lane(self):
        self._case(65, 9, 64, tail_only=True)


if __name__ == "__main__":
    unittest.main()
