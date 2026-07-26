import unittest

import torch
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrequantizedFusedMoe(unittest.TestCase):
    def test_prequantized_a_matches_internal_quantization(self):
        """Shared FP8 rows must preserve the default quantize-and-run result."""
        torch.manual_seed(7)
        m, n, k, experts, topk = 16, 128, 128, 1, 1
        block_shape = [128, 128]
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }

        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.1
        b = (torch.randn((experts, n, k), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        b_scale = torch.ones((experts, 1, 1), device="cuda", dtype=torch.float32)
        topk_ids = torch.zeros((m, topk), device="cuda", dtype=torch.int64)
        topk_weights = torch.ones((m, topk), device="cuda", dtype=torch.float32)
        sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], experts
        )
        expected = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        actual = torch.empty_like(expected)

        common = dict(
            B=b,
            bias=None,
            A_scale=None,
            B_scale=b_scale,
            B_zp=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=topk,
            config=config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=block_shape,
            filter_expert=False,
        )
        invoke_fused_moe_kernel(A=a, C=expected, **common)

        a_fp8, a_scale = sglang_per_token_group_quant_fp8(a, block_shape[1])
        invoke_fused_moe_kernel(
            A=a_fp8,
            C=actual,
            **{**common, "A_scale": a_scale},
            a_is_prequantized=True,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_glm52_and_dsv4_w13_shapes_accept_row_strided_shared_input(self):
        """One Triton entry must handle both model shapes and shared row strides."""
        for model_name, hidden_size in (
            ("glm52", 6144),
            ("dsv4_flash", 4096),
        ):
            with self.subTest(model=model_name):
                self._assert_row_strided_w13_matches_contiguous(hidden_size)

    def _assert_row_strided_w13_matches_contiguous(self, hidden_size: int):
        torch.manual_seed(hidden_size)
        m, n, experts, topk = 16, 4096, 1, 1
        block_shape = [128, 128]
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 3,
        }
        a = torch.randn(
            (m, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(0.1)
        b = (
            torch.randn(
                (experts, n, hidden_size),
                device="cuda",
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        b_scale = torch.ones(
            (
                experts,
                n // block_shape[0],
                hidden_size // block_shape[1],
            ),
            device="cuda",
            dtype=torch.float32,
        )
        topk_ids = torch.zeros((m, topk), device="cuda", dtype=torch.int64)
        topk_weights = torch.ones(
            (m, topk),
            device="cuda",
            dtype=torch.float32,
        )
        sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            config["BLOCK_SIZE_M"],
            experts,
        )
        expected = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        actual = torch.empty_like(expected)
        common = dict(
            B=b,
            bias=None,
            B_scale=b_scale,
            B_zp=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=topk,
            config=config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=block_shape,
            filter_expert=False,
        )
        invoke_fused_moe_kernel(
            A=a,
            A_scale=None,
            C=expected,
            **common,
        )

        a_fp8, a_scale = sglang_per_token_group_quant_fp8(a, block_shape[1])
        shared_rows = torch.empty(
            (m, 64 * 1024),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        shared_scales = torch.empty(
            (m, 64 * 1024 // torch.float32.itemsize),
            dtype=torch.float32,
            device="cuda",
        )
        shared_a = shared_rows[:, :hidden_size]
        shared_a_scale = shared_scales[:, : hidden_size // block_shape[1]]
        shared_a.copy_(a_fp8)
        shared_a_scale.copy_(a_scale)
        self.assertEqual(shared_a.stride(), (64 * 1024, 1))
        self.assertEqual(
            shared_a_scale.stride(),
            (64 * 1024 // torch.float32.itemsize, 1),
        )

        invoke_fused_moe_kernel(
            A=shared_a,
            A_scale=shared_a_scale,
            C=actual,
            **common,
            a_is_prequantized=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
