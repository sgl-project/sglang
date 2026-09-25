"""Registration guardrails for GGUF I-quant MMQ kernels."""

import unittest
from unittest.mock import patch

import torch
from gguf import GGMLQuantizationType as WeightType

from sglang.srt.layers.quantization import gguf as gguf_quant
from sglang.srt.layers.quantization.gguf import (
    DEQUANT_TYPES,
    KQUANT_TYPES,
    MMQ_IMATRIX_QUANT_TYPES,
    MMQ_K_ALIGNMENTS,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    STANDARD_QUANT_TYPES,
    _build_mmq_quant_types,
    _is_iq_mmq_batch_size,
    _is_iq_moe_mmq_shape,
    _select_gguf_matmul_kernel,
    _should_use_moe_mmq,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


EXPECTED_MMQ_IMATRIX_TYPES = {
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_NL,
    WeightType.IQ4_XS,
}
CUDA_MMQ_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | EXPECTED_MMQ_IMATRIX_TYPES


class TestGGUFMMQTypes(CustomTestCase):
    def test_only_upstream_supported_iq_types_are_registered(self):
        self.assertEqual(MMQ_IMATRIX_QUANT_TYPES, EXPECTED_MMQ_IMATRIX_TYPES)
        self.assertNotIn(WeightType.IQ1_M, MMQ_IMATRIX_QUANT_TYPES)
        self.assertNotIn(WeightType.Q8_1, STANDARD_QUANT_TYPES)
        self.assertNotIn(WeightType.Q8_1, DEQUANT_TYPES)
        self.assertNotIn(WeightType.Q8_1, MMVQ_QUANT_TYPES)
        self.assertNotIn(WeightType.Q8_1, MMQ_QUANT_TYPES)

    def test_iq_mmq_requires_new_kernel_capability(self):
        without_capability = _build_mmq_quant_types(False)
        with_capability = _build_mmq_quant_types(True)
        self.assertTrue(EXPECTED_MMQ_IMATRIX_TYPES.isdisjoint(without_capability))
        self.assertTrue(EXPECTED_MMQ_IMATRIX_TYPES <= with_capability)

        if is_cuda() and gguf_quant._supports_iq_mmq:
            self.assertTrue(EXPECTED_MMQ_IMATRIX_TYPES <= MMQ_QUANT_TYPES)
        else:
            self.assertTrue(EXPECTED_MMQ_IMATRIX_TYPES.isdisjoint(MMQ_QUANT_TYPES))

    def test_old_kernel_capability_keeps_iq_fallbacks(self):
        without_capability = _build_mmq_quant_types(False)
        with patch.object(gguf_quant, "MMQ_QUANT_TYPES", without_capability):
            for qweight_type in EXPECTED_MMQ_IMATRIX_TYPES:
                self.assertEqual(
                    _select_gguf_matmul_kernel(9, 18944, qweight_type, 1024),
                    "dequantize",
                )
                self.assertEqual(
                    _select_gguf_matmul_kernel(32, 18944, qweight_type, 1024),
                    "dequantize",
                )
                self.assertFalse(
                    _should_use_moe_mmq(
                        qweight_type,
                        WeightType.Q5_0,
                        128,
                        256,
                        4,
                        1024,
                        512,
                    )
                )
                self.assertFalse(
                    _should_use_moe_mmq(
                        WeightType.Q5_0,
                        qweight_type,
                        128,
                        256,
                        4,
                        1024,
                        512,
                    )
                )

            self.assertEqual(
                _select_gguf_matmul_kernel(7, 18944, WeightType.Q5_0, 1024),
                "mmq",
            )
            self.assertTrue(
                _should_use_moe_mmq(
                    WeightType.Q5_0,
                    WeightType.Q2_K,
                    65,
                    8,
                    2,
                    1024,
                    512,
                )
            )

    def test_dense_kernel_selection(self):
        with patch.object(gguf_quant, "MMQ_QUANT_TYPES", CUDA_MMQ_TYPES):
            # Wide I-quant matrices use MMVQ through 8, MMQ through 16.
            self.assertEqual(
                _select_gguf_matmul_kernel(8, 5121, WeightType.IQ3_S, 1024),
                "mmvq",
            )
            self.assertEqual(
                _select_gguf_matmul_kernel(9, 5121, WeightType.IQ3_S, 1024),
                "mmq",
            )
            self.assertEqual(
                _select_gguf_matmul_kernel(16, 5121, WeightType.IQ3_S, 1024),
                "mmq",
            )
            self.assertEqual(
                _select_gguf_matmul_kernel(17, 5121, WeightType.IQ3_S, 1024),
                "dequantize",
            )

            # Narrow I-quant matrices stay on MMVQ through 16, then dequantize.
            self.assertEqual(
                _select_gguf_matmul_kernel(16, 5120, WeightType.IQ3_S, 1024),
                "mmvq",
            )
            self.assertEqual(
                _select_gguf_matmul_kernel(17, 5120, WeightType.IQ3_S, 1024),
                "dequantize",
            )
            self.assertEqual(
                _select_gguf_matmul_kernel(32, 5120, WeightType.IQ3_S, 1024),
                "dequantize",
            )
            self.assertEqual(
                _select_gguf_matmul_kernel(33, 5120, WeightType.IQ3_S, 1024),
                "dequantize",
            )

            # Existing standard and K-quant routing remains unchanged for
            # aligned model dimensions.
            self.assertEqual(
                _select_gguf_matmul_kernel(2, 5121, WeightType.Q5_0, 1024),
                "mmvq",
            )
            self.assertEqual(
                _select_gguf_matmul_kernel(3, 5121, WeightType.Q5_0, 1024),
                "mmq",
            )
            self.assertEqual(
                _select_gguf_matmul_kernel(64, 5121, WeightType.Q2_K, 1024),
                "mmq",
            )

            # Every MMQ loader consumes a complete K tile.
            for qweight_type, unaligned, aligned in (
                (WeightType.Q5_0, 32, 256),
                (WeightType.Q8_0, 96, 128),
                (WeightType.Q2_K, 256, 512),
                (WeightType.Q3_K, 256, 512),
                (WeightType.IQ4_NL, 96, 256),
            ):
                num_tokens = 9 if qweight_type in MMQ_IMATRIX_QUANT_TYPES else 7
                self.assertEqual(
                    _select_gguf_matmul_kernel(
                        num_tokens, 5121, qweight_type, unaligned
                    ),
                    "dequantize",
                )
                self.assertEqual(
                    _select_gguf_matmul_kernel(num_tokens, 5121, qweight_type, aligned),
                    "mmq",
                )

            # IQ1_M intentionally keeps the dequantization fallback.
            self.assertEqual(
                _select_gguf_matmul_kernel(8, 5121, WeightType.IQ1_M, 1024),
                "mmvq",
            )
            self.assertEqual(
                _select_gguf_matmul_kernel(9, 5121, WeightType.IQ1_M, 1024),
                "dequantize",
            )

            self.assertEqual(
                _select_gguf_matmul_kernel(1, 5121, WeightType.Q8_1, 32),
                "unsupported",
            )

            self.assertEqual(MMQ_K_ALIGNMENTS[WeightType.Q8_0], 128)
            self.assertEqual(MMQ_K_ALIGNMENTS[WeightType.Q2_K], 512)
            self.assertEqual(MMQ_K_ALIGNMENTS[WeightType.IQ4_NL], 256)

        self.assertTrue(_is_iq_mmq_batch_size(16, 5121))
        self.assertFalse(_is_iq_mmq_batch_size(17, 5121))
        self.assertTrue(_is_iq_mmq_batch_size(16, 5120))
        self.assertFalse(_is_iq_mmq_batch_size(17, 5120))

    def test_dense_selector_terminal_paths(self):
        self.assertEqual(
            _select_gguf_matmul_kernel(1, 1024, WeightType.F16, 1024),
            "unquantized",
        )
        self.assertEqual(_select_gguf_matmul_kernel(1, 1024, -1, 1024), "unsupported")

    def test_empty_dense_input_bypasses_selector(self):
        x = torch.empty((0, 4), dtype=torch.bfloat16)
        qweight = torch.empty((3, 4), dtype=torch.uint8)

        with patch.object(gguf_quant, "_select_gguf_matmul_kernel") as selector:
            output = gguf_quant.fused_mul_mat_gguf(x, qweight, -1)

        selector.assert_not_called()
        self.assertEqual(output.shape, (0, 3))
        self.assertEqual(output.dtype, x.dtype)

    def test_moe_iq_mmq_requires_reuse(self):
        self.assertFalse(_is_iq_moe_mmq_shape(127, 8, 8))
        self.assertFalse(_is_iq_moe_mmq_shape(128, 256, 2))
        self.assertTrue(_is_iq_moe_mmq_shape(128, 256, 4))
        self.assertTrue(_is_iq_moe_mmq_shape(256, 256, 2))
        self.assertFalse(_is_iq_moe_mmq_shape(256, 257, 4))

        max_tokens = (
            gguf_quant.CUDA_MAX_GRID_Y * gguf_quant.IQ_MOE_MMQ_BLOCK_SIZE
            - (256 + 1) * (gguf_quant.IQ_MOE_MMQ_BLOCK_SIZE - 1)
        ) // 4
        self.assertTrue(_is_iq_moe_mmq_shape(max_tokens, 256, 4))
        self.assertFalse(_is_iq_moe_mmq_shape(max_tokens + 1, 256, 4))

    def test_moe_kernel_selection(self):
        with patch.object(gguf_quant, "MMQ_QUANT_TYPES", CUDA_MMQ_TYPES):
            # Existing standard/K kernels retain their 64/65 boundary.
            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.Q5_0, WeightType.Q2_K, 64, 8, 2, 1024, 512
                )
            )
            self.assertTrue(
                _should_use_moe_mmq(
                    WeightType.Q5_0, WeightType.Q2_K, 65, 8, 2, 1024, 512
                )
            )

            # Any I-quant weight selects the conservative I-quant gate.
            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.Q5_0, WeightType.IQ3_S, 127, 8, 8, 1024, 512
                )
            )
            self.assertTrue(
                _should_use_moe_mmq(
                    WeightType.IQ3_S, WeightType.Q5_0, 128, 256, 4, 1024, 512
                )
            )
            self.assertTrue(
                _should_use_moe_mmq(
                    WeightType.Q5_0, WeightType.IQ3_S, 128, 256, 4, 1024, 512
                )
            )
            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.IQ3_S, WeightType.Q5_0, 256, 257, 4, 1024, 512
                )
            )
            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.IQ1_M, WeightType.Q5_0, 256, 8, 8, 1024, 512
                )
            )
            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.Q5_0, WeightType.IQ1_M, 256, 8, 8, 1024, 512
                )
            )

            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.Q8_1, WeightType.IQ3_S, 128, 256, 4, 1024, 512
                )
            )
            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.IQ3_S, WeightType.Q8_1, 128, 256, 4, 1024, 512
                )
            )

            # Both projections must satisfy their type-specific K alignment.
            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.IQ4_NL, WeightType.Q5_0, 128, 256, 4, 96, 512
                )
            )
            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.Q5_0, WeightType.IQ4_NL, 128, 256, 4, 1024, 96
                )
            )
            self.assertTrue(
                _should_use_moe_mmq(
                    WeightType.IQ4_NL,
                    WeightType.IQ4_NL,
                    128,
                    256,
                    4,
                    256,
                    512,
                )
            )
            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.Q2_K, WeightType.IQ3_S, 128, 256, 4, 256, 512
                )
            )
            self.assertFalse(
                _should_use_moe_mmq(
                    WeightType.IQ3_S, WeightType.Q8_0, 128, 256, 4, 256, 96
                )
            )

    def test_fused_moe_uses_high_level_align_wrapper(self):
        num_tokens, hidden_size, num_experts, top_k = 128, 256, 2, 2
        x = torch.ones((num_tokens, hidden_size), dtype=torch.bfloat16)
        w1 = torch.empty((num_experts, 512, 16), dtype=torch.uint8)
        w2 = torch.empty((num_experts, hidden_size, 16), dtype=torch.uint8)
        topk_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32)
        topk_weights = torch.full((num_tokens, top_k), 0.5, dtype=x.dtype)
        sorted_token_ids = torch.arange(num_tokens * top_k, dtype=torch.int32)
        expert_ids = torch.zeros((num_tokens * top_k // 4,), dtype=torch.int32)
        num_tokens_post_padded = torch.tensor([num_tokens * top_k], dtype=torch.int32)
        first_projection = torch.ones((num_tokens * top_k, 512), dtype=x.dtype)
        activated = torch.ones((num_tokens * top_k, 256), dtype=x.dtype)
        second_projection = torch.ones((num_tokens * top_k, hidden_size), dtype=x.dtype)

        def reduce_experts(inp, output):
            output.copy_(inp.sum(dim=1))

        with (
            patch.object(gguf_quant, "MMQ_QUANT_TYPES", CUDA_MMQ_TYPES),
            patch.object(
                gguf_quant, "ggml_moe_get_block_size", return_value=4, create=True
            ),
            patch.object(
                gguf_quant,
                "moe_align_block_size",
                return_value=(
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                ),
                create=True,
            ) as align,
            patch.object(
                gguf_quant,
                "ggml_moe_a8",
                side_effect=(first_projection, second_projection),
                create=True,
            ) as moe_a8,
            patch.object(
                gguf_quant, "silu_and_mul", return_value=activated, create=True
            ),
            patch.object(
                gguf_quant, "moe_sum", side_effect=reduce_experts, create=True
            ),
        ):
            output = gguf_quant.fused_moe_gguf(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                WeightType.IQ3_S,
                WeightType.Q5_0,
                "silu",
            )

        align_args = align.call_args.args
        self.assertIs(align_args[0], topk_ids)
        self.assertEqual(align_args[1:], (4, num_experts))
        self.assertEqual(moe_a8.call_count, 2)
        self.assertEqual(moe_a8.call_args_list[0].args[5], WeightType.IQ3_S)
        self.assertEqual(moe_a8.call_args_list[1].args[5], WeightType.Q5_0)
        torch.testing.assert_close(output, torch.ones_like(output))

    def test_fused_moe_uses_vector_fallback_below_iq_threshold(self):
        num_tokens, hidden_size, num_experts, top_k = 127, 256, 2, 2
        x = torch.ones((num_tokens, hidden_size), dtype=torch.bfloat16)
        w1 = torch.empty((num_experts, 512, 16), dtype=torch.uint8)
        w2 = torch.empty((num_experts, hidden_size, 16), dtype=torch.uint8)
        topk_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32)
        topk_weights = torch.full((num_tokens, top_k), 0.5, dtype=x.dtype)
        first_projection = torch.ones((num_tokens * top_k, 512), dtype=x.dtype)
        activated = torch.ones((num_tokens * top_k, 256), dtype=x.dtype)
        second_projection = torch.ones((num_tokens * top_k, hidden_size), dtype=x.dtype)

        def reduce_experts(inp, output):
            output.copy_(inp.sum(dim=1))

        with (
            patch.object(gguf_quant, "MMQ_QUANT_TYPES", CUDA_MMQ_TYPES),
            patch.object(gguf_quant, "moe_align_block_size", create=True) as align,
            patch.object(gguf_quant, "ggml_moe_a8", create=True) as moe_a8,
            patch.object(
                gguf_quant,
                "ggml_moe_a8_vec",
                side_effect=(first_projection, second_projection),
                create=True,
            ) as moe_a8_vec,
            patch.object(
                gguf_quant, "silu_and_mul", return_value=activated, create=True
            ),
            patch.object(
                gguf_quant, "moe_sum", side_effect=reduce_experts, create=True
            ),
        ):
            output = gguf_quant.fused_moe_gguf(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                WeightType.IQ3_S,
                WeightType.Q5_0,
                "silu",
            )

        align.assert_not_called()
        moe_a8.assert_not_called()
        self.assertEqual(moe_a8_vec.call_count, 2)
        torch.testing.assert_close(output, torch.ones_like(output))


if __name__ == "__main__":
    unittest.main()
