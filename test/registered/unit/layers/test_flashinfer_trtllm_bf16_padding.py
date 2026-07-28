"""CPU coverage for biased BF16 FlashInfer TRT-LLM MoE weight preparation."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import (
    _prepare_flashinfer_trtllm_bf16_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def gpt_oss_activation(
    gate_up: torch.Tensor, alpha: float, limit: float
) -> torch.Tensor:
    """GPT-OSS SwiGLU for the FlashInfer [up, gate] projection order."""
    up, gate = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1.0)


class TestFlashInferTrtllmBf16Padding(CustomTestCase):
    def test_biased_bf16_weights_match_bias_free_padded_moe(self):
        """Bias folding must preserve each expert's GPT-OSS MoE output."""
        torch.manual_seed(0)
        num_experts = 2
        hidden_size = 3
        intermediate_size = 5
        kernel_intermediate_size = 128
        alpha = 1.702
        limit = 7.0

        x = torch.randn(4, hidden_size, dtype=torch.bfloat16)
        w13 = torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16
        )
        w2 = torch.zeros(
            num_experts,
            hidden_size,
            kernel_intermediate_size,
            dtype=torch.bfloat16,
        )
        w2[..., :intermediate_size] = torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16
        )
        w13_bias = torch.randn(num_experts, 2 * intermediate_size, dtype=torch.float32)
        w2_bias = torch.randn(num_experts, hidden_size, dtype=torch.float32)

        prepared_w13, prepared_w2, kernel_hidden_size = (
            _prepare_flashinfer_trtllm_bf16_weights(
                w13,
                w2,
                w13_bias,
                w2_bias,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                gemm1_alpha=alpha,
            )
        )

        self.assertEqual(kernel_hidden_size, 128)
        self.assertEqual(prepared_w13.shape, (num_experts, 256, 128))
        self.assertEqual(prepared_w2.shape, (num_experts, 128, 128))

        x_padded = F.pad(x, (0, kernel_hidden_size - hidden_size), value=1.0)
        for expert in range(num_experts):
            expected = F.linear(
                gpt_oss_activation(
                    F.linear(x.float(), w13[expert].float(), w13_bias[expert]),
                    alpha,
                    limit,
                ),
                w2[expert, :, :intermediate_size].float(),
                w2_bias[expert],
            )
            actual = F.linear(
                gpt_oss_activation(
                    F.linear(x_padded, prepared_w13[expert]), alpha, limit
                ),
                prepared_w2[expert],
            )[:, :hidden_size]

            torch.testing.assert_close(actual.float(), expected, atol=0.03, rtol=0.03)

    def test_preparation_preserves_up_then_gate_projection_order(self):
        """FlashInfer weights must retain FusedMoE's [up, gate] row order."""
        hidden_size = 3
        intermediate_size = 5
        w13 = (
            torch.arange(2 * 2 * intermediate_size * hidden_size, dtype=torch.float32)
            .reshape(2, 2 * intermediate_size, hidden_size)
            .to(torch.bfloat16)
        )
        w2 = torch.zeros(2, hidden_size, 128, dtype=torch.bfloat16)

        prepared_w13, _, _ = _prepare_flashinfer_trtllm_bf16_weights(
            w13,
            w2,
            None,
            None,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gemm1_alpha=1.702,
        )

        torch.testing.assert_close(
            prepared_w13[:, :intermediate_size, :hidden_size],
            w13[:, :intermediate_size],
        )
        torch.testing.assert_close(
            prepared_w13[:, 128 : 128 + intermediate_size, :hidden_size],
            w13[:, intermediate_size:],
        )

    def test_bias_folding_requires_spare_padded_channels(self):
        """Bias folding must reject kernel shapes with no synthetic channels."""
        with self.assertRaises(ValueError):
            _prepare_flashinfer_trtllm_bf16_weights(
                torch.zeros(2, 10, 128, dtype=torch.bfloat16),
                torch.zeros(2, 128, 128, dtype=torch.bfloat16),
                torch.zeros(2, 10, dtype=torch.float32),
                None,
                hidden_size=128,
                intermediate_size=5,
                gemm1_alpha=1.702,
            )

        with self.assertRaises(ValueError):
            _prepare_flashinfer_trtllm_bf16_weights(
                torch.zeros(2, 10, 128, dtype=torch.bfloat16),
                torch.zeros(2, 128, 128, dtype=torch.bfloat16),
                None,
                torch.zeros(2, 128, dtype=torch.float32),
                hidden_size=128,
                intermediate_size=5,
                gemm1_alpha=1.702,
            )

        with self.assertRaises(ValueError):
            _prepare_flashinfer_trtllm_bf16_weights(
                torch.zeros(2, 256, 3, dtype=torch.bfloat16),
                torch.zeros(2, 3, 128, dtype=torch.bfloat16),
                None,
                torch.zeros(2, 3, dtype=torch.float32),
                hidden_size=3,
                intermediate_size=128,
                gemm1_alpha=1.702,
            )


if __name__ == "__main__":
    unittest.main()
