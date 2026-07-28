"""CPU coverage for biased BF16 FlashInfer TRT-LLM MoE weight preparation."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from sglang.srt.layers.quantization import unquant
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
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
    @staticmethod
    def _make_restore_layer() -> SimpleNamespace:
        return SimpleNamespace(
            num_local_experts=1,
            hidden_size=2880,
            intermediate_size_per_partition=128,
            moe_runner_config=SimpleNamespace(is_gated=True),
        )

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

    def test_preparation_reads_runtime_padded_w13_layout(self):
        """Only logical [up, gate] rows survive a runtime-padded W13 input."""
        hidden_size = 3
        intermediate_size = 5
        w13 = torch.full((2, 256, hidden_size), 9.0, dtype=torch.bfloat16)
        up = torch.full((2, intermediate_size, hidden_size), 2.0, dtype=torch.bfloat16)
        gate = torch.full(
            (2, intermediate_size, hidden_size), 3.0, dtype=torch.bfloat16
        )
        w13[:, :intermediate_size] = up
        w13[:, 128 : 128 + intermediate_size] = gate

        prepared_w13, _, _ = _prepare_flashinfer_trtllm_bf16_weights(
            w13,
            torch.zeros(2, hidden_size, 128, dtype=torch.bfloat16),
            None,
            None,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gemm1_alpha=1.702,
        )

        torch.testing.assert_close(
            prepared_w13[:, :intermediate_size, :hidden_size], up
        )
        torch.testing.assert_close(
            prepared_w13[:, 128 : 128 + intermediate_size, :hidden_size], gate
        )
        self.assertEqual(prepared_w13[:, intermediate_size:128].count_nonzero(), 0)
        self.assertEqual(
            prepared_w13[:, 128 + intermediate_size :, :hidden_size].count_nonzero(),
            0,
        )

    def test_preparation_supports_each_bias_independently(self):
        """W13 and W2 bias folding must not require the other bias tensor."""
        w13 = torch.zeros(2, 10, 3, dtype=torch.bfloat16)
        w2 = torch.zeros(2, 3, 128, dtype=torch.bfloat16)
        w13_bias = torch.arange(20, dtype=torch.float32).view(2, 10)
        w2_bias = torch.arange(6, dtype=torch.float32).view(2, 3)

        w13_only, w2_without_bias, _ = _prepare_flashinfer_trtllm_bf16_weights(
            w13,
            w2,
            w13_bias,
            None,
            hidden_size=3,
            intermediate_size=5,
            gemm1_alpha=1.702,
        )
        torch.testing.assert_close(w13_only[:, :5, 3], w13_bias[:, :5].bfloat16())
        torch.testing.assert_close(w13_only[:, 128:133, 3], w13_bias[:, 5:].bfloat16())
        self.assertEqual(w2_without_bias.count_nonzero(), 0)

        w2_only_w13, w2_only, _ = _prepare_flashinfer_trtllm_bf16_weights(
            w13,
            w2,
            None,
            w2_bias,
            hidden_size=3,
            intermediate_size=5,
            gemm1_alpha=1.702,
        )
        self.assertNotEqual(w2_only_w13[:, 5, 3].count_nonzero(), 0)
        torch.testing.assert_close(w2_only[:, :3, 5], w2_bias.bfloat16())

    def test_preparation_rejects_invalid_bias_shapes_without_broadcasting(self):
        """Per-expert bias tensors must exactly match the prepared weights."""
        w13 = torch.zeros(2, 10, 3, dtype=torch.bfloat16)
        w2 = torch.zeros(2, 3, 128, dtype=torch.bfloat16)

        for bad_w13_bias in (
            torch.zeros(1, 10, dtype=torch.float32),
            torch.zeros(2, 9, dtype=torch.float32),
        ):
            with self.subTest(w13_bias_shape=tuple(bad_w13_bias.shape)):
                with self.assertRaises(ValueError):
                    _prepare_flashinfer_trtllm_bf16_weights(
                        w13,
                        w2,
                        bad_w13_bias,
                        None,
                        hidden_size=3,
                        intermediate_size=5,
                        gemm1_alpha=1.702,
                    )

        for bad_w2_bias in (
            torch.zeros(1, 3, dtype=torch.float32),
            torch.zeros(2, 4, dtype=torch.float32),
        ):
            with self.subTest(w2_bias_shape=tuple(bad_w2_bias.shape)):
                with self.assertRaises(ValueError):
                    _prepare_flashinfer_trtllm_bf16_weights(
                        w13,
                        w2,
                        None,
                        bad_w2_bias,
                        hidden_size=3,
                        intermediate_size=5,
                        gemm1_alpha=1.702,
                    )

    @patch.object(
        unquant,
        "get_moe_runner_backend",
        return_value=unquant.MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
    )
    def test_hot_load_replaces_prepared_w13_with_canonical_shape(self, _):
        """Prepared W13's padded hidden width must be discarded before reload."""
        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        layer = self._make_restore_layer()
        param = Parameter(torch.empty(1, 256, 2944, dtype=torch.bfloat16))

        method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
            layer, param, "model.experts.w13_weight"
        )

        self.assertEqual(tuple(param.shape), (1, 256, 2880))

    @patch.object(
        unquant,
        "get_moe_runner_backend",
        return_value=unquant.MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
    )
    def test_hot_load_replaces_prepared_w2_with_canonical_shape(self, _):
        """Prepared W2's padded output width must be discarded before reload."""
        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        layer = self._make_restore_layer()
        param = Parameter(torch.empty(1, 2944, 128, dtype=torch.bfloat16))

        method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
            layer, param, "model.experts.w2_weight"
        )

        self.assertEqual(tuple(param.shape), (1, 2880, 128))

    @patch.object(
        unquant,
        "get_moe_runner_backend",
        return_value=unquant.MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
    )
    def test_hot_load_rejects_unrecognized_weight_numel(self, _):
        """Hot reload must not reshape tensors outside either known layout."""
        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)

        with self.assertRaisesRegex(RuntimeError, "Cannot restore"):
            method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                self._make_restore_layer(),
                Parameter(torch.empty(1, 17, 17, dtype=torch.bfloat16)),
                "model.experts.w13_weight",
            )

    def test_non_gated_flashinfer_path_skips_bf16_preparation(self):
        """Non-gated FlashInfer MoE keeps its pre-existing block-layout path."""
        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        method.use_flashinfer_trtllm_moe = True
        method.use_deep_gemm = False
        method._cache_permute_indices = {}
        layer = SimpleNamespace(
            moe_runner_config=SimpleNamespace(is_gated=False),
            w13_weight=Parameter(torch.zeros(1, 5, 3, dtype=torch.bfloat16)),
            w2_weight=Parameter(torch.zeros(1, 3, 128, dtype=torch.bfloat16)),
            num_local_experts=1,
        )
        flashinfer = ModuleType("flashinfer")
        flashinfer.__path__ = []
        fused_moe = ModuleType("flashinfer.fused_moe")
        fused_moe.__path__ = []
        core = ModuleType("flashinfer.fused_moe.core")
        calls = []

        def w13_permute(_, weight, *__, **kwargs):
            calls.append("w13_permute")
            self.assertFalse(kwargs["is_gated_act_gemm"])
            return torch.arange(weight.shape[0])

        def w2_permute(_, weight, __):
            calls.append("w2_permute")
            return torch.arange(weight.shape[0])

        def block_layout(weight, _):
            calls.append("block_layout")
            return weight

        core._maybe_get_cached_w3_w1_permute_indices = w13_permute
        core.convert_to_block_layout = block_layout
        core.get_w2_permute_indices_with_cache = w2_permute

        with (
            patch.object(unquant, "_is_cpu", False),
            patch.object(
                unquant,
                "_prepare_flashinfer_trtllm_bf16_weights",
                side_effect=AssertionError("non-gated path must not prepare weights"),
            ),
            patch.dict(
                "sys.modules",
                {
                    "flashinfer": flashinfer,
                    "flashinfer.fused_moe": fused_moe,
                    "flashinfer.fused_moe.core": core,
                },
            ),
        ):
            method.process_weights_after_loading(layer)

        self.assertEqual(
            calls,
            ["w13_permute", "w2_permute", "block_layout", "block_layout"],
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
