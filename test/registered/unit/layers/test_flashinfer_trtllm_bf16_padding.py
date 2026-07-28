"""CPU coverage for biased BF16 FlashInfer TRT-LLM MoE weight preparation."""

import unittest
from contextlib import nullcontext
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
from sglang.srt.lora.trtllm_lora_temp import lora_layer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# The runner imports quantization registry state, so initialize unquant first.
# isort: off
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import BypassedTopKOutput, PackedTopKOutput, TopKConfig

# isort: on

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

    def test_bf16_runner_pads_inputs_and_restores_logical_width(self):
        """Both BF16 kernels receive padded inputs and return logical outputs."""
        tokens, hidden_size, kernel_hidden_size = 3, 2880, 2944
        input_pad_value = 1.0
        alpha = torch.tensor([1.702])
        beta = torch.tensor([1.0])
        clamp_limit = torch.tensor([7.0])
        quant_info = flashinfer_trtllm.FlashInferTrtllmBf16MoeQuantInfo(
            gemm1_weights=torch.empty(1),
            gemm2_weights=torch.empty(1),
            global_num_experts=1,
            local_expert_offset=0,
            kernel_hidden_size=kernel_hidden_size,
            input_pad_value=input_pad_value,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=clamp_limit,
        )
        runner_config = MoeRunnerConfig(
            activation="silu",
            is_gated=True,
            num_fused_shared_experts=0,
            num_local_experts=1,
            intermediate_size_per_partition=128,
            top_k=1,
        )
        hidden_states = torch.randn(tokens, hidden_size, dtype=torch.bfloat16)
        kernel_result = torch.randn(kernel_hidden_size, tokens).transpose(0, 1)

        flashinfer = ModuleType("flashinfer")
        flashinfer.__path__ = []
        fused_moe = ModuleType("flashinfer.fused_moe")
        fused_moe.__path__ = []
        core = ModuleType("flashinfer.fused_moe.core")
        core.ActivationType = SimpleNamespace(
            Swiglu=SimpleNamespace(value=1),
            Geglu=SimpleNamespace(value=2),
        )

        for use_routed_topk in (False, True):
            with self.subTest(use_routed_topk=use_routed_topk):
                kernel_calls = []

                def mock_kernel(**kwargs):
                    kernel_calls.append(kwargs)
                    return kernel_result

                if use_routed_topk:
                    fused_moe.trtllm_bf16_routed_moe = mock_kernel
                    topk_output = PackedTopKOutput(
                        packed_topk_ids=torch.zeros(tokens, 1, dtype=torch.int32),
                        router_logits=torch.empty(tokens, 1),
                    )
                else:
                    fused_moe.trtllm_bf16_moe = mock_kernel
                    topk_output = BypassedTopKOutput(
                        hidden_states=hidden_states,
                        router_logits=torch.empty(tokens, 1),
                        topk_config=TopKConfig(top_k=1),
                    )

                dispatch_output = StandardDispatchOutput(
                    hidden_states=hidden_states,
                    hidden_states_scale=None,
                    topk_output=topk_output,
                )
                with (
                    patch.dict(
                        "sys.modules",
                        {
                            "flashinfer": flashinfer,
                            "flashinfer.fused_moe": fused_moe,
                            "flashinfer.fused_moe.core": core,
                        },
                    ),
                    patch.object(flashinfer_trtllm, "get_tp_group", return_value=None),
                    patch.object(
                        flashinfer_trtllm,
                        "use_symmetric_memory",
                        return_value=nullcontext(),
                    ),
                    patch.object(
                        flashinfer_trtllm, "is_allocation_symmetric", return_value=False
                    ),
                ):
                    output = (
                        flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_bf16(
                            dispatch_output,
                            quant_info,
                            runner_config,
                            use_routed_topk=use_routed_topk,
                        )
                    )

                self.assertEqual(len(kernel_calls), 1)
                kernel_input = kernel_calls[0]["hidden_states"]
                self.assertEqual(
                    tuple(kernel_input.shape), (tokens, kernel_hidden_size)
                )
                torch.testing.assert_close(kernel_input[:, :hidden_size], hidden_states)
                torch.testing.assert_close(
                    kernel_input[:, hidden_size:],
                    torch.full(
                        (tokens, kernel_hidden_size - hidden_size),
                        input_pad_value,
                        dtype=hidden_states.dtype,
                    ),
                )
                self.assertIs(kernel_calls[0]["gemm1_alpha"], alpha)
                self.assertIs(kernel_calls[0]["gemm1_beta"], beta)
                self.assertIs(kernel_calls[0]["gemm1_clamp_limit"], clamp_limit)
                self.assertEqual(
                    tuple(output.hidden_states.shape), (tokens, hidden_size)
                )
                self.assertTrue(output.hidden_states.is_contiguous())
                torch.testing.assert_close(
                    output.hidden_states, kernel_result[:, :hidden_size]
                )

    def test_forward_cuda_builds_bf16_padding_payload_for_gated_and_non_gated(self):
        """Gated preparation values flow through, while non-gated uses safe defaults."""

        class FakeBackend:
            def is_triton_kernels(self):
                return False

            def is_deep_gemm(self):
                return False

        class CaptureRunner:
            runner_backend = FakeBackend()

            def run(self, dispatch_output, quant_info):
                self.quant_info = quant_info
                return dispatch_output

        dispatch_output = StandardDispatchOutput(
            hidden_states=torch.empty(2, 2880, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_output=SimpleNamespace(),
        )
        alpha = torch.tensor([1.702])
        beta = torch.tensor([1.0])
        clamp_limit = torch.tensor([7.0])

        for is_gated in (True, False):
            with self.subTest(is_gated=is_gated):
                method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
                method.use_flashinfer_cutlass = False
                method.use_flashinfer_trtllm_moe = True
                method.runner = CaptureRunner()
                if is_gated:
                    method._flashinfer_kernel_hidden_size = 2944
                    method._flashinfer_input_pad_value = 1.0
                    method._flashinfer_gemm1_alpha = alpha
                    method._flashinfer_gemm1_beta = beta
                    method._flashinfer_gemm1_clamp_limit = clamp_limit
                layer = SimpleNamespace(
                    w13_weight=Parameter(torch.empty(1)),
                    w2_weight=Parameter(torch.empty(1)),
                    num_experts=4,
                    moe_ep_rank=1,
                    num_local_experts=2,
                    hidden_size=2880,
                    moe_runner_config=SimpleNamespace(is_gated=is_gated),
                )

                method.forward_cuda(layer, dispatch_output)
                quant_info = method.runner.quant_info

                self.assertEqual(
                    quant_info.kernel_hidden_size, 2944 if is_gated else 2880
                )
                self.assertEqual(quant_info.input_pad_value, 1.0 if is_gated else 0.0)
                self.assertIs(quant_info.gemm1_alpha, alpha if is_gated else None)
                self.assertIs(quant_info.gemm1_beta, beta if is_gated else None)
                self.assertIs(
                    quant_info.gemm1_clamp_limit, clamp_limit if is_gated else None
                )

    def test_bf16_lora_payload_uses_prepared_values_or_non_gated_fallback(self):
        """BF16 LoRA construction must satisfy the padded runner payload contract."""
        alpha = torch.tensor([1.702])
        beta = torch.tensor([1.0])
        clamp_limit = torch.tensor([7.0])

        for is_gated in (True, False):
            with self.subTest(is_gated=is_gated):
                quant_method = SimpleNamespace(quant_config=None, block_quant=False)
                if is_gated:
                    quant_method._flashinfer_kernel_hidden_size = 2944
                    quant_method._flashinfer_input_pad_value = 1.0
                    quant_method._flashinfer_gemm1_alpha = alpha
                    quant_method._flashinfer_gemm1_beta = beta
                    quant_method._flashinfer_gemm1_clamp_limit = clamp_limit
                base_layer = SimpleNamespace(
                    quant_method=quant_method,
                    w13_weight=Parameter(torch.empty(1, 2, 2, 2)),
                    w2_weight=Parameter(torch.empty(1, 2, 2, 2)),
                    num_experts=4,
                    moe_ep_rank=1,
                    num_local_experts=2,
                    hidden_size=2880,
                    moe_runner_config=SimpleNamespace(is_gated=is_gated),
                )
                layer = SimpleNamespace()

                with patch.object(lora_layer, "_warm_sgl_trtllm_moe_module"):
                    lora_layer.init_experimental_sgl_trtllm_lora(layer, base_layer)

                quant_info = layer._quant_info
                self.assertEqual(
                    quant_info.kernel_hidden_size, 2944 if is_gated else 2880
                )
                self.assertEqual(quant_info.input_pad_value, 1.0 if is_gated else 0.0)
                self.assertIs(quant_info.gemm1_alpha, alpha if is_gated else None)
                self.assertIs(quant_info.gemm1_beta, beta if is_gated else None)
                self.assertIs(
                    quant_info.gemm1_clamp_limit, clamp_limit if is_gated else None
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
