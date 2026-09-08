"""Coverage for FlashInfer TRT-LLM per-channel FP8 MoE integration."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch
from compressed_tensors.quantization import QuantizationStrategy

import sglang.srt.layers.quantization.fp8  # noqa: F401
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as flashinfer_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp8MoeQuantInfo,
    align_fp8_per_channel_moe_weights_for_flashinfer_trtllm,
    fused_experts_none_to_flashinfer_trtllm_fp8,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend, RoutingMethodType
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
    CompressedTensorsW8A8Fp8MoE,
)
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingRunner:
    runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM

    def run(self, dispatch_output, quant_info):
        self.call = (dispatch_output, quant_info)
        return "result"


class TestFlashInferTrtllmFp8PerChannel(CustomTestCase):
    def setUp(self):
        self.moe_flags = get_flags().moe
        self.saved_runner_backend = self.moe_flags.runner_backend
        self.moe_flags.runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM

    def tearDown(self):
        self.moe_flags.runner_backend = self.saved_runner_backend

    def test_aligns_weights_and_scales_with_padding(self):
        from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a

        num_experts, hidden_size, intermediate_size = 2, 64, 33
        padded_intermediate = 48
        w13 = (
            torch.arange(num_experts * 2 * intermediate_size * hidden_size)
            .remainder(31)
            .reshape(num_experts, 2 * intermediate_size, hidden_size)
            .to(torch.float8_e4m3fn)
        )
        w2 = (
            torch.arange(num_experts * hidden_size * intermediate_size)
            .remainder(29)
            .reshape(num_experts, hidden_size, intermediate_size)
            .to(torch.float8_e4m3fn)
        )
        w13_scale = torch.arange(
            1, num_experts * 2 * intermediate_size + 1, dtype=torch.float32
        ).reshape(num_experts, 2 * intermediate_size, 1)
        w2_scale = torch.arange(
            1, num_experts * hidden_size + 1, dtype=torch.float32
        ).reshape(num_experts, hidden_size, 1)
        layer = torch.nn.Module()
        layer.moe_runner_config = MoeRunnerConfig(is_gated=True)
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        expected_w13 = torch.nn.functional.pad(
            w13.reshape(num_experts, 2, intermediate_size, hidden_size).flip(1),
            (0, 0, 0, padded_intermediate - intermediate_size),
        ).reshape(num_experts, 2 * padded_intermediate, hidden_size)
        expected_w13_scale = torch.nn.functional.pad(
            w13_scale.squeeze(-1).reshape(num_experts, 2, intermediate_size).flip(1),
            (0, padded_intermediate - intermediate_size),
            value=1.0,
        ).reshape(num_experts, 2 * padded_intermediate)
        expected_w2 = torch.nn.functional.pad(
            w2, (0, padded_intermediate - intermediate_size)
        )
        expected_w13 = torch.stack(
            [reorder_rows_for_gated_act_gemm(weight) for weight in expected_w13]
        )
        expected_w13_scale = torch.stack(
            [
                reorder_rows_for_gated_act_gemm(scale.unsqueeze(-1)).squeeze(-1)
                for scale in expected_w13_scale
            ]
        )
        expected_w13 = torch.stack(
            [shuffle_matrix_a(weight.view(torch.uint8), 128) for weight in expected_w13]
        ).view(torch.float8_e4m3fn)
        expected_w2 = torch.stack(
            [shuffle_matrix_a(weight.view(torch.uint8), 128) for weight in expected_w2]
        ).view(torch.float8_e4m3fn)
        expected_w13_scale = torch.stack(
            [shuffle_matrix_a(scale.unsqueeze(-1), 128) for scale in expected_w13_scale]
        ).squeeze(-1)
        expected_w2_scale = torch.stack(
            [shuffle_matrix_a(scale, 128) for scale in w2_scale]
        ).squeeze(-1)

        align_fp8_per_channel_moe_weights_for_flashinfer_trtllm(layer)

        self.assertTrue(torch.equal(layer.w13_weight, expected_w13))
        self.assertTrue(torch.equal(layer.w2_weight, expected_w2))
        self.assertTrue(torch.equal(layer.w13_weight_scale, expected_w13_scale))
        self.assertTrue(torch.equal(layer.w2_weight_scale, expected_w2_scale))
        self.assertEqual(layer.intermediate_size_per_partition, padded_intermediate)
        self.assertTrue(torch.equal(layer.output1_scales_scalar, torch.ones(2)))
        self.assertTrue(torch.equal(layer.output1_scales_gate_scalar, torch.ones(2)))
        self.assertTrue(torch.equal(layer.output2_scales_scalar, torch.ones(2)))

    def test_aligns_non_gated_weights_to_128_rows(self):
        from flashinfer import shuffle_matrix_a

        num_experts, hidden_size, intermediate_size = 1, 128, 129
        padded_intermediate = 256
        w13 = (
            torch.arange(num_experts * intermediate_size * hidden_size)
            .remainder(31)
            .reshape(num_experts, intermediate_size, hidden_size)
            .to(torch.float8_e4m3fn)
        )
        w2 = (
            torch.arange(num_experts * hidden_size * intermediate_size)
            .remainder(29)
            .reshape(num_experts, hidden_size, intermediate_size)
            .to(torch.float8_e4m3fn)
        )
        w13_scale = torch.arange(1, intermediate_size + 1, dtype=torch.float32).reshape(
            1, intermediate_size, 1
        )
        w2_scale = torch.arange(1, hidden_size + 1, dtype=torch.float32).reshape(
            1, hidden_size, 1
        )
        layer = torch.nn.Module()
        layer.moe_runner_config = MoeRunnerConfig(activation="relu2", is_gated=False)
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        expected_w13 = torch.nn.functional.pad(
            w13, (0, 0, 0, padded_intermediate - intermediate_size)
        )
        expected_w2 = torch.nn.functional.pad(
            w2, (0, padded_intermediate - intermediate_size)
        )
        expected_scale = torch.nn.functional.pad(
            w13_scale.squeeze(-1),
            (0, padded_intermediate - intermediate_size),
            value=1.0,
        )
        expected_w13 = torch.stack(
            [shuffle_matrix_a(weight.view(torch.uint8), 128) for weight in expected_w13]
        ).view(torch.float8_e4m3fn)
        expected_w2 = torch.stack(
            [shuffle_matrix_a(weight.view(torch.uint8), 128) for weight in expected_w2]
        ).view(torch.float8_e4m3fn)
        expected_scale = torch.stack(
            [shuffle_matrix_a(scale.unsqueeze(-1), 128) for scale in expected_scale]
        ).squeeze(-1)

        align_fp8_per_channel_moe_weights_for_flashinfer_trtllm(layer)

        self.assertTrue(torch.equal(layer.w13_weight, expected_w13))
        self.assertTrue(torch.equal(layer.w2_weight, expected_w2))
        self.assertTrue(torch.equal(layer.w13_weight_scale, expected_scale))
        self.assertEqual(layer.intermediate_size_per_partition, padded_intermediate)

    def test_compressed_tensors_routes_channel_scales_to_flashinfer(self):
        weight_quant = SimpleNamespace(
            strategy=QuantizationStrategy.CHANNEL, dynamic=False
        )
        input_quant = SimpleNamespace(strategy=QuantizationStrategy.TOKEN, dynamic=True)
        method = CompressedTensorsW8A8Fp8MoE(weight_quant, input_quant)
        method.moe_runner_config = MoeRunnerConfig(activation="silu", is_gated=True)
        method.runner = _RecordingRunner()
        layer = SimpleNamespace(
            w13_weight=torch.empty(2, 64, 64, dtype=torch.float8_e4m3fn),
            w2_weight=torch.empty(2, 64, 32, dtype=torch.float8_e4m3fn),
            w13_weight_scale=torch.ones(2, 64),
            w2_weight_scale=torch.ones(2, 64),
            output1_scales_scalar=torch.ones(2),
            output1_scales_gate_scalar=torch.ones(2),
            output2_scales_scalar=torch.ones(2),
            num_experts=8,
            num_local_experts=2,
            moe_ep_rank=1,
            routing_method_type=RoutingMethodType.Default,
        )
        dispatch_output = SimpleNamespace(
            hidden_states=torch.empty(1, 64), topk_output=object()
        )

        result = method.apply_weights(layer, dispatch_output)

        self.assertEqual(result, "result")
        self.assertIs(method.runner.call[0], dispatch_output)
        quant_info = method.runner.call[1]
        self.assertIsInstance(quant_info, FlashInferTrtllmFp8MoeQuantInfo)
        self.assertTrue(quant_info.per_channel_quant)
        self.assertIs(quant_info.w13_per_channel_weight_scale, layer.w13_weight_scale)
        self.assertIs(quant_info.w2_per_channel_weight_scale, layer.w2_weight_scale)
        self.assertEqual(quant_info.local_expert_offset, 2)
        self.assertEqual(quant_info.routing_method_type, int(RoutingMethodType.Default))

    def test_runner_quantizes_per_token_and_calls_per_channel_kernel(self):
        hidden_states = torch.tensor(
            [[1.0, -2.0, 3.0, -4.0], [0.25, 0.5, -0.75, 1.0]],
            dtype=torch.bfloat16,
        )
        router_logits = torch.zeros(2, 2)
        topk_config = TopKConfig(top_k=1, renormalize=False)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=topk_config,
            ),
        )
        quant_info = FlashInferTrtllmFp8MoeQuantInfo(
            w13_weight=torch.zeros(2, 32, 4, dtype=torch.float8_e4m3fn),
            w2_weight=torch.zeros(2, 4, 16, dtype=torch.float8_e4m3fn),
            global_num_experts=2,
            local_expert_offset=0,
            local_num_experts=2,
            intermediate_size=16,
            routing_method_type=int(RoutingMethodType.Renormalize),
            block_quant=False,
            per_channel_quant=True,
            w13_per_channel_weight_scale=torch.ones(2, 32),
            w2_per_channel_weight_scale=torch.ones(2, 4),
            output1_scales_scalar=torch.ones(2),
            output1_scales_gate_scalar=torch.ones(2),
            output2_scales_scalar=torch.ones(2),
        )
        expected = torch.full_like(hidden_states, 7)
        quantized_hidden_states = hidden_states.to(torch.float8_e4m3fn)
        hidden_states_scale = torch.ones(2, 1, dtype=torch.float32)

        with (
            mock.patch.object(flashinfer_runner, "get_tp_group", return_value=None),
            mock.patch.object(
                flashinfer_runner, "is_allocation_symmetric", return_value=False
            ),
            mock.patch.object(
                flashinfer_runner,
                "use_symmetric_memory",
                return_value=nullcontext(),
            ),
            mock.patch.object(
                flashinfer_runner,
                "scaled_fp8_quant",
                return_value=(quantized_hidden_states, hidden_states_scale),
            ) as quantize,
            mock.patch.object(
                flashinfer_runner,
                "trtllm_fp8_per_channel_scale_moe_wrapper",
                return_value=expected,
            ) as kernel,
        ):
            result = fused_experts_none_to_flashinfer_trtllm_fp8(
                dispatch_output,
                quant_info,
                MoeRunnerConfig(activation="silu", is_gated=True),
            )

        self.assertTrue(torch.equal(result.hidden_states, expected))
        quantize.assert_called_once_with(hidden_states, use_per_token_if_dynamic=True)
        kwargs = kernel.call_args.kwargs
        self.assertIs(kwargs["hidden_states"], quantized_hidden_states)
        self.assertIs(kwargs["hidden_states_scale"], hidden_states_scale)
        self.assertIs(kwargs["routing_logits"], router_logits)
        self.assertFalse(kwargs["norm_topk_prob"])
        self.assertIs(
            kwargs["gemm1_per_channel_weight_scale"],
            quant_info.w13_per_channel_weight_scale,
        )
        self.assertEqual(kwargs["top_k"], 1)

    def test_routed_runner_uses_packed_topk_and_local_expert_metadata(self):
        hidden_states = torch.ones(2, 4, dtype=torch.bfloat16)
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(2, 1, dtype=torch.float32),
            topk_ids=torch.tensor([[2], [3]], dtype=torch.int32),
            router_logits=torch.zeros(2, 8),
        )
        dispatch_output = StandardDispatchOutput(hidden_states, None, topk_output)
        quantized_hidden_states = hidden_states.to(torch.float8_e4m3fn)
        hidden_states_scale = torch.ones(2, 1, dtype=torch.float32)
        packed_topk = torch.tensor([[2 << 16], [3 << 16]], dtype=torch.int32)
        expected = torch.full_like(hidden_states, 5)
        quant_info = FlashInferTrtllmFp8MoeQuantInfo(
            w13_weight=torch.zeros(2, 32, 4, dtype=torch.float8_e4m3fn),
            w2_weight=torch.zeros(2, 4, 16, dtype=torch.float8_e4m3fn),
            global_num_experts=8,
            local_expert_offset=2,
            local_num_experts=2,
            intermediate_size=16,
            routing_method_type=int(RoutingMethodType.DeepSeekV3),
            block_quant=False,
            per_channel_quant=True,
            w13_per_channel_weight_scale=torch.ones(2, 32),
            w2_per_channel_weight_scale=torch.ones(2, 4),
            output1_scales_scalar=torch.ones(2),
            output1_scales_gate_scalar=torch.ones(2),
            output2_scales_scalar=torch.ones(2),
            use_routing_scales_on_input=True,
        )

        with (
            mock.patch.object(flashinfer_runner, "get_tp_group", return_value=None),
            mock.patch.object(
                flashinfer_runner, "is_allocation_symmetric", return_value=False
            ),
            mock.patch.object(
                flashinfer_runner,
                "use_symmetric_memory",
                return_value=nullcontext(),
            ),
            mock.patch.object(
                flashinfer_runner,
                "scaled_fp8_quant",
                return_value=(quantized_hidden_states, hidden_states_scale),
            ),
            mock.patch.object(
                flashinfer_runner,
                "_get_packed_topk_ids_for_flashinfer_routed",
                return_value=packed_topk,
            ),
            mock.patch.object(
                flashinfer_runner,
                "trtllm_fp8_per_channel_scale_routed_moe_wrapper",
                return_value=expected,
            ) as kernel,
        ):
            result = fused_experts_none_to_flashinfer_trtllm_fp8(
                dispatch_output,
                quant_info,
                MoeRunnerConfig(activation="silu", is_gated=True, top_k=1),
                use_routed_topk=True,
            )

        self.assertTrue(torch.equal(result.hidden_states, expected))
        kwargs = kernel.call_args.kwargs
        self.assertIs(kwargs["topk_ids"], packed_topk)
        self.assertEqual(kwargs["num_experts"], 8)
        self.assertEqual(kwargs["local_expert_offset"], 2)
        self.assertEqual(kwargs["local_num_experts"], 2)
        self.assertFalse(kwargs["use_routing_scales_on_input"])
        self.assertEqual(kwargs["routing_method_type"], RoutingMethodType.TopK)


if __name__ == "__main__":
    unittest.main()
