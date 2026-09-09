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
    fused_experts_none_to_flashinfer_trtllm_fp8,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
)
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    RoutingMethodType,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
    CompressedTensorsW8A8Fp8MoE,
)
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
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

    def test_real_loader_preserves_canonical_scales_across_reload(self):
        from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        num_experts, hidden_size, checkpoint_intermediate = 2, 128, 33
        weight_quant = SimpleNamespace(
            strategy=QuantizationStrategy.CHANNEL, dynamic=False
        )
        input_quant = SimpleNamespace(strategy=QuantizationStrategy.TOKEN, dynamic=True)
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
                a2a_backend=MoeA2ABackend.NONE,
            ),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
        ):
            method = CompressedTensorsW8A8Fp8MoE(weight_quant, input_quant)
            layer = FusedMoE(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=checkpoint_intermediate,
                layer_id=0,
                top_k=1,
                params_dtype=torch.bfloat16,
                quant_method=method,
                routing_method_type=RoutingMethodType.Renormalize,
                gate_up_interleaved=False,
            )

        # FusedMoE owns TRT-LLM padding. The post-load conversion must consume
        # this already-aligned representation rather than introducing another
        # padding policy.
        self.assertEqual(layer.intermediate_size_per_partition, 128)
        canonical_scale_params = (
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )

        def load_and_process(generation: int):
            for expert_id in range(num_experts):
                offset = generation * 17 + expert_id * 5
                shard_values = {
                    "w1": (
                        torch.arange(checkpoint_intermediate * hidden_size)
                        .add(offset)
                        .remainder(31)
                        .reshape(checkpoint_intermediate, hidden_size)
                        .to(torch.float8_e4m3fn)
                    ),
                    "w3": (
                        torch.arange(checkpoint_intermediate * hidden_size)
                        .add(offset + 3)
                        .remainder(29)
                        .reshape(checkpoint_intermediate, hidden_size)
                        .to(torch.float8_e4m3fn)
                    ),
                    "w2": (
                        torch.arange(hidden_size * checkpoint_intermediate)
                        .add(offset + 7)
                        .remainder(23)
                        .reshape(hidden_size, checkpoint_intermediate)
                        .to(torch.float8_e4m3fn)
                    ),
                }
                scale_values = {
                    "w1": torch.arange(
                        1 + offset,
                        1 + offset + checkpoint_intermediate,
                        dtype=torch.float32,
                    ).unsqueeze(-1),
                    "w3": torch.arange(
                        101 + offset,
                        101 + offset + checkpoint_intermediate,
                        dtype=torch.float32,
                    ).unsqueeze(-1),
                    "w2": torch.arange(
                        201 + offset,
                        201 + offset + hidden_size,
                        dtype=torch.float32,
                    ).unsqueeze(-1),
                }
                for shard_id in ("w1", "w3", "w2"):
                    prefix = "w13" if shard_id in ("w1", "w3") else "w2"
                    for suffix, loaded in (
                        ("weight", shard_values[shard_id]),
                        ("weight_scale", scale_values[shard_id]),
                    ):
                        name = f"{prefix}_{suffix}"
                        layer.weight_loader(
                            getattr(layer, name),
                            loaded,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )

            canonical_w13 = layer.w13_weight.detach().clone()
            canonical_w2 = layer.w2_weight.detach().clone()
            canonical_w13_scale = layer.w13_weight_scale.detach().clone()
            canonical_w2_scale = layer.w2_weight_scale.detach().clone()
            intermediate_size = canonical_w2.shape[2]

            expected_w13 = (
                canonical_w13.reshape(num_experts, 2, intermediate_size, hidden_size)
                .flip(1)
                .reshape(num_experts, 2 * intermediate_size, hidden_size)
            )
            expected_w13_scale = (
                canonical_w13_scale.squeeze(-1)
                .reshape(num_experts, 2, intermediate_size)
                .flip(1)
                .reshape(num_experts, 2 * intermediate_size)
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
                [
                    shuffle_matrix_a(weight.view(torch.uint8), 128)
                    for weight in expected_w13
                ]
            ).view(torch.float8_e4m3fn)
            expected_w2 = torch.stack(
                [
                    shuffle_matrix_a(weight.view(torch.uint8), 128)
                    for weight in canonical_w2
                ]
            ).view(torch.float8_e4m3fn)
            expected_w13_scale = torch.stack(
                [
                    shuffle_matrix_a(scale.unsqueeze(-1), 128)
                    for scale in expected_w13_scale
                ]
            ).squeeze(-1)
            expected_w2_scale = torch.stack(
                [shuffle_matrix_a(scale, 128) for scale in canonical_w2_scale]
            ).squeeze(-1)

            method.process_weights_after_loading(layer)

            self.assertIs(layer.w13_weight_scale, canonical_scale_params[0])
            self.assertIs(layer.w2_weight_scale, canonical_scale_params[1])
            self.assertTrue(torch.equal(layer.w13_weight_scale, canonical_w13_scale))
            self.assertTrue(torch.equal(layer.w2_weight_scale, canonical_w2_scale))
            self.assertTrue(torch.equal(layer.w13_weight, expected_w13))
            self.assertTrue(torch.equal(layer.w2_weight, expected_w2))
            self.assertTrue(
                torch.equal(layer.w13_per_channel_weight_scale, expected_w13_scale)
            )
            self.assertTrue(
                torch.equal(layer.w2_per_channel_weight_scale, expected_w2_scale)
            )

        load_and_process(generation=0)
        derived_scale_params = (
            layer.w13_per_channel_weight_scale,
            layer.w2_per_channel_weight_scale,
        )
        load_and_process(generation=1)
        self.assertIs(layer.w13_per_channel_weight_scale, derived_scale_params[0])
        self.assertIs(layer.w2_per_channel_weight_scale, derived_scale_params[1])

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
            w13_per_channel_weight_scale=torch.ones(2, 64),
            w2_per_channel_weight_scale=torch.ones(2, 64),
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
        self.assertIs(
            quant_info.w13_per_channel_weight_scale,
            layer.w13_per_channel_weight_scale,
        )
        self.assertIs(
            quant_info.w2_per_channel_weight_scale,
            layer.w2_per_channel_weight_scale,
        )
        self.assertEqual(quant_info.local_expert_offset, 2)
        self.assertEqual(quant_info.routing_method_type, int(RoutingMethodType.Default))

    def test_routed_compressed_tensors_rejects_non_channel_fp8(self):
        input_quant = SimpleNamespace(
            strategy=QuantizationStrategy.TENSOR, dynamic=False
        )
        with get_flags().moe.override(
            runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
        ):
            with self.assertRaisesRegex(ValueError, "only with per-channel weights"):
                CompressedTensorsW8A8Fp8MoE(
                    SimpleNamespace(
                        strategy=QuantizationStrategy.TENSOR, dynamic=False
                    ),
                    input_quant,
                )

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
                "_get_routing_for_flashinfer_routed",
                return_value=(topk_output.topk_ids, topk_output.topk_weights),
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
        self.assertIs(kwargs["topk_ids"], topk_output.topk_ids)
        self.assertIs(kwargs["topk_weights"], topk_output.topk_weights)
        self.assertEqual(kwargs["num_experts"], 8)
        self.assertEqual(kwargs["local_expert_offset"], 2)
        self.assertEqual(kwargs["local_num_experts"], 2)
        self.assertFalse(kwargs["use_routing_scales_on_input"])
        self.assertEqual(kwargs["routing_method_type"], RoutingMethodType.TopK)


if __name__ == "__main__":
    unittest.main()
