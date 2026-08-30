"""Online MXFP8 MoE loading must never materialize the full BF16 model."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 640


class _RecordingLayer:
    def __init__(self, is_gated: bool = False):
        self.moe_runner_config = MoeRunnerConfig(is_gated=is_gated)
        self.params = {}

    def register_parameter(self, name, param):
        self.params[name] = param
        setattr(self, name, param)

    @staticmethod
    def _map_global_expert_id_to_local_expert_id(expert_id):
        return expert_id if 0 <= expert_id < NUM_EXPERTS else -1


def _backend(*, flashinfer_trtllm: bool):
    backend = MagicMock()
    backend.is_flashinfer_trtllm.return_value = flashinfer_trtllm
    backend.is_flashinfer_trtllm_routed.return_value = False
    backend.is_cutlass.return_value = False
    return backend


def _method(*, flashinfer_trtllm: bool = True):
    from sglang.srt.layers.quantization import fp8 as fp8_quant

    quant_config = SimpleNamespace(
        use_mxfp8=True,
        weight_block_size=[1, 32],
        is_fp4_experts=False,
        dequant_fp4_to_fp8=False,
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
    )
    with patch.object(
        fp8_quant,
        "get_moe_runner_backend",
        return_value=_backend(flashinfer_trtllm=flashinfer_trtllm),
    ):
        return fp8_quant.Fp8MoEMethod(quant_config)


def _create_online_weights(
    *,
    flashinfer_trtllm: bool = True,
    blackwell: bool | None = None,
    is_gated: bool = False,
    weight_loader=None,
):
    from sglang.srt.layers.quantization import fp8 as fp8_quant

    method = _method(flashinfer_trtllm=flashinfer_trtllm)
    layer = _RecordingLayer(is_gated=is_gated)
    original_weight_loader = weight_loader or MagicMock()
    blackwell = flashinfer_trtllm if blackwell is None else blackwell
    with (
        patch.object(fp8_quant, "get_parallel") as parallel,
        patch.object(fp8_quant, "is_blackwell_supported", return_value=blackwell),
        patch.object(fp8_quant, "is_flashinfer_available", return_value=blackwell),
        patch.object(
            fp8_quant,
            "get_moe_runner_backend",
            return_value=_backend(flashinfer_trtllm=flashinfer_trtllm),
        ),
    ):
        parallel.return_value.tp_size = 1
        method.create_weights(
            layer=layer,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            params_dtype=torch.bfloat16,
            weight_loader=original_weight_loader,
        )
    return method, layer, original_weight_loader


class TestMxfp8OnlineMoeLoading(CustomTestCase):
    def test_flashinfer_online_mxfp8_allocates_final_dtypes(self):
        _, layer, _ = _create_online_weights()

        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w2_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w13_weight_scale_inv.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_scale_inv.dtype, torch.uint8)
        self.assertTrue(callable(layer.w13_weight.weight_loader))
        self.assertIs(
            layer.w13_weight.weight_loader,
            layer.w13_weight_scale_inv.weight_loader,
        )

    def test_gated_online_mxfp8_allocates_merged_w13(self):
        _, layer, _ = _create_online_weights(is_gated=True)
        self.assertEqual(
            layer.w13_weight.shape,
            (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        )
        self.assertEqual(
            layer.w13_weight_scale_inv.shape,
            (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32),
        )

    def test_other_backends_use_load_time_quantization(self):
        method, layer, original_weight_loader = _create_online_weights(
            flashinfer_trtllm=False
        )
        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w13_weight_scale_inv.dtype, torch.uint8)
        self.assertTrue(callable(layer.w13_weight_scale_inv.weight_loader))

        loaded_weight = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)
        with patch(
            "sglang.srt.layers.quantization.fp8.mxfp8_group_quantize",
            return_value=(
                loaded_weight.to(torch.float8_e4m3fn),
                torch.zeros(HIDDEN, INTERMEDIATE // 32, dtype=torch.uint8),
            ),
        ) as quantize:
            layer.w2_weight.weight_loader(
                layer.w2_weight,
                loaded_weight,
                weight_name="experts.0.down_proj.weight",
                shard_id="w2",
                expert_id=0,
            )
        quantize.assert_called_once()
        self.assertEqual(original_weight_loader.call_count, 2)

        method._process_mxfp8_moe_weights = MagicMock()
        with patch(
            "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
            return_value=_backend(flashinfer_trtllm=False),
        ):
            method.process_weights_after_loading_block_quant(layer)
        method._process_mxfp8_moe_weights.assert_called_once_with(layer, quantize=False)

    def test_weight_loader_quantizes_and_stores_weight_and_scale(self):
        _, layer, original_weight_loader = _create_online_weights()
        online_loader = layer.w13_weight.weight_loader
        loaded_weight = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)

        def fake_quantize(weight, is_sf_swizzled_layout):
            self.assertEqual(weight.shape, (INTERMEDIATE, HIDDEN))
            self.assertFalse(is_sf_swizzled_layout)
            return (
                weight.to(torch.float8_e4m3fn),
                torch.zeros(INTERMEDIATE * HIDDEN // 32, dtype=torch.uint8),
            )

        with patch(
            "sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_quantize",
            side_effect=fake_quantize,
            create=True,
        ):
            online_loader(
                layer.w13_weight,
                loaded_weight,
                weight_name="experts.0.up_proj.weight",
                shard_id="w3",
                expert_id=0,
            )

        self.assertEqual(original_weight_loader.call_count, 2)
        weight_call, scale_call = original_weight_loader.call_args_list
        self.assertEqual(weight_call.args[1].dtype, torch.float8_e4m3fn)
        self.assertEqual(weight_call.kwargs["shard_id"], "w3")
        self.assertIs(scale_call.args[0], layer.w13_weight_scale_inv)
        self.assertEqual(scale_call.args[1].dtype, torch.uint8)
        self.assertEqual(
            scale_call.kwargs["weight_name"],
            "experts.0.up_proj.weight_scale_inv",
        )

    def test_fused_weight_loader_quantizes_weight_and_scale(self):
        calls = []

        def record_fused(_label, param, loaded_weight, weight_name, shard_id):
            calls.append((param, loaded_weight, weight_name, shard_id))

        _, layer, _ = _create_online_weights(
            weight_loader=partial(record_fused, "fused")
        )
        loaded_weight = torch.randn(
            NUM_EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
        )

        def fake_quantize(weight, is_sf_swizzled_layout):
            self.assertEqual(weight.shape, (NUM_EXPERTS * INTERMEDIATE, HIDDEN))
            self.assertFalse(is_sf_swizzled_layout)
            return (
                weight.to(torch.float8_e4m3fn),
                torch.zeros(
                    NUM_EXPERTS * INTERMEDIATE * HIDDEN // 32,
                    dtype=torch.uint8,
                ),
            )

        with patch(
            "sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_quantize",
            side_effect=fake_quantize,
            create=True,
        ):
            layer.w13_weight.weight_loader(
                layer.w13_weight,
                loaded_weight,
                weight_name="experts.gate_up_proj.weight",
                shard_id="w13",
            )

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[1][0], layer.w13_weight_scale_inv)
        self.assertEqual(calls[1][2], "experts.gate_up_proj.weight_scale_inv")
        self.assertEqual(calls[1][1].shape, (NUM_EXPERTS, INTERMEDIATE, HIDDEN // 32))

    def test_fused_weight_loader_supports_unset_shard_id(self):
        calls = []

        def record_fused(_label, param, loaded_weight, weight_name, shard_id):
            calls.append((param, loaded_weight, weight_name, shard_id))

        _, layer, _ = _create_online_weights(
            flashinfer_trtllm=False,
            weight_loader=partial(record_fused, "fused"),
        )

        def fake_quantize(weight):
            return (
                weight.to(torch.float8_e4m3fn),
                torch.zeros(weight.shape[0], weight.shape[1] // 32, dtype=torch.uint8),
            )

        with patch(
            "sglang.srt.layers.quantization.fp8.mxfp8_group_quantize",
            side_effect=fake_quantize,
        ):
            for param, loaded_weight, weight_name in (
                (
                    layer.w13_weight,
                    torch.randn(
                        NUM_EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
                    ),
                    "experts.gate_up_proj.weight",
                ),
                (
                    layer.w2_weight,
                    torch.randn(
                        NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16
                    ),
                    "experts.down_proj.weight",
                ),
            ):
                param.weight_loader(
                    param,
                    loaded_weight,
                    weight_name=weight_name,
                    shard_id=None,
                )

        self.assertEqual(len(calls), 4)
        self.assertIs(calls[1][0], layer.w13_weight_scale_inv)
        self.assertIs(calls[3][0], layer.w2_weight_scale_inv)
        self.assertTrue(all(call[3] is None for call in calls))

    def test_nonlocal_expert_is_skipped_before_quantization(self):
        _, layer, original_weight_loader = _create_online_weights()
        online_loader = layer.w2_weight.weight_loader
        with patch(
            "sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_quantize",
            create=True,
        ) as quantize:
            online_loader(
                layer.w2_weight,
                torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16),
                weight_name="experts.9.down_proj.weight",
                shard_id="w2",
                expert_id=9,
            )
        quantize.assert_not_called()
        original_weight_loader.assert_not_called()

    def test_flashinfer_backend_falls_back_to_triton_off_blackwell(self):
        from sglang.srt.layers.quantization import fp8 as fp8_quant

        _, layer, _ = _create_online_weights(
            flashinfer_trtllm=True,
            blackwell=False,
        )
        loaded_weight = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
        quantized = loaded_weight.to(torch.float8_e4m3fn)
        scale = torch.zeros(INTERMEDIATE, HIDDEN // 32, dtype=torch.uint8)
        with patch.object(
            fp8_quant,
            "mxfp8_group_quantize",
            return_value=(quantized, scale),
        ) as triton_quantize:
            layer.w13_weight.weight_loader(
                layer.w13_weight,
                loaded_weight,
                weight_name="experts.0.up_proj.weight",
                shard_id="w3",
                expert_id=0,
            )
        triton_quantize.assert_called_once()

    def test_post_load_only_finalizes_layout(self):
        method = _method()
        method._process_mxfp8_moe_weights = MagicMock()
        layer = MagicMock()
        method.process_weights_after_loading_block_quant(layer)
        method._process_mxfp8_moe_weights.assert_called_once_with(layer, quantize=False)

    def test_serialized_mxfp8_keeps_existing_loader_and_finalization(self):
        from sglang.srt.layers.quantization import fp8 as fp8_quant

        method = _method()
        method.quant_config.is_checkpoint_fp8_serialized = True
        layer = _RecordingLayer()
        original_weight_loader = MagicMock()
        with (
            patch.object(fp8_quant, "get_parallel") as parallel,
            patch.object(
                fp8_quant,
                "get_moe_runner_backend",
                return_value=_backend(flashinfer_trtllm=True),
            ),
        ):
            parallel.return_value.tp_size = 1
            method.create_weights(
                layer=layer,
                num_experts=NUM_EXPERTS,
                hidden_size=HIDDEN,
                intermediate_size_per_partition=INTERMEDIATE,
                params_dtype=torch.bfloat16,
                weight_loader=original_weight_loader,
            )

        self.assertIs(layer.w13_weight.weight_loader, original_weight_loader)
        self.assertIs(layer.w13_weight_scale_inv.weight_loader, original_weight_loader)
        method._process_mxfp8_moe_weights = MagicMock()
        method.process_weights_after_loading_block_quant(layer)
        method._process_mxfp8_moe_weights.assert_called_once_with(layer, quantize=False)


if __name__ == "__main__":
    unittest.main()
