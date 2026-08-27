"""Regression tests for Ascend integer online-quantization selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUOnlineW8A8Int8LinearMethod,
    get_npu_online_linear_method,
)
from sglang.srt.hardware_backend.npu.quantization.online_quantization import (
    get_npu_online_integer_quant_spec,
    get_npu_online_moe_integer_quant_spec,
    npu_format_online_moe_scale,
    validate_npu_online_source_dtype,
)
from sglang.srt.layers.linear import LinearBase
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=3, suite="stage-a-unit-test-npu")


class TestOnlineIntegerQuantizationSelection(CustomTestCase):
    """Dense W4A4 caused zero accuracy and must remain on the W8A8 path."""

    @patch(
        "sglang.srt.hardware_backend.npu.quantization.online_quantization.get_server_args"
    )
    def test_w4a4_mode_is_mixed_dense_and_moe(self, get_server_args):
        get_server_args.return_value = SimpleNamespace(
            online_quantization="w4a4_int"
        )

        for projection in (
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
        ):
            method = get_npu_online_linear_method(f"model.layers.0.{projection}")
            self.assertIsInstance(method, NPUOnlineW8A8Int8LinearMethod)

        self.assertIsNone(get_npu_online_linear_method("model.layers.0.mlp.gate"))
        self.assertIsNone(get_npu_online_linear_method("lm_head"))
        self.assertEqual(
            get_npu_online_moe_integer_quant_spec("w13").mode,
            "w4a4_int",
        )
        self.assertEqual(
            get_npu_online_moe_integer_quant_spec("w2").mode,
            "w4a4_int",
        )

    @patch(
        "sglang.srt.hardware_backend.npu.quantization.online_quantization.get_server_args"
    )
    @patch("sglang.srt.layers.linear.current_platform.is_npu")
    def test_linear_selection_uses_live_platform(self, is_npu, get_server_args):
        get_server_args.return_value = SimpleNamespace(
            online_quantization="w8a8_int"
        )
        is_npu.return_value = True
        npu_layer = LinearBase(2, 2, prefix="model.layers.0.qkv_proj")
        self.assertIsInstance(npu_layer.quant_method, NPUOnlineW8A8Int8LinearMethod)

        is_npu.return_value = False
        cpu_layer = LinearBase(2, 2, prefix="model.layers.0.qkv_proj")
        self.assertNotIsInstance(cpu_layer.quant_method, NPUOnlineW8A8Int8LinearMethod)

        quant_config = MagicMock()
        quant_method = object()
        quant_config.get_quant_method.return_value = quant_method
        is_npu.return_value = True
        quantized_layer = LinearBase(2, 2, quant_config=quant_config)
        self.assertIs(quantized_layer.quant_method, quant_method)
        quant_config.get_quant_method.assert_called_once()

    def test_bf16_source_is_supported(self):
        validate_npu_online_source_dtype(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "FP16 or BF16"):
            validate_npu_online_source_dtype(torch.float32)

    def test_w8_moe_scale_matches_output_dtype(self):
        spec = get_npu_online_integer_quant_spec("w8a8_int")
        scale = torch.ones((2, 3, 1), dtype=torch.float32)

        bf16_scale = npu_format_online_moe_scale(
            scale=scale,
            spec=spec,
            weight_prefix="w13",
            output_dtype=torch.bfloat16,
        )
        fp16_scale = npu_format_online_moe_scale(
            scale=scale,
            spec=spec,
            weight_prefix="w13",
            output_dtype=torch.float16,
        )

        self.assertEqual(bf16_scale.dtype, torch.bfloat16)
        self.assertEqual(fp16_scale.dtype, torch.float32)

    def test_w4_moe_scale_preserves_gmm_layouts_and_bits(self):
        spec = get_npu_online_integer_quant_spec("w4a4_int")
        scale = torch.tensor([[[1.5], [2.5], [3.5]], [[4.5], [5.5], [6.5]]])

        w13 = npu_format_online_moe_scale(
            scale=scale,
            spec=spec,
            weight_prefix="w13",
            output_dtype=torch.float16,
        )
        w2 = npu_format_online_moe_scale(
            scale=scale,
            spec=spec,
            weight_prefix="w2",
            output_dtype=torch.float16,
        )
        expected = scale.squeeze(-1).contiguous().view(torch.int32).to(torch.int64)

        self.assertEqual(w13.shape, (2, 3))
        self.assertTrue(torch.equal(w13, expected.squeeze(1)))
        self.assertEqual(w2.shape, (2, 3))
        self.assertTrue(torch.equal(w2, expected))

    def test_w4a4_rejects_dense_qwen_and_accepts_qwen_moe(self):
        def args_for(architecture):
            args = ServerArgs(model_path="dummy", online_quantization="w4a4_int")
            args.model_config = SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture])
            )
            return args

        with self.assertRaisesRegex(ValueError, "dense models.*MoE-only"):
            args_for("Qwen3ForCausalLM")._validate_npu_online_quantization()

        args_for("Qwen3MoeForCausalLM")._validate_npu_online_quantization()


if __name__ == "__main__":
    unittest.main()
