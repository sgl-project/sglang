"""Regression tests for Ascend integer online-quantization selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUOnlineW8A8Int8LinearMethod,
    get_npu_online_linear_method,
)
from sglang.srt.hardware_backend.npu.quantization.online_quantization import (
    get_npu_online_moe_integer_quant_spec,
    validate_npu_online_source_dtype,
)
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
            online_quantization="w4a4_int4"
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
            "w4a4_int4",
        )
        self.assertEqual(
            get_npu_online_moe_integer_quant_spec("w2").mode,
            "w8a8_int8",
        )

    def test_bf16_source_is_supported(self):
        validate_npu_online_source_dtype(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "FP16 or BF16"):
            validate_npu_online_source_dtype(torch.float32)


if __name__ == "__main__":
    unittest.main()
