import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.modelopt_fp8_input import (
    ModelOptFp8Input,
    normalize_and_validate_modelopt_fp8_input,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestModelOptFp8InputRejection(CustomTestCase):
    """CPU contract rejection; CUDA execution lives in the qualified SM89 suite."""

    def test_frozen_fields_preserve_tensor_identity(self):
        qx, scale = torch.zeros(1, 16), torch.ones(1)
        value = ModelOptFp8Input(qx, scale, torch.bfloat16)
        self.assertIs(value.qx, qx)
        self.assertIs(value.scale, scale)
        self.assertIsNone(value.row_scales)
        with self.assertRaises(AttributeError):
            value.scale = torch.zeros(1)

    def test_invalid_container_and_tuple(self):
        layer = SimpleNamespace(orig_dtype=torch.bfloat16)
        for value in (None, [], object()):
            with self.subTest(value=type(value)):
                with self.assertRaises(TypeError):
                    normalize_and_validate_modelopt_fp8_input(value, layer)
        for value in ((), (None,), (None, None, None, None)):
            with self.subTest(length=len(value)):
                with self.assertRaises(ValueError):
                    normalize_and_validate_modelopt_fp8_input(value, layer)

    def test_non_fp8_and_cpu_payloads_rejected(self):
        layer = SimpleNamespace(orig_dtype=torch.bfloat16)
        with self.assertRaises(TypeError):
            normalize_and_validate_modelopt_fp8_input((torch.zeros(1, 16), None), layer)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            normalize_and_validate_modelopt_fp8_input(
                (torch.zeros(1, 16, dtype=torch.float8_e4m3fn), None), layer
            )


if __name__ == "__main__":
    unittest.main()
