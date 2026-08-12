import os
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.quantization.weight_only_fp8 import (
    FP8_WEIGHT_DTYPE,
    WeightOnlyFP8Linear,
    dequantize_rowwise_fp8_weight,
)
from sglang.test.test_utils import CustomTestCase


def _make_linear(device: torch.device) -> WeightOnlyFP8Linear:
    torch.manual_seed(0)
    linear = WeightOnlyFP8Linear(64, 32, bias=True, compute_dtype=torch.bfloat16)
    linear.weight.data = (torch.randn(32, 64, device=device) * 0.1).to(FP8_WEIGHT_DTYPE)
    linear.weight_scale.data = torch.rand(32, device=device, dtype=torch.float32) + 0.5
    linear.bias.data = torch.randn(32, device=device, dtype=torch.bfloat16)
    return linear.to(device)


def _reference(linear: WeightOnlyFP8Linear, x: torch.Tensor) -> torch.Tensor:
    dequant = dequantize_rowwise_fp8_weight(
        linear.weight, linear.weight_scale, torch.bfloat16
    )
    return torch.nn.functional.linear(x, dequant, linear.bias)


class TestWeightOnlyFP8DequantCache(CustomTestCase):
    def test_cpu_forward_stays_fp8(self):
        linear = _make_linear(torch.device("cpu"))
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        reference = _reference(linear, x)
        self.assertTrue(torch.equal(reference, linear(x)))
        self.assertEqual(linear.weight.dtype, FP8_WEIGHT_DTYPE)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_first_forward_promotes_bit_identically(self):
        linear = _make_linear(torch.device("cuda"))
        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
        reference = _reference(linear, x)

        out = linear(x)
        self.assertEqual(linear.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(reference, out))
        self.assertTrue(torch.equal(reference, linear(x)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_promotion_dtype_follows_input_when_unset(self):
        linear = _make_linear(torch.device("cuda"))
        linear.compute_dtype = None
        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
        linear(x)
        self.assertEqual(linear.weight.dtype, torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_inference_mode_promotion_supports_torch_compile(self):
        linear = _make_linear(torch.device("cuda"))
        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
        reference = _reference(linear, x)

        with torch.inference_mode():
            eager_out = linear(x)
        self.assertFalse(linear.weight.is_inference())

        compiled = torch.compile(linear, fullgraph=True)
        with torch.inference_mode():
            compiled_out = compiled(x)
        self.assertTrue(torch.equal(reference, eager_out))
        self.assertTrue(torch.equal(reference, compiled_out))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_env_kill_switch(self):
        linear = _make_linear(torch.device("cuda"))
        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
        reference = _reference(linear, x)
        with patch.dict(os.environ, {"SGLANG_DIFFUSION_FP8_WEIGHT_DEQUANT_CACHE": "0"}):
            out = linear(x)
        self.assertEqual(linear.weight.dtype, FP8_WEIGHT_DTYPE)
        self.assertTrue(torch.equal(reference, out))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_low_memory_keeps_fp8(self):
        linear = _make_linear(torch.device("cuda"))
        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
        with patch("torch.cuda.mem_get_info", return_value=(0, 0)):
            linear(x)
        self.assertEqual(linear.weight.dtype, FP8_WEIGHT_DTYPE)


if __name__ == "__main__":
    unittest.main()
