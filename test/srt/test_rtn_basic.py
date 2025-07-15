import unittest

import torch

from sglang.srt.layers.quantization.rtn import RTNConfig, rtn_dequantize, rtn_quantize
from sglang.test.test_utils import CustomTestCase


class TestRTNBasic(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_size = 256
        cls.output_size = 512
        cls.group_size = 128
        cls.dtype = torch.bfloat16

    def test_rtn_quantize_dequantize_8bit(self):
        """Test RTN 8-bit quantization and dequantization."""
        tensor = torch.randn(
            self.input_size, self.output_size, dtype=self.dtype, device="cuda"
        )

        qweight, scale = rtn_quantize(tensor, num_bits=8, group_size=self.group_size)

        self.assertEqual(qweight.dtype, torch.uint8)
        self.assertEqual(qweight.shape, tensor.shape)

        num_groups = self.input_size * self.output_size // self.group_size
        self.assertEqual(scale.shape, (self.input_size, num_groups // self.input_size))

        dequant_tensor = rtn_dequantize(qweight, scale)

        self.assertEqual(dequant_tensor.shape, tensor.shape)
        self.assertEqual(dequant_tensor.dtype, scale.dtype)

        error = torch.abs(tensor.float() - dequant_tensor.float()).mean()
        self.assertLess(error, 0.1, "Quantization error too high")

    def test_rtn_quantize_dequantize_4bit(self):
        """Test RTN 4-bit quantization and dequantization."""
        tensor = torch.randn(
            self.input_size, self.output_size, dtype=self.dtype, device="cuda"
        )

        qweight, scale = rtn_quantize(tensor, num_bits=4, group_size=self.group_size)

        self.assertEqual(qweight.dtype, torch.uint8)
        self.assertEqual(qweight.shape, (self.input_size // 2, self.output_size))

        num_groups = self.input_size * self.output_size // self.group_size
        self.assertEqual(scale.shape, (self.input_size, num_groups // self.input_size))

        dequant_tensor = rtn_dequantize(qweight, scale)

        self.assertEqual(dequant_tensor.shape, tensor.shape)
        self.assertEqual(dequant_tensor.dtype, scale.dtype)

        error = torch.abs(tensor.float() - dequant_tensor.float()).mean()
        self.assertLess(error, 0.2, "Quantization error too high")

    def test_rtn_config_validation(self):
        """Test RTN config validation."""
        config_4bit = RTNConfig(weight_bits=4, group_size=128)
        self.assertEqual(config_4bit.weight_bits, 4)
        self.assertEqual(config_4bit.group_size, 128)

        config_8bit = RTNConfig(weight_bits=8, group_size=64)
        self.assertEqual(config_8bit.weight_bits, 8)
        self.assertEqual(config_8bit.group_size, 64)

        with self.assertRaises(ValueError):
            RTNConfig(weight_bits=16, group_size=128)

        with self.assertRaises(ValueError):
            RTNConfig(weight_bits=1, group_size=128)

    def test_rtn_config_methods(self):
        """Test RTN config methods."""
        config = RTNConfig(weight_bits=8, group_size=128)

        self.assertEqual(config.get_name(), "rtn")

        supported_dtypes = config.get_supported_act_dtypes()
        self.assertIn(torch.bfloat16, supported_dtypes)
        self.assertIn(torch.half, supported_dtypes)

        self.assertEqual(config.get_min_capability(), 80)

        config_dict = {"bits": 4, "group_size": 64}
        new_config = RTNConfig.from_config(config_dict)
        self.assertEqual(new_config.weight_bits, 4)
        self.assertEqual(new_config.group_size, 64)

    def test_rtn_quantize_deterministic(self):
        """Test that RTN quantization is deterministic."""
        tensor = torch.randn(
            self.input_size, self.output_size, dtype=self.dtype, device="cuda"
        )

        qweight1, scale1 = rtn_quantize(tensor, num_bits=8, group_size=self.group_size)
        qweight2, scale2 = rtn_quantize(tensor, num_bits=8, group_size=self.group_size)

        torch.testing.assert_close(qweight1, qweight2)
        torch.testing.assert_close(scale1, scale2)


if __name__ == "__main__":
    unittest.main()
