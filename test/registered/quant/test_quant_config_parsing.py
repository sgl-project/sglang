import unittest
from unittest.mock import MagicMock, patch

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.parameter import BlockQuantScaleParameter
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (
    SUPPORTED_STRATEGIES,
    CompressedTensorsW8A16Fp8,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestCompressedTensorsW8A16Fp8(CustomTestCase):
    def test_block_strategy_registers_block_scale(self):
        def noop_loader(*args, **kwargs):
            pass

        block_scheme = CompressedTensorsW8A16Fp8(
            strategy=QuantizationStrategy.BLOCK,
            is_static_input_scheme=False,
            weight_block_size=[128, 128],
        )
        layer = torch.nn.Module()
        block_scheme.create_weights(
            layer=layer,
            input_size=1024,
            output_partition_sizes=[2048],
            input_size_per_partition=1024,
            params_dtype=torch.bfloat16,
            weight_loader=noop_loader,
        )

        self.assertIn(QuantizationStrategy.BLOCK, SUPPORTED_STRATEGIES)
        self.assertEqual(layer.weight_block_size, [128, 128])
        self.assertIsInstance(layer.weight_scale, BlockQuantScaleParameter)
        self.assertEqual(tuple(layer.weight_scale.shape), (16, 8))
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)

        static_block_scheme = CompressedTensorsW8A16Fp8(
            strategy=QuantizationStrategy.BLOCK,
            is_static_input_scheme=True,
            weight_block_size=[128, 128],
        )
        static_layer = torch.nn.Module()
        static_block_scheme.create_weights(
            layer=static_layer,
            input_size=1024,
            output_partition_sizes=[2048],
            input_size_per_partition=1024,
            params_dtype=torch.bfloat16,
            weight_loader=noop_loader,
        )
        with patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w8a16_fp8.prepare_fp8_layer_for_marlin"
        ):
            static_block_scheme.process_weights_after_loading(static_layer)

        self.assertIsInstance(static_layer.input_scale, torch.nn.Parameter)
        self.assertTrue(hasattr(static_layer, "weight_scale_inv"))

        channel_scheme = CompressedTensorsW8A16Fp8(
            strategy=QuantizationStrategy.CHANNEL,
            is_static_input_scheme=False,
        )
        channel_layer = torch.nn.Module()
        channel_scheme.create_weights(
            layer=channel_layer,
            input_size=1024,
            output_partition_sizes=[2048],
            input_size_per_partition=1024,
            params_dtype=torch.bfloat16,
            weight_loader=noop_loader,
        )

        self.assertIsNone(channel_layer.weight_block_size)


class TestQuantLogString(CustomTestCase):
    def test_qwen_fp8_config(self):
        # Example from Qwen/Qwen3-4B-Thinking-2507-FP8
        quant_config = {
            "activation_scheme": "dynamic",
            "modules_to_not_convert": ["lm_head"],
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }

        # Create a raw instance
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=fp8, fmt=e4m3"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test Qwen FP8] Result: {result}")
        self.assertEqual(result, expected)

    def test_llama_gptq_int4_config(self):
        # Example from hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
        quant_config = {"bits": 4, "quant_method": "gptq", "group_size": 128}
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=gptq, bits=4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test Llama GPTQ] Result: {result}")
        self.assertEqual(result, expected)

    def test_awq_config(self):
        quant_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
        }
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=awq, bits=4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test AWQ] Result: {result}")
        self.assertEqual(result, expected)

    def test_modelopt_nvfp4(self):
        quant_config = {"quant_method": "modelopt_fp4", "quant_algo": "NVFP4"}
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=modelopt_fp4, quant_algo=NVFP4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test ModelOpt] Result: {result}")
        self.assertEqual(result, expected)

    def test_no_quant_config(self):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=None)

        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test No Quant] Result: {result}")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
