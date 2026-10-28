import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.gptq.gptq import CPUGPTQConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")


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


class TestCPUGPTQDynamicConfig(CustomTestCase):
    def test_excluded_tp3_attention_projection(self):
        """Excluded attention projections must not require INT4 block alignment."""
        config = CPUGPTQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "desc_act": False,
                "sym": True,
                "dynamic": {"-:.*attn.*": {}},
            }
        )
        self.assertTrue(config.is_sym)
        layer = LinearBase(
            input_size=2048,
            output_size=72,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix="model.layers.0.linear_attn.in_proj_ba",
        )
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
        layer.quant_method.create_weights(
            layer=layer,
            input_size_per_partition=2048,
            output_partition_sizes=[12, 12],
            input_size=2048,
            output_size=72,
            params_dtype=torch.bfloat16,
        )
        self.assertEqual(layer.weight.shape, (24, 2048))
        self.assertNotIn("qweight", layer._parameters)


if __name__ == "__main__":
    unittest.main()
