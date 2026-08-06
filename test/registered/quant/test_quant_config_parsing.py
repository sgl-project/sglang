import ast
import re
import unittest
from unittest.mock import MagicMock, patch

import sglang.srt.layers.quantization as quantization
import sglang.srt.utils as srt_utils
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.quantization import (
    CPU_QUANTIZATION_METHODS,
    QUANTIZATION_METHODS,
    get_quantization_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


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


class TestCpuUnsupportedQuantizationMessage(CustomTestCase):
    """The CPU rejection message must advertise only CPU-supported methods."""

    def test_cpu_error_lists_only_cpu_methods(self):
        # A method that exists in the registry but is not CPU-supported. This
        # request must be rejected on CPU, and the rejection message must not
        # advertise methods that CPU cannot serve.
        method = "awq_marlin"
        self.assertIn(method, QUANTIZATION_METHODS)
        self.assertNotIn(method, CPU_QUANTIZATION_METHODS)

        # Enter the real CPU validation branch by forcing only the two
        # platform-detection predicates; get_quantization_config itself runs
        # unmodified.
        with (
            patch.object(srt_utils, "is_cpu", return_value=True),
            patch.object(quantization, "cpu_has_amx_support", return_value=True),
        ):
            with self.assertRaises(ValueError) as ctx:
                get_quantization_config(method)

        message = str(ctx.exception)
        match = re.search(r"Available methods on CPU:\s*(\[.*\])", message)
        self.assertIsNotNone(
            match, f"CPU error message not in the expected form: {message!r}"
        )
        advertised = set(ast.literal_eval(match.group(1)))

        # The advertised set must be exactly the CPU-supported methods, and in
        # particular must not include the method that was just rejected.
        self.assertEqual(advertised, set(CPU_QUANTIZATION_METHODS))
        self.assertNotIn(method, advertised)


if __name__ == "__main__":
    unittest.main()
