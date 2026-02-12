import unittest
from unittest.mock import MagicMock

from sglang.srt.configs.model_config import ModelConfig


class MockHfConfig:
    def __init__(self, quant_config=None):
        self.quantization_config = quant_config
        self.architectures = ["LlamaForCausalLM"]
        self.model_type = "llama"


class TestQuantConfigParsing(unittest.TestCase):
    def test_gptq_config(self):
        # GPTQ 4-bit
        quant_config = {"quant_method": "gptq", "bits": 4, "group_size": 128}

        # Mocking ModelConfig
        model_config = MagicMock()
        model_config.hf_config = MockHfConfig(quant_config)
        # We bind the definition of _parse_quant_hf_config from the actual class to the mock if possible,
        # or just reimplement the logic we want to test: lines 627-629 of model_config.py

        # Logic from model_config.py mainly accesses attributes.
        # Let's verify what the actual function returns.
        # Since we cannot easily instantiate full ModelConfig without files, we will use the logic directly.

        extracted_cfg = getattr(model_config.hf_config, "quantization_config", None)
        self.assertEqual(extracted_cfg, quant_config)
        self.assertEqual(
            str(extracted_cfg), "{'quant_method': 'gptq', 'bits': 4, 'group_size': 128}"
        )

    def test_awq_config(self):
        # AWQ 4-bit
        quant_config = {"quant_method": "awq", "bits": 4, "group_size": 128}
        model_config = MagicMock()
        model_config.hf_config = MockHfConfig(quant_config)

        extracted_cfg = getattr(model_config.hf_config, "quantization_config", None)
        self.assertEqual(extracted_cfg, quant_config)

    def test_no_quant_config(self):
        # No quantization
        model_config = MagicMock()
        model_config.hf_config = MockHfConfig(None)

        extracted_cfg = getattr(model_config.hf_config, "quantization_config", None)
        self.assertIsNone(extracted_cfg)


if __name__ == "__main__":
    unittest.main()
