import unittest
from unittest.mock import patch

import sglang as sgl
from sglang.srt.configs.model_config import ModelConfig
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


class TestSRTEngineWithQuantArgs(CustomTestCase):

    def test_1_quantization_args(self):

        # we only test fp8 because other methods are currently dependent on vllm. We can add other methods back to test after vllm dependency is resolved.
        quantization_args_list = [
            # "awq",
            "fp8",
            # "gptq",
            # "marlin",
            # "gptq_marlin",
            # "awq_marlin",
            # "bitsandbytes",
            # "gguf",
        ]

        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        for quantization_args in quantization_args_list:
            engine = sgl.Engine(
                model_path=model_path, random_seed=42, quantization=quantization_args
            )
            engine.generate(prompt, sampling_params)
            engine.shutdown()

    def test_2_torchao_args(self):

        # we don't test int8dq because currently there is conflict between int8dq and capture cuda graph
        torchao_args_list = [
            # "int8dq",
            "int8wo",
            "fp8wo",
            "fp8dq-per_tensor",
            "fp8dq-per_row",
        ] + [f"int4wo-{group_size}" for group_size in [32, 64, 128, 256]]

        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        for torchao_config in torchao_args_list:
            engine = sgl.Engine(
                model_path=model_path, random_seed=42, torchao_config=torchao_config
            )
            engine.generate(prompt, sampling_params)
            engine.shutdown()

    def test_3_fp8_defaults_to_modelopt(self):
        """Test that FP8 models default to ModelOpt on NVIDIA GPUs when no quantization is specified."""

        mock_model_path = "mock_model_path"
        mock_quant_cfg = {"quant_method": "fp8"}

        with patch("torch.version.hip", None), patch(
            "sglang.srt.configs.model_config.ModelConfig._parse_quant_hf_config",
            return_value=mock_quant_cfg,
        ):
            config = ModelConfig(model_path=mock_model_path, quantization=None)
            self.assertEqual(config.quantization, "modelopt")

    def test_4_user_choice_not_overridden(self):
        """Test that user's explicit quantization choice is not overridden even for FP8 on NVIDIA GPUs."""

        mock_model_path = "mock_model_path"
        mock_quant_cfg = {"quant_method": "fp8"}

        with patch("torch.version.hip", None), patch(
            "sglang.srt.configs.model_config.ModelConfig._parse_quant_hf_config",
            return_value=mock_quant_cfg,
        ):
            config = ModelConfig(model_path=mock_model_path, quantization="fp8")
            self.assertEqual(config.quantization, "fp8")


if __name__ == "__main__":
    unittest.main()
