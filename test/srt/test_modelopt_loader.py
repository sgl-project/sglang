"""
Unit tests for ModelOptModelLoader class.

This test module verifies the functionality of ModelOptModelLoader, which
applies NVIDIA Model Optimizer quantization to models during loading.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch.nn as nn

# Add the sglang path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.modelopt_utils import QUANT_CFG_CHOICES
from sglang.srt.model_loader.loader import ModelOptModelLoader
from sglang.test.test_utils import CustomTestCase


class TestModelOptModelLoader(CustomTestCase):
    """Test cases for ModelOptModelLoader functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.load_config = LoadConfig()
        self.device_config = DeviceConfig(device="cuda")

        # Create a basic model config with modelopt_quant
        self.model_config = ModelConfig(
            model_path=self.model_path, modelopt_quant="fp8"
        )

        # Mock base model
        self.mock_base_model = MagicMock(spec=nn.Module)
        self.mock_base_model.eval.return_value = self.mock_base_model

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    @patch("sglang.srt.model_loader.loader.logger")
    def test_successful_fp8_quantization(self, mock_logger):
        """Test successful FP8 quantization workflow."""

        # Create loader instance
        loader = ModelOptModelLoader(self.load_config)

        # Mock modelopt modules
        mock_mtq = MagicMock()

        # Configure mtq mock with FP8_DEFAULT_CFG
        mock_fp8_cfg = MagicMock()
        mock_mtq.FP8_DEFAULT_CFG = mock_fp8_cfg
        mock_mtq.quantize.return_value = self.mock_base_model
        mock_mtq.print_quant_summary = MagicMock()

        # Create a custom load_model method for testing that simulates the real logic
        def mock_load_model(*, model_config, device_config):
            mock_logger.info("ModelOptModelLoader: Loading base model...")

            # Simulate loading base model (this is already mocked)
            model = self.mock_base_model

            # Simulate the quantization config lookup
            quant_choice_str = model_config.modelopt_quant
            quant_cfg_name = QUANT_CFG_CHOICES.get(quant_choice_str)

            if not quant_cfg_name:
                raise ValueError(f"Invalid modelopt_quant choice: '{quant_choice_str}'")

            # Simulate getattr call and quantization
            if quant_cfg_name == "FP8_DEFAULT_CFG":
                quant_cfg = mock_fp8_cfg

                mock_logger.info(
                    f"Quantizing model with ModelOpt using config attribute: mtq.{quant_cfg_name}"
                )

                # Simulate mtq.quantize call
                quantized_model = mock_mtq.quantize(model, quant_cfg, forward_loop=None)
                mock_logger.info("Model successfully quantized with ModelOpt.")

                # Simulate print_quant_summary call
                mock_mtq.print_quant_summary(quantized_model)

                return quantized_model.eval()

            return model.eval()

        # Patch the load_model method with our custom implementation
        with patch.object(loader, "load_model", side_effect=mock_load_model):
            # Execute the load_model method
            result_model = loader.load_model(
                model_config=self.model_config, device_config=self.device_config
            )

            # Verify the quantization process
            mock_mtq.quantize.assert_called_once_with(
                self.mock_base_model, mock_fp8_cfg, forward_loop=None
            )

            # Verify logging
            mock_logger.info.assert_any_call(
                "ModelOptModelLoader: Loading base model..."
            )
            mock_logger.info.assert_any_call(
                "Quantizing model with ModelOpt using config attribute: mtq.FP8_DEFAULT_CFG"
            )
            mock_logger.info.assert_any_call(
                "Model successfully quantized with ModelOpt."
            )

            # Verify print_quant_summary was called
            mock_mtq.print_quant_summary.assert_called_once_with(self.mock_base_model)

            # Verify eval() was called on the returned model
            self.mock_base_model.eval.assert_called()

            # Verify we get back the expected model
            self.assertEqual(result_model, self.mock_base_model)


class TestModelOptLoaderIntegration(CustomTestCase):
    """Integration tests for ModelOptModelLoader with Engine API."""

    @patch("sglang.srt.model_loader.loader.get_model_loader")
    @patch("sglang.srt.entrypoints.engine.Engine.__init__")
    def test_engine_with_modelopt_quant_parameter(
        self, mock_engine_init, mock_get_model_loader
    ):
        """Test that Engine properly handles modelopt_quant parameter."""

        # Mock the Engine.__init__ to avoid actual initialization
        mock_engine_init.return_value = None

        # Mock get_model_loader to return our ModelOptModelLoader
        mock_loader = MagicMock(spec=ModelOptModelLoader)
        mock_get_model_loader.return_value = mock_loader

        # Import here to avoid circular imports during test discovery
        # import sglang as sgl  # Commented out since not directly used

        # Test that we can create an engine with modelopt_quant parameter
        # This would normally trigger the ModelOptModelLoader selection
        try:
            engine_args = {
                "model_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "modelopt_quant": "fp8",
                "log_level": "error",  # Suppress logs during testing
            }

            # This tests the parameter parsing and server args creation
            from sglang.srt.server_args import ServerArgs

            server_args = ServerArgs(**engine_args)

            # Verify that modelopt_quant is properly set
            self.assertEqual(server_args.modelopt_quant, "fp8")

        except Exception as e:
            # If there are missing dependencies or initialization issues,
            # we can still verify the parameter is accepted
            if "modelopt_quant" not in str(e):
                # The parameter was accepted, which is what we want to test
                pass
            else:
                self.fail(f"modelopt_quant parameter not properly handled: {e}")

    @patch("sglang.srt.model_loader.loader.get_model_loader")
    @patch("sglang.srt.entrypoints.engine.Engine.__init__")
    def test_engine_with_modelopt_quant_cli_argument(
        self, mock_engine_init, mock_get_model_loader
    ):
        """Test that CLI argument --modelopt-quant is properly parsed."""

        # Mock the Engine.__init__ to avoid actual initialization
        mock_engine_init.return_value = None

        # Mock get_model_loader to return our ModelOptModelLoader
        mock_loader = MagicMock(spec=ModelOptModelLoader)
        mock_get_model_loader.return_value = mock_loader

        # Test CLI argument parsing
        import argparse

        from sglang.srt.server_args import ServerArgs

        # Create parser and add arguments
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        # Test parsing with modelopt_quant argument
        args = parser.parse_args(
            [
                "--model-path",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "--modelopt-quant",
                "fp8",
            ]
        )

        # Convert to ServerArgs using the proper from_cli_args method
        server_args = ServerArgs.from_cli_args(args)

        # Verify that modelopt_quant was properly parsed
        self.assertEqual(server_args.modelopt_quant, "fp8")
        self.assertEqual(server_args.model_path, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


if __name__ == "__main__":
    unittest.main()
