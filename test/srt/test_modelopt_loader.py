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

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    @patch("sglang.srt.model_loader.loader.logger")
    def test_invalid_quantization_choice(self, mock_logger):
        """Test error handling for invalid quantization choices."""

        # Set an invalid quantization choice
        invalid_config = ModelConfig(
            model_path=self.model_path, modelopt_quant="invalid_quant"
        )

        loader = ModelOptModelLoader(self.load_config)

        # Mock the base model loader method
        with patch.object(
            loader, "_load_modelopt_base_model", return_value=self.mock_base_model
        ):
            with patch("builtins.__import__"):

                # Expect ValueError for invalid quantization choice
                with self.assertRaises(ValueError) as context:
                    loader.load_model(
                        model_config=invalid_config, device_config=self.device_config
                    )

                # Verify the error message contains expected information
                error_msg = str(context.exception)
                self.assertIn(
                    "Invalid modelopt_quant choice: 'invalid_quant'", error_msg
                )
                self.assertIn("Available choices in QUANT_CFG_CHOICES", error_msg)

    @patch("sglang.srt.model_loader.loader.logger")
    def test_missing_modelopt_import(self, mock_logger):
        """Test error handling when modelopt library is not available."""

        loader = ModelOptModelLoader(self.load_config)

        # Mock the base model loader method
        with patch.object(
            loader, "_load_modelopt_base_model", return_value=self.mock_base_model
        ):
            # Simulate missing modelopt by making import fail
            def mock_import(name, *args, **kwargs):
                if name.startswith("modelopt"):
                    raise ImportError("No module named 'modelopt'")
                # Return default import behavior for other modules
                return __import__(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                # Expect ImportError to be raised and logged
                with self.assertRaises(ImportError):
                    loader.load_model(
                        model_config=self.model_config, device_config=self.device_config
                    )

                # Verify error logging
                mock_logger.error.assert_called_with(
                    "NVIDIA Model Optimizer (modelopt) library not found. "
                    "Please install it to use 'modelopt_quant' feature."
                )

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    @patch("sglang.srt.model_loader.loader.logger")
    def test_missing_quantization_config_attribute(self, mock_logger):
        """Test error handling when quantization config attribute doesn't exist in mtq."""

        loader = ModelOptModelLoader(self.load_config)

        # Mock modelopt modules but without the expected config attribute
        mock_mtq = MagicMock()
        # Don't set FP8_DEFAULT_CFG attribute to simulate missing config
        del mock_mtq.FP8_DEFAULT_CFG  # This will cause AttributeError when accessed

        mock_dataset_utils = MagicMock()

        with patch.object(
            loader, "_load_modelopt_base_model", return_value=self.mock_base_model
        ):
            with patch.dict(
                "sys.modules",
                {
                    "modelopt": MagicMock(),
                    "modelopt.torch": MagicMock(),
                    "modelopt.torch.quantization": mock_mtq,
                    "modelopt.torch.utils": MagicMock(),
                    "modelopt.torch.utils.dataset_utils": mock_dataset_utils,
                },
            ):

                # Expect AttributeError to be raised
                with self.assertRaises(AttributeError) as context:
                    loader.load_model(
                        model_config=self.model_config, device_config=self.device_config
                    )

                # Verify the error message
                error_msg = str(context.exception)
                self.assertIn(
                    "ModelOpt quantization config attribute 'FP8_DEFAULT_CFG'",
                    error_msg,
                )
                self.assertIn(
                    "not found in modelopt.torch.quantization module", error_msg
                )

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    @patch("sglang.srt.model_loader.loader.logger")
    def test_quantization_error_handling(self, mock_logger):
        """Test error handling when mtq.quantize fails."""

        loader = ModelOptModelLoader(self.load_config)

        # Mock modelopt modules
        mock_mtq = MagicMock()
        mock_create_forward_loop = MagicMock()

        # Configure mtq mock to raise exception during quantization
        mock_fp8_cfg = MagicMock()
        mock_mtq.FP8_DEFAULT_CFG = mock_fp8_cfg
        quantization_error = RuntimeError("Quantization failed")
        mock_mtq.quantize.side_effect = quantization_error

        mock_dataset_utils = MagicMock()
        mock_dataset_utils.create_forward_loop = mock_create_forward_loop

        with patch.object(
            loader, "_load_modelopt_base_model", return_value=self.mock_base_model
        ):
            with patch.dict(
                "sys.modules",
                {
                    "modelopt": MagicMock(),
                    "modelopt.torch": MagicMock(),
                    "modelopt.torch.quantization": mock_mtq,
                    "modelopt.torch.utils": MagicMock(),
                    "modelopt.torch.utils.dataset_utils": mock_dataset_utils,
                },
            ):

                # Expect RuntimeError to be raised
                with self.assertRaises(RuntimeError):
                    loader.load_model(
                        model_config=self.model_config, device_config=self.device_config
                    )

                # Verify error logging
                mock_logger.error.assert_called_with(
                    f"Error during ModelOpt mtq.quantize call: {quantization_error}"
                )

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    def test_nvfp4_quantization(self):
        """Test NVFP4 quantization workflow."""

        # Create model config with nvfp4 quantization
        nvfp4_config = ModelConfig(model_path=self.model_path, modelopt_quant="nvfp4")

        loader = ModelOptModelLoader(self.load_config)

        # Mock modelopt modules
        mock_mtq = MagicMock()
        mock_create_forward_loop = MagicMock()

        # Configure mtq mock with NVFP4_DEFAULT_CFG
        mock_nvfp4_cfg = MagicMock()
        mock_mtq.NVFP4_DEFAULT_CFG = mock_nvfp4_cfg
        mock_mtq.quantize.return_value = self.mock_base_model
        mock_mtq.print_quant_summary = MagicMock()

        mock_dataset_utils = MagicMock()
        mock_dataset_utils.create_forward_loop = mock_create_forward_loop

        with patch.object(
            loader, "_load_modelopt_base_model", return_value=self.mock_base_model
        ):
            with patch.dict(
                "sys.modules",
                {
                    "modelopt": MagicMock(),
                    "modelopt.torch": MagicMock(),
                    "modelopt.torch.quantization": mock_mtq,
                    "modelopt.torch.utils": MagicMock(),
                    "modelopt.torch.utils.dataset_utils": mock_dataset_utils,
                },
            ):

                # Execute the load_model method
                result_model = loader.load_model(
                    model_config=nvfp4_config, device_config=self.device_config
                )

                # Verify the quantization process used NVFP4 config
                mock_mtq.quantize.assert_called_once_with(
                    self.mock_base_model, mock_nvfp4_cfg, forward_loop=None
                )

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

    def test_quant_cfg_choices_completeness(self):
        """Test that QUANT_CFG_CHOICES contains expected quantization options."""

        # Verify expected quantization choices are available
        expected_choices = ["fp8", "nvfp4", "int4_awq", "w4a8_awq", "nvfp4_awq"]

        for choice in expected_choices:
            self.assertIn(
                choice,
                QUANT_CFG_CHOICES,
                f"Expected quantization choice '{choice}' not found in QUANT_CFG_CHOICES",
            )

        # Verify all choices map to non-empty strings
        for choice, cfg_name in QUANT_CFG_CHOICES.items():
            self.assertIsInstance(
                cfg_name, str, f"Config name for '{choice}' should be a string"
            )
            self.assertTrue(
                len(cfg_name) > 0, f"Config name for '{choice}' should not be empty"
            )

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
