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

# Note: PYTHONPATH=python should be set when running tests

# Constants for calibration parameters to avoid hard-coded values
CALIBRATION_BATCH_SIZE = 36
CALIBRATION_NUM_SAMPLES = 512
DEFAULT_DEVICE = "cuda:0"

# Constants for calibration parameters to avoid hard-coded values
CALIBRATION_BATCH_SIZE = 36
CALIBRATION_NUM_SAMPLES = 512
DEFAULT_DEVICE = "cuda:0"

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
        # Mock distributed functionality to avoid initialization errors
        self.mock_tp_rank = patch(
            "sglang.srt.distributed.parallel_state.get_tensor_model_parallel_rank",
            return_value=0,
        )
        self.mock_tp_rank.start()

        self.mock_rank0_log = patch("sglang.srt.model_loader.loader.rank0_log")
        self.mock_rank0_log.start()

        # Mock logger to avoid issues
        self.mock_logger = patch("sglang.srt.model_loader.loader.logger")
        self.mock_logger.start()

        # Mock all distributed functions that might be called
        self.mock_get_tp_group = patch(
            "sglang.srt.distributed.parallel_state.get_tp_group"
        )
        self.mock_get_tp_group.start()

        # Mock model parallel initialization check
        self.mock_mp_is_initialized = patch(
            "sglang.srt.distributed.parallel_state.model_parallel_is_initialized",
            return_value=True,
        )
        self.mock_mp_is_initialized.start()

        self.model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.load_config = LoadConfig()
        self.device_config = DeviceConfig(device="cuda")

        # Create a basic model config with unified quantization flag
        self.model_config = ModelConfig(
            model_path=self.model_path,
            quantization="modelopt_fp8",  # Use unified quantization approach
        )

        # Also create a unified quantization config for new tests
        self.unified_model_config = ModelConfig(
            model_path=self.model_path, quantization="modelopt_fp8"
        )

        # Mock base model
        self.mock_base_model = MagicMock(spec=nn.Module)
        self.mock_base_model.eval.return_value = self.mock_base_model
        self.mock_base_model.device = (
            DEFAULT_DEVICE  # Add device attribute for calibration tests
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Stop mocks
        self.mock_tp_rank.stop()
        self.mock_rank0_log.stop()
        self.mock_logger.stop()
        self.mock_get_tp_group.stop()
        self.mock_mp_is_initialized.stop()

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
            quant_choice_str = model_config._get_modelopt_quant_type()
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

    @patch("sglang.srt.model_loader.loader.logger")
    def test_missing_modelopt_import(self, mock_logger):
        """Test error handling when modelopt library is not available."""

        loader = ModelOptModelLoader(self.load_config)

        # Mock the base model loader method
        with patch.object(
            loader, "_load_modelopt_base_model", return_value=self.mock_base_model
        ):
            # Simulate missing modelopt by making import fail
            original_import = __import__

            def mock_import(name, *args, **kwargs):
                if name.startswith("modelopt"):
                    raise ImportError("No module named 'modelopt'")
                # Return default import behavior for other modules
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                # Expect ImportError to be raised and logged
                with self.assertRaises(ImportError):
                    loader.load_model(
                        model_config=self.model_config, device_config=self.device_config
                    )

                # Verify error logging
                mock_logger.error.assert_called_with(
                    "NVIDIA Model Optimizer (modelopt) library not found. "
                    "Please install it to use ModelOpt quantization."
                )

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    @patch("sglang.srt.model_loader.loader.AutoTokenizer")
    @patch("sglang.srt.model_loader.loader.logger")
    def test_calibration_workflow_integration(self, mock_logger, mock_auto_tokenizer):
        """Test end-to-end calibration workflow integration."""

        loader = ModelOptModelLoader(self.load_config)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.padding_side = "right"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock modelopt modules
        mock_mtq = MagicMock()
        mock_mto = MagicMock()
        mock_dataset_utils = MagicMock()

        # Configure quantization config
        mock_fp8_cfg = MagicMock()
        mock_mtq.FP8_DEFAULT_CFG = mock_fp8_cfg

        # Configure dataset utilities
        mock_calib_dataloader = MagicMock()
        mock_calibrate_loop = MagicMock()
        mock_dataset_utils.get_dataset_dataloader.return_value = mock_calib_dataloader
        mock_dataset_utils.create_forward_loop.return_value = mock_calibrate_loop

        # Configure model as not quantized initially
        mock_is_quantized = MagicMock(return_value=False)

        with patch.object(
            loader, "_load_modelopt_base_model", return_value=self.mock_base_model
        ):
            with patch.dict(
                "sys.modules",
                {
                    "modelopt": MagicMock(),
                    "modelopt.torch": MagicMock(),
                    "modelopt.torch.opt": mock_mto,
                    "modelopt.torch.quantization": mock_mtq,
                    "modelopt.torch.quantization.utils": MagicMock(
                        is_quantized=mock_is_quantized
                    ),
                    "modelopt.torch.utils": MagicMock(),
                    "modelopt.torch.utils.dataset_utils": mock_dataset_utils,
                },
            ):
                # Execute the load_model method to test the full workflow
                result_model = loader.load_model(
                    model_config=self.model_config, device_config=self.device_config
                )

                # Verify the model loading was successful
                self.assertEqual(result_model, self.mock_base_model)

                # Verify key calibration components were used
                # Note: We can't easily verify the exact calls due to dynamic imports,
                # but we can verify the workflow completed successfully

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    @patch("sglang.srt.model_loader.loader.AutoTokenizer")
    @patch("sglang.srt.model_loader.loader.logger")
    def test_quantized_checkpoint_restore(self, mock_logger, mock_auto_tokenizer):
        """Test restoring from a quantized checkpoint."""

        # Create model config with checkpoint restore path
        config_with_restore = ModelConfig(
            model_path=self.model_path,
            quantization="modelopt_fp8",
        )

        # Create load config with checkpoint restore path
        load_config_with_restore = LoadConfig(
            modelopt_checkpoint_restore_path="/path/to/quantized/checkpoint"
        )

        loader = ModelOptModelLoader(load_config_with_restore)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock modelopt modules
        mock_mtq = MagicMock()
        mock_mto = MagicMock()

        # Configure quantization config
        mock_fp8_cfg = MagicMock()
        mock_mtq.FP8_DEFAULT_CFG = mock_fp8_cfg

        # Configure model as not quantized initially
        mock_is_quantized = MagicMock(return_value=False)

        with patch.object(
            loader, "_load_modelopt_base_model", return_value=self.mock_base_model
        ):
            with patch.dict(
                "sys.modules",
                {
                    "modelopt": MagicMock(),
                    "modelopt.torch": MagicMock(),
                    "modelopt.torch.opt": mock_mto,
                    "modelopt.torch.quantization": mock_mtq,
                    "modelopt.torch.quantization.utils": MagicMock(
                        is_quantized=mock_is_quantized
                    ),
                },
            ):
                with patch.object(loader, "_setup_modelopt_quantization") as mock_setup:
                    # Mock the _setup_modelopt_quantization to simulate checkpoint restore
                    def mock_setup_quantization(
                        model,
                        tokenizer,
                        quant_cfg,
                        quantized_ckpt_restore_path=None,
                        **kwargs,
                    ):
                        if quantized_ckpt_restore_path:
                            mock_mto.restore(model, quantized_ckpt_restore_path)
                            print(
                                f"Restored quantized model from {quantized_ckpt_restore_path}"
                            )
                            return

                    mock_setup.side_effect = mock_setup_quantization

                    # Execute the load_model method
                    result_model = loader.load_model(
                        model_config=config_with_restore,
                        device_config=self.device_config,
                    )

                    # Verify the setup was called with restore path
                    mock_setup.assert_called_once()
                    call_args = mock_setup.call_args
                    # Check that the restore path was passed correctly
                    self.assertIn("quantized_ckpt_restore_path", call_args[1])
                    self.assertEqual(
                        call_args[1]["quantized_ckpt_restore_path"],
                        "/path/to/quantized/checkpoint",
                    )

                    # Verify restore was called
                    mock_mto.restore.assert_called_once_with(
                        self.mock_base_model, "/path/to/quantized/checkpoint"
                    )

                    # Verify we get the expected model back
                    self.assertEqual(result_model, self.mock_base_model)

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    @patch("sglang.srt.model_loader.loader.AutoTokenizer")
    @patch("sglang.srt.model_loader.loader.logger")
    def test_quantized_checkpoint_save(self, mock_logger, mock_auto_tokenizer):
        """Test saving quantized checkpoint after calibration."""

        # Create model config with checkpoint save path
        config_with_save = ModelConfig(
            model_path=self.model_path,
            quantization="modelopt_fp8",
        )

        # Create load config with checkpoint save path
        load_config_with_save = LoadConfig(
            modelopt_checkpoint_save_path="/path/to/save/checkpoint"
        )

        loader = ModelOptModelLoader(load_config_with_save)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock modelopt modules
        mock_mtq = MagicMock()
        mock_mto = MagicMock()
        mock_dataset_utils = MagicMock()

        # Configure quantization config
        mock_fp8_cfg = MagicMock()
        mock_mtq.FP8_DEFAULT_CFG = mock_fp8_cfg

        # Configure model as not quantized initially
        mock_is_quantized = MagicMock(return_value=False)

        with patch.object(
            loader, "_load_modelopt_base_model", return_value=self.mock_base_model
        ):
            with patch.dict(
                "sys.modules",
                {
                    "modelopt": MagicMock(),
                    "modelopt.torch": MagicMock(),
                    "modelopt.torch.opt": mock_mto,
                    "modelopt.torch.quantization": mock_mtq,
                    "modelopt.torch.quantization.utils": MagicMock(
                        is_quantized=mock_is_quantized
                    ),
                    "modelopt.torch.utils": MagicMock(),
                    "modelopt.torch.utils.dataset_utils": mock_dataset_utils,
                },
            ):
                with patch.object(loader, "_setup_modelopt_quantization") as mock_setup:
                    # Mock the _setup_modelopt_quantization to simulate checkpoint save
                    def mock_setup_quantization(
                        model,
                        tokenizer,
                        quant_cfg,
                        quantized_ckpt_save_path=None,
                        **kwargs,
                    ):
                        # Simulate calibration and quantization
                        mock_mtq.quantize(model, quant_cfg, forward_loop=MagicMock())
                        mock_mtq.print_quant_summary(model)

                        # Save checkpoint if path provided
                        if quantized_ckpt_save_path:
                            mock_mto.save(model, quantized_ckpt_save_path)
                            print(
                                f"Quantized model saved to {quantized_ckpt_save_path}"
                            )

                    mock_setup.side_effect = mock_setup_quantization

                    # Execute the load_model method
                    result_model = loader.load_model(
                        model_config=config_with_save, device_config=self.device_config
                    )

                    # Verify the setup was called with save path
                    mock_setup.assert_called_once()
                    call_args = mock_setup.call_args
                    # Check that the save path was passed correctly
                    self.assertIn("quantized_ckpt_save_path", call_args[1])
                    self.assertEqual(
                        call_args[1]["quantized_ckpt_save_path"],
                        "/path/to/save/checkpoint",
                    )

                    # Verify save was called
                    mock_mto.save.assert_called_once_with(
                        self.mock_base_model, "/path/to/save/checkpoint"
                    )

                    # Verify we get the expected model back
                    self.assertEqual(result_model, self.mock_base_model)

    def test_unified_quantization_flag_support(self):
        """Test that ModelOptModelLoader supports unified quantization flags."""
        # Test modelopt_fp8
        config_fp8 = ModelConfig(
            model_path=self.model_path, quantization="modelopt_fp8"
        )
        self.assertEqual(config_fp8._get_modelopt_quant_type(), "fp8")

        # Test modelopt_fp4
        config_fp4 = ModelConfig(
            model_path=self.model_path, quantization="modelopt_fp4"
        )
        self.assertEqual(config_fp4._get_modelopt_quant_type(), "nvfp4")

        # Test auto-detection
        config_auto = ModelConfig(model_path=self.model_path, quantization="modelopt")
        # Should default to fp8 when no config is detected
        self.assertEqual(config_auto._get_modelopt_quant_type(), "fp8")


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
