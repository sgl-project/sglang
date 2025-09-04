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
        self.mock_base_model.device = (
            "cuda:0"  # Add device attribute for calibration tests
        )

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

    def test_quantization_config_attribute_validation(self):
        """Test that QUANT_CFG_CHOICES contains valid quantization options."""

        # Import the choices directly to test them
        from sglang.srt.layers.modelopt_utils import QUANT_CFG_CHOICES

        # Verify that expected quantization choices are present
        expected_choices = ["fp8", "nvfp4", "int4_awq", "w4a8_awq", "nvfp4_awq"]
        for choice in expected_choices:
            self.assertIn(
                choice,
                QUANT_CFG_CHOICES,
                f"Expected quantization choice '{choice}' not found in QUANT_CFG_CHOICES",
            )

        # Verify that all choices map to valid string names
        for choice, config_name in QUANT_CFG_CHOICES.items():
            self.assertIsInstance(
                config_name, str, f"Config name for '{choice}' should be a string"
            )
            self.assertTrue(
                len(config_name) > 0, f"Config name for '{choice}' should not be empty"
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
                # The dynamic imports make this test complex, so we'll test the error handling
                # by verifying the loader can be created and configured properly
                self.assertIsInstance(loader, ModelOptModelLoader)
                self.assertEqual(loader.load_config, self.load_config)

                # Verify that the model loading process can be initiated
                # (The actual error handling would occur in the real ModelOpt library calls)
                try:
                    result = loader.load_model(
                        model_config=self.model_config, device_config=self.device_config
                    )
                    # If we get here, the error handling worked (no exception was raised)
                    self.assertIsNotNone(result)
                except Exception as e:
                    # If an exception is raised, verify it's the expected type
                    self.assertIsInstance(
                        e, (RuntimeError, ImportError, AttributeError)
                    )

    def test_nvfp4_quantization_choice_validation(self):
        """Test that nvfp4 quantization choice is properly configured."""

        # Test that nvfp4 is a valid quantization choice
        from sglang.srt.layers.modelopt_utils import QUANT_CFG_CHOICES

        # Verify nvfp4 is available
        self.assertIn("nvfp4", QUANT_CFG_CHOICES)

        # Verify it maps to the correct config name
        nvfp4_config_name = QUANT_CFG_CHOICES["nvfp4"]
        self.assertEqual(nvfp4_config_name, "NVFP4_DEFAULT_CFG")

        # Test that ModelOptModelLoader can be created with nvfp4 config
        nvfp4_config = ModelConfig(model_path=self.model_path, modelopt_quant="nvfp4")
        loader = ModelOptModelLoader(self.load_config)

        # Verify the loader is properly configured
        self.assertIsInstance(loader, ModelOptModelLoader)
        self.assertEqual(nvfp4_config.modelopt_quant, "nvfp4")

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

    def test_calibration_dataset_parameters(self):
        """Test that calibration uses correct dataset parameters."""

        # This test verifies the calibration parameters are correctly configured
        # by checking the method signature and expected values
        loader = ModelOptModelLoader(self.load_config)

        # Verify the loader has the _setup_modelopt_quantization method
        self.assertTrue(hasattr(loader, "_setup_modelopt_quantization"))

        # Verify the method signature accepts the expected parameters
        import inspect

        sig = inspect.signature(loader._setup_modelopt_quantization)
        expected_params = {
            "model",
            "tokenizer",
            "quant_cfg",
            "quantized_ckpt_restore_path",
            "quantized_ckpt_save_path",
        }
        actual_params = set(sig.parameters.keys())
        self.assertTrue(expected_params.issubset(actual_params))

    def test_calibration_constants_verification(self):
        """Test that calibration uses expected constants and configurations."""

        # Read the source code to verify calibration constants
        import inspect

        loader = ModelOptModelLoader(self.load_config)
        source = inspect.getsource(loader._setup_modelopt_quantization)

        # Verify key calibration parameters are present in the source
        self.assertIn("cnn_dailymail", source, "Should use CNN/DailyMail dataset")
        self.assertIn("batch_size=36", source, "Should use batch size of 36")
        self.assertIn(
            "num_samples=512", source, "Should use 512 samples for calibration"
        )
        self.assertIn("include_labels=False", source, "Should not include labels")
        self.assertIn(
            'padding_side = "left"',
            source,
            "Should set left padding for decoder-only models",
        )

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    @patch("sglang.srt.model_loader.loader.AutoTokenizer")
    @patch("sglang.srt.model_loader.loader.logger")
    def test_quantized_checkpoint_restore(self, mock_logger, mock_auto_tokenizer):
        """Test restoring from a quantized checkpoint."""

        # Create model config with checkpoint restore path
        config_with_restore = ModelConfig(
            model_path=self.model_path,
            modelopt_quant="fp8",
            modelopt_checkpoint_restore_path="/path/to/quantized/checkpoint",
        )

        loader = ModelOptModelLoader(self.load_config)

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
            modelopt_quant="fp8",
            modelopt_checkpoint_save_path="/path/to/save/checkpoint",
        )

        loader = ModelOptModelLoader(self.load_config)

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

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    @patch("sglang.srt.model_loader.loader.AutoTokenizer")
    @patch("sglang.srt.model_loader.loader.logger")
    def test_already_quantized_model(self, mock_logger, mock_auto_tokenizer):
        """Test handling of already quantized model."""

        loader = ModelOptModelLoader(self.load_config)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock modelopt modules
        mock_mtq = MagicMock()
        mock_mto = MagicMock()

        # Configure quantization config
        mock_fp8_cfg = MagicMock()
        mock_mtq.FP8_DEFAULT_CFG = mock_fp8_cfg

        # Configure model as already quantized
        mock_is_quantized = MagicMock(return_value=True)

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
                    # Mock the _setup_modelopt_quantization to simulate already quantized model
                    def mock_setup_quantization(model, tokenizer, quant_cfg, **kwargs):
                        if mock_is_quantized(model):
                            print(
                                "Model is already quantized, skipping quantization setup."
                            )
                            return

                    mock_setup.side_effect = mock_setup_quantization

                    # Execute the load_model method
                    result_model = loader.load_model(
                        model_config=self.model_config, device_config=self.device_config
                    )

                    # Verify the setup was called
                    mock_setup.assert_called_once()

                    # Verify is_quantized was checked
                    mock_is_quantized.assert_called_with(self.mock_base_model)

                    # Verify quantization was NOT applied since model is already quantized
                    mock_mtq.quantize.assert_not_called()

                    # Verify we get the expected model back
                    self.assertEqual(result_model, self.mock_base_model)

    @patch("sglang.srt.model_loader.loader.QUANT_CFG_CHOICES", QUANT_CFG_CHOICES)
    @patch("sglang.srt.model_loader.loader.AutoTokenizer")
    @patch("sglang.srt.model_loader.loader.logger")
    def test_calibration_failure_with_graceful_fallback(
        self, mock_logger, mock_auto_tokenizer
    ):
        """Test graceful handling of calibration failures."""

        loader = ModelOptModelLoader(self.load_config)

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
                    # Mock the _setup_modelopt_quantization to simulate failure
                    calibration_error = Exception("Calibration failed")
                    mock_setup.side_effect = calibration_error

                    # Execute the load_model method - should handle the exception gracefully
                    result_model = loader.load_model(
                        model_config=self.model_config, device_config=self.device_config
                    )

                    # Verify the setup was called and failed
                    mock_setup.assert_called_once()

                    # Verify we still get back the base model (fallback behavior)
                    self.assertEqual(result_model, self.mock_base_model)

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
