"""
Unit tests for ModelOpt export functionality in SGLang.

These tests verify the integration of ModelOpt export API with SGLang's model loading
and quantization workflow.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import ModelOptModelLoader


class TestModelOptExport(unittest.TestCase):
    """Test suite for ModelOpt export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.export_dir = os.path.join(self.temp_dir, "exported_model")
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoint")

        # Mock model
        self.mock_model = Mock(spec=torch.nn.Module)
        self.mock_model.device = torch.device("cuda:0")

        # Mock tokenizer
        self.mock_tokenizer = Mock()

        # Mock quantization config
        self.mock_quant_cfg = Mock()

        # Create ModelOptModelLoader instance
        self.load_config = LoadConfig()
        self.model_loader = ModelOptModelLoader(self.load_config)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_export_files(self, export_dir: str):
        """Create mock export files for testing validation."""
        os.makedirs(export_dir, exist_ok=True)

        # Create config.json
        config = {
            "model_type": "test_model",
            "architectures": ["TestModel"],
            "quantization_config": {
                "quant_method": "modelopt",
                "bits": 8,
            },
        }
        with open(os.path.join(export_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create tokenizer_config.json
        tokenizer_config = {"tokenizer_class": "TestTokenizer"}
        with open(os.path.join(export_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)

        # Create model file
        with open(os.path.join(export_dir, "model.safetensors"), "w") as f:
            f.write("mock_model_data")

    @patch("sglang.srt.model_loader.loader.os.makedirs")
    @patch("modelopt.torch.export.export_hf_checkpoint")
    def test_export_modelopt_checkpoint_success(self, mock_export, mock_makedirs):
        """Test successful model export."""
        # Arrange
        mock_export.return_value = None
        mock_makedirs.return_value = None

        # Act
        self.model_loader._export_modelopt_checkpoint(self.mock_model, self.export_dir)

        # Assert
        mock_makedirs.assert_called_once_with(self.export_dir, exist_ok=True)
        mock_export.assert_called_once_with(self.mock_model, export_dir=self.export_dir)

    @patch("modelopt.torch.export.export_hf_checkpoint")
    def test_export_modelopt_checkpoint_import_error(self, mock_export):
        """Test handling of import error when ModelOpt export is not available."""
        # Arrange
        with patch.dict("sys.modules", {"modelopt.torch.export": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):

                # Act & Assert
                with self.assertRaises(ImportError) as context:
                    self.model_loader._export_modelopt_checkpoint(
                        self.mock_model, self.export_dir
                    )

                self.assertIn(
                    "ModelOpt export functionality is not available",
                    str(context.exception),
                )

    @patch("modelopt.torch.export.export_hf_checkpoint")
    def test_export_modelopt_checkpoint_export_failure(self, mock_export):
        """Test handling of export failure."""
        # Arrange
        mock_export.side_effect = RuntimeError("Export failed")

        # Act & Assert
        with self.assertRaises(RuntimeError):
            self.model_loader._export_modelopt_checkpoint(
                self.mock_model, self.export_dir
            )

    @patch("modelopt.torch.opt.restore")
    @patch("modelopt.torch.quantization.utils.is_quantized")
    def test_setup_quantization_with_export_from_checkpoint(
        self, mock_is_quantized, mock_restore
    ):
        """Test export functionality when restoring from checkpoint."""
        # Arrange
        mock_is_quantized.return_value = False
        mock_restore.return_value = None

        with patch.object(
            self.model_loader, "_export_modelopt_checkpoint"
        ) as mock_export:
            # Act
            self.model_loader._setup_modelopt_quantization(
                self.mock_model,
                self.mock_tokenizer,
                self.mock_quant_cfg,
                quantized_ckpt_restore_path=self.checkpoint_dir,
                export_path=self.export_dir,
            )

            # Assert
            mock_restore.assert_called_once_with(self.mock_model, self.checkpoint_dir)
            mock_export.assert_called_once_with(self.mock_model, self.export_dir)

    @patch("modelopt.torch.quantization.quantize")
    @patch("modelopt.torch.quantization.print_quant_summary")
    @patch("modelopt.torch.quantization.utils.is_quantized")
    @patch("modelopt.torch.utils.dataset_utils.get_dataset_dataloader")
    @patch("modelopt.torch.utils.dataset_utils.create_forward_loop")
    def test_setup_quantization_with_export_after_calibration(
        self,
        mock_create_loop,
        mock_get_dataloader,
        mock_is_quantized,
        mock_print_summary,
        mock_quantize,
    ):
        """Test export functionality after calibration-based quantization."""
        # Arrange
        mock_is_quantized.return_value = False
        mock_dataloader = Mock()
        mock_get_dataloader.return_value = mock_dataloader
        mock_calibrate_loop = Mock()
        mock_create_loop.return_value = mock_calibrate_loop
        mock_quantize.return_value = None
        mock_print_summary.return_value = None

        with patch.object(
            self.model_loader, "_export_modelopt_checkpoint"
        ) as mock_export:
            # Act
            self.model_loader._setup_modelopt_quantization(
                self.mock_model,
                self.mock_tokenizer,
                self.mock_quant_cfg,
                export_path=self.export_dir,
            )

            # Assert
            mock_quantize.assert_called_once_with(
                self.mock_model, self.mock_quant_cfg, forward_loop=mock_calibrate_loop
            )
            mock_export.assert_called_once_with(self.mock_model, self.export_dir)

    def test_setup_quantization_without_export(self):
        """Test quantization setup without export path specified."""
        with patch("modelopt.torch.quantization.utils.is_quantized", return_value=True):
            # Act
            with patch.object(
                self.model_loader, "_export_modelopt_checkpoint"
            ) as mock_export:
                self.model_loader._setup_modelopt_quantization(
                    self.mock_model,
                    self.mock_tokenizer,
                    self.mock_quant_cfg,
                    export_path=None,  # No export path
                )

                # Assert
                mock_export.assert_not_called()

    def test_model_config_export_path_integration(self):
        """Test that ModelConfig properly handles export path."""
        # Arrange & Act
        model_config = ModelConfig(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            modelopt_quant="fp8",
            modelopt_export_path=self.export_dir,
        )

        # Assert
        self.assertEqual(model_config.modelopt_export_path, self.export_dir)

    def test_quantize_and_serve_config_validation(self):
        """Test quantize_and_serve configuration validation."""
        # Test valid configuration
        model_config = ModelConfig(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            quantization="modelopt_fp8",
            quantize_and_serve=True,
        )
        self.assertTrue(model_config.quantize_and_serve)

        # Test invalid configuration - no quantization
        with self.assertRaises(ValueError) as context:
            ModelConfig(
                model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                quantize_and_serve=True,
            )
        self.assertIn("requires ModelOpt quantization", str(context.exception))

    def test_quantize_and_serve_with_pre_quantized_model(self):
        """Test that quantize_and_serve fails with pre-quantized models."""
        with patch.object(ModelConfig, "_is_already_quantized", return_value=True):
            with self.assertRaises(ValueError) as context:
                ModelConfig(
                    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    quantization="modelopt_fp8",
                    quantize_and_serve=True,
                )
            self.assertIn(
                "cannot be used with pre-quantized models", str(context.exception)
            )

    def test_quantize_and_serve_workflow_selection(self):
        """Test that quantize_and_serve selects the correct workflow."""
        with patch(
            "modelopt.torch.quantization.utils.is_quantized", return_value=False
        ):
            with patch.object(
                self.model_loader, "_quantize_and_serve_workflow"
            ) as mock_serve:
                with patch.object(self.model_loader, "_load_modelopt_base_model"):
                    mock_serve.return_value = Mock()

                    # Create model config with quantize_and_serve=True
                    model_config = ModelConfig(
                        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        quantization="modelopt_fp8",
                        quantize_and_serve=True,
                    )
                    device_config = DeviceConfig()

                    # Act
                    self.model_loader.load_model(
                        model_config=model_config,
                        device_config=device_config,
                    )

                    # Assert
                    mock_serve.assert_called_once_with(model_config, device_config)

    def test_standard_workflow_selection(self):
        """Test that standard workflow is selected by default."""
        with patch(
            "modelopt.torch.quantization.utils.is_quantized", return_value=False
        ):
            with patch.object(
                self.model_loader, "_standard_quantization_workflow"
            ) as mock_standard:
                with patch.object(self.model_loader, "_load_modelopt_base_model"):
                    mock_standard.return_value = Mock()

                    # Create model config without quantize_and_serve
                    model_config = ModelConfig(
                        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        quantization="modelopt_fp8",
                        quantize_and_serve=False,
                    )
                    device_config = DeviceConfig()

                    # Act
                    self.model_loader.load_model(
                        model_config=model_config,
                        device_config=device_config,
                    )

                    # Assert
                    mock_standard.assert_called_once_with(model_config, device_config)

    @patch("modelopt.torch.quantization.quantize")
    @patch("modelopt.torch.quantization.utils.is_quantized")
    @patch("modelopt.torch.utils.dataset_utils.get_dataset_dataloader")
    @patch("modelopt.torch.utils.dataset_utils.create_forward_loop")
    def test_quantize_and_serve_workflow_no_export(
        self, mock_create_loop, mock_get_dataloader, mock_is_quantized, mock_quantize
    ):
        """Test that quantize-and-serve workflow doesn't export."""
        # Arrange
        mock_is_quantized.return_value = False
        mock_dataloader = Mock()
        mock_get_dataloader.return_value = mock_dataloader
        mock_calibrate_loop = Mock()
        mock_create_loop.return_value = mock_calibrate_loop
        mock_quantize.return_value = None

        model_config = ModelConfig(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            quantization="modelopt_fp8",
            quantize_and_serve=True,
            modelopt_export_path=self.export_dir,  # Should be ignored
        )
        device_config = DeviceConfig()

        with patch.object(
            self.model_loader, "_setup_modelopt_quantization"
        ) as mock_setup:
            with patch.object(
                self.model_loader, "_load_modelopt_base_model"
            ) as mock_load_base:
                mock_load_base.return_value = self.mock_model

                # Act
                result = self.model_loader._quantize_and_serve_workflow(
                    model_config, device_config
                )

                # Assert
                mock_setup.assert_called_once()
                args, kwargs = mock_setup.call_args
                self.assertIsNone(kwargs.get("export_path"))  # No export in serve mode
                self.assertIsNotNone(result)

    def test_validate_export_success(self):
        """Test validation of a valid export directory."""
        # Arrange
        self._create_mock_export_files(self.export_dir)

        # Act
        result = self._validate_export(self.export_dir)

        # Assert
        self.assertTrue(result)

    def test_validate_export_missing_directory(self):
        """Test validation of non-existent export directory."""
        # Act
        result = self._validate_export("/non/existent/path")

        # Assert
        self.assertFalse(result)

    def test_validate_export_missing_config(self):
        """Test validation when config.json is missing."""
        # Arrange
        os.makedirs(self.export_dir, exist_ok=True)
        # Don't create config.json

        # Act
        result = self._validate_export(self.export_dir)

        # Assert
        self.assertFalse(result)

    def test_validate_export_missing_model_files(self):
        """Test validation when model files are missing."""
        # Arrange
        os.makedirs(self.export_dir, exist_ok=True)
        with open(os.path.join(self.export_dir, "config.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(self.export_dir, "tokenizer_config.json"), "w") as f:
            json.dump({}, f)
        # Don't create model files

        # Act
        result = self._validate_export(self.export_dir)

        # Assert
        self.assertFalse(result)

    def test_get_export_info_success(self):
        """Test getting export information from a valid export."""
        # Arrange
        self._create_mock_export_files(self.export_dir)

        # Act
        info = self._get_export_info(self.export_dir)

        # Assert
        self.assertIsNotNone(info)
        self.assertEqual(info["model_type"], "test_model")
        self.assertEqual(info["architectures"], ["TestModel"])
        self.assertEqual(info["quantization_config"]["quant_method"], "modelopt")
        self.assertEqual(info["export_dir"], self.export_dir)

    def test_get_export_info_invalid_export(self):
        """Test getting export information from an invalid export."""
        # Act
        info = self._get_export_info("/non/existent/path")

        # Assert
        self.assertIsNone(info)

    def test_export_error_handling_in_setup(self):
        """Test that export errors don't break the quantization process."""
        with patch("modelopt.torch.quantization.utils.is_quantized", return_value=True):
            with patch.object(
                self.model_loader,
                "_export_modelopt_checkpoint",
                side_effect=Exception("Export failed"),
            ):
                # Act - should not raise exception
                try:
                    self.model_loader._setup_modelopt_quantization(
                        self.mock_model,
                        self.mock_tokenizer,
                        self.mock_quant_cfg,
                        export_path=self.export_dir,
                    )
                except Exception as e:
                    self.fail(
                        f"Setup should handle export errors gracefully, but got: {e}"
                    )

    # Helper methods (extracted from the deleted utility module)
    def _validate_export(self, export_dir: str) -> bool:
        """Validate that an exported model directory contains the expected files."""
        required_files = ["config.json", "tokenizer_config.json"]
        model_files = ["model.safetensors", "pytorch_model.bin"]

        if not os.path.exists(export_dir):
            return False

        # Check required files
        for file in required_files:
            if not os.path.exists(os.path.join(export_dir, file)):
                return False

        # Check for at least one model file
        has_model_file = any(
            os.path.exists(os.path.join(export_dir, file)) for file in model_files
        )

        return has_model_file

    def _get_export_info(self, export_dir: str) -> dict:
        """Get information about an exported model."""
        if not self._validate_export(export_dir):
            return None

        try:
            config_path = os.path.join(export_dir, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)

            return {
                "model_type": config.get("model_type", "unknown"),
                "architectures": config.get("architectures", []),
                "quantization_config": config.get("quantization_config", {}),
                "export_dir": export_dir,
            }
        except Exception:
            return None


class TestModelOptExportIntegration(unittest.TestCase):
    """Integration tests for ModelOpt export with full model loading workflow."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.export_dir = os.path.join(self.temp_dir, "exported_model")

    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("sglang.srt.model_loader.loader.get_model_architecture")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_full_workflow_with_export(self, mock_model, mock_tokenizer, mock_arch):
        """Test the complete workflow from model config to export."""
        # Arrange
        mock_arch.return_value = ("TestModel", "TestConfig")
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock(spec=torch.nn.Module)

        model_config = ModelConfig(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            modelopt_quant="fp8",
            modelopt_export_path=self.export_dir,
        )

        load_config = LoadConfig()
        device_config = DeviceConfig()

        # Mock the quantization and export process
        with patch.object(
            ModelOptModelLoader, "_setup_modelopt_quantization"
        ) as mock_setup:
            with patch.object(
                ModelOptModelLoader, "_load_modelopt_base_model"
            ) as mock_load_base:
                mock_load_base.return_value = mock_model.return_value

                # Act
                model_loader = ModelOptModelLoader(load_config)
                result = model_loader.load_model(
                    model_config=model_config,
                    device_config=device_config,
                )

                # Assert
                self.assertIsNotNone(result)
                mock_setup.assert_called_once()
                # Verify export_path was passed to setup
                args, kwargs = mock_setup.call_args
                self.assertEqual(kwargs.get("export_path"), self.export_dir)


if __name__ == "__main__":
    unittest.main()
