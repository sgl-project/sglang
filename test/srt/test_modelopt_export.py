"""
Unit tests for ModelOpt export functionality in SGLang.

These tests verify the integration of ModelOpt export API with SGLang's model loading
and quantization workflow.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import ModelOptModelLoader

# Note: PYTHONPATH=python should be set when running tests

# Check if modelopt is available
try:
    import modelopt

    MODELOPT_AVAILABLE = True
except ImportError:
    MODELOPT_AVAILABLE = False


class TestModelOptExport(unittest.TestCase):
    """Test suite for ModelOpt export functionality."""

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

        # Stop mocks
        self.mock_tp_rank.stop()
        self.mock_rank0_log.stop()
        self.mock_logger.stop()
        self.mock_get_tp_group.stop()
        self.mock_mp_is_initialized.stop()

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

    @unittest.skipIf(not MODELOPT_AVAILABLE, "nvidia-modelopt not available")
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

    @unittest.skipIf(not MODELOPT_AVAILABLE, "nvidia-modelopt not available")
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
            mock_export.assert_called_once_with(self.mock_model, self.export_dir, None)

    @unittest.skipIf(not MODELOPT_AVAILABLE, "nvidia-modelopt not available")
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
            mock_export.assert_called_once_with(self.mock_model, self.export_dir, None)

    @unittest.skipIf(not MODELOPT_AVAILABLE, "nvidia-modelopt not available")
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

    def test_quantize_and_serve_config_validation(self):
        """Test that quantize_and_serve is properly disabled."""
        # Test that quantize-and-serve mode raises NotImplementedError
        with self.assertRaises(NotImplementedError) as context:
            ModelConfig(
                model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                quantization="modelopt_fp8",
                quantize_and_serve=True,
            )

        # Verify the error message contains helpful instructions
        error_msg = str(context.exception)
        self.assertIn("disabled due to compatibility issues", error_msg)
        self.assertIn("separate quantize-then-deploy workflow", error_msg)

        # Test invalid configuration - no quantization
        with self.assertRaises(ValueError) as context:
            ModelConfig(
                model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                quantize_and_serve=True,
            )
        self.assertIn("requires ModelOpt quantization", str(context.exception))

    @unittest.skipIf(not MODELOPT_AVAILABLE, "nvidia-modelopt not available")
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


@unittest.skipIf(not MODELOPT_AVAILABLE, "nvidia-modelopt not available")
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
