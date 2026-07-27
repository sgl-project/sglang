"""
Unit tests for ModelOptModelLoader class.

This test module verifies the functionality of ModelOptModelLoader, which
applies NVIDIA Model Optimizer quantization to models during loading.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.layers.modelopt_utils import QUANT_CFG_CHOICES
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4A16LinearMethod,
)
from sglang.srt.model_loader.loader import ModelOptModelLoader
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Note: PYTHONPATH=python should be set when running tests

# Constants for calibration parameters to avoid hard-coded values
CALIBRATION_BATCH_SIZE = 36
CALIBRATION_NUM_SAMPLES = 512
DEFAULT_DEVICE = "cuda:0"

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")


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
        self.device_config = DeviceConfig(device=get_device())

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


class TestParseQuantHfConfig(CustomTestCase):
    """Tests for _parse_quant_hf_config and _parse_modelopt_quant_config.

    Regression tests for the fix where quant_method='modelopt' ignoring quant_algo.
    """

    # (quant_config_input, expected_quant_method)
    _MODELOPT_CASES = [
        ({"quant_method": "modelopt", "quant_algo": "FP8"}, "modelopt_fp8"),
        ({"quant_method": "modelopt", "quant_algo": "FP4"}, "modelopt_fp4"),
        ({"quant_method": "modelopt", "quant_algo": "NVFP4"}, "modelopt_fp4"),
        ({"quant_algo": "NVFP4_AWQ"}, "modelopt_fp4"),
        ({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}, "w4afp8"),
        ({"quant_algo": "FP8"}, "modelopt_fp8"),
        ({"quant_algo": "FP4"}, "modelopt_fp4"),
        ({"quant_algo": "MIXED_PRECISION"}, "w4afp8"),
        ({"quant_method": "modelopt"}, "modelopt"),
    ]

    def setUp(self):
        """Set up a real ModelConfig using TinyLlama (already used elsewhere)."""
        self.mock_tp_rank = patch(
            "sglang.srt.distributed.parallel_state.get_tensor_model_parallel_rank",
            return_value=0,
        )
        self.mock_tp_rank.start()

        self.mock_mp_is_initialized = patch(
            "sglang.srt.distributed.parallel_state.model_parallel_is_initialized",
            return_value=True,
        )
        self.mock_mp_is_initialized.start()

        self.model_config = ModelConfig(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )

    def tearDown(self):
        self.mock_tp_rank.stop()
        self.mock_mp_is_initialized.stop()

    def test_modelopt_quant_parsing(self):
        """Modelopt quant configs must resolve to the correct quant_method."""
        for quant_cfg_input, expected in self._MODELOPT_CASES:
            with self.subTest(quant_cfg=quant_cfg_input):
                self.model_config.hf_config.quantization_config = dict(quant_cfg_input)
                result = self.model_config._parse_quant_hf_config()
                self.assertEqual(result["quant_method"], expected)

    def test_awq_flat_config_defaults_group_size(self):
        """NVFP4_AWQ flat config.json omits group_size; from_config must default it to 16."""
        cfg = ModelOptFp4Config.from_config(
            {
                "quant_algo": "NVFP4_AWQ",
                "ignore": ["lm_head"],
                "quant_method": "modelopt",
            }
        )
        self.assertEqual(cfg.group_size, 16)
        self.assertTrue(cfg.is_awq)

    def test_non_modelopt_quant_method_unchanged(self):
        """Non-modelopt quant_method (e.g. 'gptq') must NOT enter the modelopt path."""
        self.model_config.hf_config.quantization_config = {
            "quant_method": "gptq",
            "bits": 4,
        }
        result = self.model_config._parse_quant_hf_config()
        self.assertEqual(result["quant_method"], "gptq")
        self.assertNotIn("quant_algo", result)


class TestModelOptMixedPrecisionConfig(CustomTestCase):
    def test_nemotron_mixed_precision_with_nvfp4_layers_uses_modelopt_mixed(self):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = MagicMock()
        model_config.hf_config.model_type = "nemotron_h"
        model_config.hf_config.architectures = ["NemotronHForCausalLM"]

        result = model_config._parse_modelopt_quant_config(
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
                        "backbone.layers.0.mixer.out_proj": {"quant_algo": "FP8"},
                        "backbone.layers.1.mixer.experts.0.up_proj": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                        },
                        "backbone.layers.1.mixer.experts.0.down_proj": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                        },
                    },
                }
            }
        )

        self.assertEqual(result["quant_method"], "modelopt_mixed")

    def test_qwen_mixed_precision_with_nvfp4a16_layers_uses_modelopt_mixed(self):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = MagicMock()
        model_config.hf_config.model_type = "qwen3_5_moe"
        model_config.hf_config.architectures = ["Qwen3_5MoeForConditionalGeneration"]

        result = model_config._parse_modelopt_quant_config(
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                        "model.language_model.layers.0.mlp.shared_expert.up_proj": {
                            "quant_algo": "W4A16_NVFP4",
                            "group_size": 16,
                        },
                        "model.language_model.layers.0.linear_attn.in_proj_qkv": {
                            "quant_algo": "FP8"
                        },
                    },
                }
            }
        )

        self.assertEqual(result["quant_method"], "modelopt_mixed")

    def test_mixed_precision_override_does_not_hijack_w4afp8(self):
        self.assertIsNone(
            ModelOptMixedPrecisionConfig.override_quantization_method(
                {"quant_method": "w4afp8", "quant_algo": "MIXED_PRECISION"},
                "w4afp8",
            )
        )

    @patch(
        "sglang.srt.layers.quantization.modelopt_quant.envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.get",
        return_value=True,
    )
    def test_explicit_nvfp4_per_token_activation_false_overrides_env(self, _):
        config = ModelOptFp4Config(use_per_token_activation=False)

        self.assertFalse(config.use_per_token_activation)

    def test_lm_head_guard_accepts_modelopt_fp4_marlin_runtime_state(self):
        lm_head = nn.Module()
        lm_head.weight = nn.Parameter(
            torch.empty(128, 496640, dtype=torch.int32), requires_grad=False
        )
        lm_head.weight_scale = nn.Parameter(torch.empty(1))
        lm_head.weight_global_scale = nn.Parameter(torch.empty(1))
        lm_head.workspace = torch.empty(1)
        lm_head.input_size_per_partition = 2048
        lm_head.output_size_per_partition = 128000

        self.assertTrue(
            should_apply_lm_head_quant_method(
                lm_head, ModelOptNvFp4A16LinearMethod(ModelOptFp4Config())
            )
        )

    def test_lm_head_guard_rejects_stale_modelopt_fp4_method_on_dense_head(self):
        lm_head = nn.Module()
        lm_head.weight = nn.Parameter(torch.empty(128000, 2048))

        self.assertFalse(
            should_apply_lm_head_quant_method(
                lm_head, ModelOptFp4LinearMethod(ModelOptFp4Config())
            )
        )

    def test_lm_head_guard_rejects_stale_modelopt_fp4_attrs_on_dense_head(self):
        lm_head = nn.Module()
        lm_head.weight = nn.Parameter(torch.empty(128000, 2048))
        lm_head.weight_scale = nn.Parameter(torch.empty(1))
        lm_head.weight_global_scale = nn.Parameter(torch.empty(1))
        lm_head.workspace = torch.empty(1)
        lm_head.input_size_per_partition = 2048
        lm_head.output_size_per_partition = 128000

        self.assertFalse(
            should_apply_lm_head_quant_method(
                lm_head, ModelOptNvFp4A16LinearMethod(ModelOptFp4Config())
            )
        )

    def test_mixed_precision_quant_layer_resolution_after_mapping(self):
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
                    "backbone.layers.1.mixer.experts.0.up_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                    "backbone.layers.2.mixer.q_proj": {"quant_algo": "FP8"},
                    "backbone.layers.2.mixer.k_proj": {"quant_algo": "FP8"},
                    "backbone.layers.2.mixer.v_proj": {"quant_algo": "FP8"},
                },
                "packed_modules_mapping": {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                },
            }
        )
        quant_config.apply_weight_name_mapper(
            WeightsMapper(orig_to_new_prefix={"backbone.": "model."})
        )

        self.assertEqual(
            quant_config._resolve_quant_algo("model.layers.0.mixer.in_proj"),
            "FP8",
        )
        self.assertEqual(
            quant_config._resolve_quant_algo("model.layers.1.mixer.experts"),
            "NVFP4",
        )
        self.assertEqual(
            quant_config._resolve_quant_algo("model.layers.2.mixer.qkv_proj"),
            "FP8",
        )


if __name__ == "__main__":
    unittest.main()
