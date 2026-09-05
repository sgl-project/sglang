"""
Unit tests for ModelOptModelLoader class.

This test module verifies the functionality of ModelOptModelLoader, which
applies NVIDIA Model Optimizer quantization to models during loading.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.layers.modelopt_utils import QUANT_CFG_CHOICES
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4A16LinearMethod,
)
from sglang.srt.model_loader.loader import (
    DefaultModelLoader,
    ModelOptModelLoader,
    get_model_loader,
)
from sglang.srt.model_loader.weight_utils import (
    _modelopt_quant_section,
    get_quant_config,
)
from sglang.srt.models.minimax_m3 import MiniMaxM3SparseForCausalLM
from sglang.srt.models.muse_glimmer import MuseGlimmerForConditionalGeneration
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
        ({"quant_algo": "MXFP8"}, "mxfp8"),
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

    def test_modelopt_mxfp8_config(self):
        """ModelOpt MXFP8 metadata must select block scales and retain FP8 KV policy."""
        model_config = ModelConfig.__new__(ModelConfig)
        for kv_cache_config in (
            {"kv_cache_quant_algo": "FP8"},
            {"kv_cache_scheme": {"type": "float", "num_bits": 8}},
        ):
            with self.subTest(kv_cache_config=kv_cache_config):
                result = model_config._parse_modelopt_quant_config(
                    {
                        "quantization": {
                            "quant_algo": "MXFP8",
                            "group_size": 32,
                            "exclude_modules": ["lm_head"],
                            **kv_cache_config,
                        }
                    }
                )
                self.assertEqual(result["quant_method"], "mxfp8")
                self.assertEqual(result["scale_fmt"], "ue8m0")

                quant_config = Fp8Config.from_config(result)
                self.assertEqual(quant_config.get_name(), "mxfp8")
                self.assertEqual(quant_config.activation_scheme, "dynamic")
                self.assertEqual(quant_config.weight_block_size, [1, 32])
                self.assertIn("lm_head", quant_config.ignored_layers)
                self.assertEqual(quant_config.kv_cache_quant_algo, "FP8")

        nested_result = model_config._parse_modelopt_quant_config(
            {
                "quantization": {
                    "quantization": {
                        "quant_algo": "MXFP8",
                        "group_size": 32,
                        "exclude_modules": ["lm_head"],
                    }
                }
            }
        )
        self.assertEqual(nested_result["quant_method"], "mxfp8")
        self.assertEqual(nested_result["scale_fmt"], "ue8m0")
        self.assertIn("lm_head", nested_result["modules_to_not_convert"])

    def test_modelopt_mxfp8_override(self):
        """Generic ModelOpt selection must not route MXFP8 to scalar FP8."""
        self.assertEqual(
            ModelOptFp8Config.override_quantization_method(
                {"quant_algo": "MXFP8"}, "modelopt"
            ),
            "mxfp8",
        )

    def test_modelopt_mxfp8_weight_loading(self):
        """ModelOpt MXFP8 block scales must reach native scale parameters."""
        weight = torch.empty(1)
        weights = [
            ("model.q_proj.weight_scale", weight),
            ("model.q_proj.input_weight_scale", weight),
            ("model.q_proj.weight_scale_inv", weight),
        ]

        def load_names(quant_config):
            model = nn.Module()
            model.quant_config = quant_config
            loaded_names = []
            model.load_weights = lambda weights: loaded_names.extend(
                name for name, _ in weights
            )
            with patch(
                "sglang.srt.model_loader.loader.is_cuda_alike", return_value=False
            ):
                DefaultModelLoader.load_weights_and_postprocess(
                    model, iter(weights), torch.device("cpu")
                )
            return loaded_names

        mxfp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[1, 32],
            use_mxfp8=True,
        )
        self.assertEqual(
            load_names(mxfp8_config),
            [
                "model.q_proj.weight_scale_inv",
                "model.q_proj.input_weight_scale",
                "model.q_proj.weight_scale_inv",
            ],
        )
        self.assertEqual(
            load_names(Fp8Config(is_checkpoint_fp8_serialized=True)),
            [
                "model.q_proj.weight_scale",
                "model.q_proj.input_weight_scale",
                "model.q_proj.weight_scale_inv",
            ],
        )

    def test_non_modelopt_quant_method_unchanged(self):
        """Non-modelopt quant_method (e.g. 'gptq') must NOT enter the modelopt path."""
        self.model_config.hf_config.quantization_config = {
            "quant_method": "gptq",
            "bits": 4,
        }
        result = self.model_config._parse_quant_hf_config()
        self.assertEqual(result["quant_method"], "gptq")
        self.assertNotIn("quant_algo", result)

    def test_inherited_draft_modelopt_fp4_accepts_fp8_checkpoint(self):
        # ServerArgs has already copied the target's modelopt_fp4 request to the
        # draft. Compatible FP8 metadata must not replace it with plain fp8.
        self.model_config.quantization = "modelopt_fp4"
        self.model_config.is_draft_model = True
        self.model_config.is_draft_quantization_explicit = False
        with (
            patch.object(
                self.model_config,
                "_parse_quant_hf_config",
                return_value={"quant_method": "fp8"},
            ),
            patch.object(
                self.model_config,
                "_find_quant_modelslim_config",
                return_value=None,
            ),
        ):
            self.model_config._verify_quantization()

        # Keeping modelopt_fp4 selects online FP8-to-NVFP4 conversion for
        # eligible MoE experts; this test stops at quantization-method routing.
        self.assertEqual(self.model_config.quantization, "modelopt_fp4")


class TestModelOptFp4LoaderSelection(CustomTestCase):
    def test_draft_modelopt_fp4_uses_checkpoint_exclusions(self):
        cases = (
            # Excluded MTP experts are unpacked, so an explicit draft request
            # replaces the serialized config with online weight quantization.
            ("explicit embedded draft", True, ["mtp.layers.0*"], False),
            # MTP experts present in the serialized checkpoint stay serialized.
            ("explicit serialized draft", True, [], True),
            # Inherited target quantization does not override draft exclusions.
            ("inherited embedded draft", False, ["mtp.layers.0*"], True),
        )
        for name, is_explicit, ignored_layers, is_serialized in cases:
            with self.subTest(name=name):
                model_config = SimpleNamespace(
                    model_path="target-model",
                    quantization="modelopt_fp4",
                    is_draft_model=True,
                    is_draft_quantization_explicit=is_explicit,
                    hf_config=PretrainedConfig(
                        quantization_config={
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                            "ignore": ignored_layers,
                        }
                    ),
                )

                config = get_quant_config(model_config, LoadConfig(), {})

                self.assertEqual(config.get_name(), "modelopt_fp4")
                self.assertEqual(config.is_checkpoint_nvfp4_serialized, is_serialized)

    def test_unquantized_modelopt_fp4_preserves_modelopt_workflows(self):
        model_config = SimpleNamespace(
            quantization="modelopt_fp4",
            _is_already_quantized=lambda: False,
        )

        # Online conversion runs through the regular per-layer weight loaders.
        online_loader = get_model_loader(LoadConfig(), model_config)
        self.assertIsInstance(online_loader, DefaultModelLoader)
        self.assertNotIsInstance(online_loader, ModelOptModelLoader)

        # Explicit ModelOpt checkpoint/export workflows still need its loader.
        for option in (
            "modelopt_checkpoint_restore_path",
            "modelopt_checkpoint_save_path",
            "modelopt_export_path",
        ):
            with self.subTest(option=option):
                loader = get_model_loader(
                    LoadConfig(**{option: "/tmp/modelopt"}), model_config
                )
                self.assertIsInstance(loader, ModelOptModelLoader)


class TestModelOptMixedPrecisionConfig(CustomTestCase):
    def test_fp8_pb_wo_dispatches_to_native_block_fp8(self):
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8_PB_WO"},
                },
                "packed_modules_mapping": {},
            }
        )

        # Type dispatch only needs a LinearBase instance; skip GPU weight setup.
        linear = ReplicatedLinear.__new__(ReplicatedLinear)
        method = quant_config.get_quant_method(
            linear, "model.layers.0.self_attn.q_proj"
        )

        self.assertIsInstance(method, Fp8LinearMethod)
        self.assertEqual(method.quant_config.weight_block_size, [128, 128])
        self.assertTrue(method.quant_config.is_checkpoint_fp8_serialized)
        self.assertEqual(method.quant_config.activation_scheme, "dynamic")

    def test_incomplete_inline_config_falls_back_to_hf_quant_config_file(self):
        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
        file_quantized_layers = {
            "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"}
        }
        file_config = {
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": "FP8",
                "exclude_modules": [],
                "quantized_layers": file_quantized_layers,
            },
        }
        inline_configs = (
            {
                "quant_method": "modelopt_mixed",
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": "NVFP4",
            },
            {
                "quant_method": "modelopt_mixed",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "inline.layer": {"quant_algo": "NVFP4", "group_size": 16}
                },
            },
        )

        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "hf_quant_config.json").write_text(
                json.dumps(file_config), encoding="utf-8"
            )
            for inline_config in inline_configs:
                with self.subTest(inline_config=inline_config):
                    model_config = SimpleNamespace(
                        quantization="modelopt_mixed",
                        hf_config=PretrainedConfig(
                            quantization_config=inline_config,
                        ),
                        model_path=model_path,
                        revision=None,
                        is_draft_model=False,
                        is_draft_quantization_explicit=False,
                    )

                    config = get_quant_config(
                        model_config, LoadConfig(), packed_modules_mapping
                    )

                    self.assertIsInstance(config, ModelOptMixedPrecisionConfig)
                    self.assertEqual(config.quantized_layers, file_quantized_layers)
                    self.assertEqual(config.kv_cache_quant_algo, "FP8")
                    self.assertEqual(
                        config.packed_modules_mapping, packed_modules_mapping
                    )

    @patch("sglang.srt.model_loader.weight_utils.snapshot_download")
    def test_complete_inline_config_does_not_download_metadata(self, mock_download):
        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
        inline_quantized_layers = {
            "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"}
        }
        model_config = SimpleNamespace(
            quantization="modelopt_mixed",
            hf_config=PretrainedConfig(
                quantization_config={
                    "quant_method": "modelopt_mixed",
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_scheme": {"type": "float", "num_bits": 8},
                    "exclude_modules": [],
                    "quantized_layers": inline_quantized_layers,
                }
            ),
            model_path="remote/model",
            revision=None,
            is_draft_model=False,
            is_draft_quantization_explicit=False,
        )

        config = get_quant_config(model_config, LoadConfig(), packed_modules_mapping)

        self.assertIsInstance(config, ModelOptMixedPrecisionConfig)
        self.assertEqual(config.quantized_layers, inline_quantized_layers)
        self.assertEqual(config.kv_cache_quant_algo, "FP8")
        self.assertEqual(config.packed_modules_mapping, packed_modules_mapping)
        mock_download.assert_not_called()

    def test_minimax_mixed_precision_resolves_runtime_names_and_mxfp8(self):
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "weight_block_size": [1, 32],
                "exclude_modules": ["language_model.lm_head"],
                "quantized_layers": {
                    "language_model.model.layers.3.self_attn.q_proj": {
                        "quant_algo": "MXFP8"
                    },
                    "language_model.model.layers.3.self_attn.k_proj": {
                        "quant_algo": "MXFP8"
                    },
                    "language_model.model.layers.3.self_attn.v_proj": {
                        "quant_algo": "MXFP8"
                    },
                    "language_model.model.layers.3.block_sparse_moe.experts.0.w1": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                    "language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj": {
                        "quant_algo": "MXFP8"
                    },
                },
                "packed_modules_mapping": {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                    "gate_up_proj": ["gate_proj", "up_proj"],
                },
            }
        )
        quant_config.apply_weight_name_mapper(
            MiniMaxM3SparseForCausalLM.hf_to_sglang_mapper
        )

        self.assertEqual(
            quant_config._resolve_quant_algo(
                "language_model.model.layers.3.mlp.experts"
            ),
            "NVFP4",
        )
        self.assertEqual(
            quant_config._resolve_quant_algo(
                "language_model.model.layers.3.mlp.shared_experts.gate_up_proj"
            ),
            "MXFP8",
        )

        # Type dispatch only needs a LinearBase instance; skip GPU weight setup.
        linear = ReplicatedLinear.__new__(ReplicatedLinear)
        method = quant_config.get_quant_method(
            linear, "language_model.model.layers.3.self_attn.qkv_proj"
        )
        self.assertIsInstance(method, Fp8LinearMethod)
        self.assertTrue(method.use_mxfp8)
        self.assertEqual(quant_config.mxfp8_config.weight_block_size, [1, 32])
        self.assertEqual(
            quant_config.exclude_modules,
            ["language_model.lm_head", "lm_head"],
        )

    def test_muse_glimmer_mixed_precision_resolves_runtime_names(self):
        """The vendor keys quant metadata under ``model.language_model.*``;
        it must resolve for the ``model.*`` modules the runtime builds.
        """
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.language_model.layers.0.mlp.gate_proj": {
                        "quant_algo": "W4A16_NVFP4",
                        "group_size": 16,
                    },
                    "model.language_model.layers.0.mlp.up_proj": {
                        "quant_algo": "W4A16_NVFP4",
                        "group_size": 16,
                    },
                    "model.language_model.layers.0.self_attn.q_proj": {
                        "quant_algo": "FP8"
                    },
                    "model.language_model.layers.0.self_attn.k_proj": {
                        "quant_algo": "FP8"
                    },
                    "model.language_model.layers.0.self_attn.v_proj": {
                        "quant_algo": "FP8"
                    },
                    "model.language_model.layers.0.self_attn.gate_proj": {
                        "quant_algo": "FP8"
                    },
                    "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                    "model.vision_tower.layers.0.attn.q_proj": {"quant_algo": "FP8"},
                },
                "packed_modules_mapping": (
                    MuseGlimmerForConditionalGeneration.packed_modules_mapping
                ),
            }
        )
        quant_config.apply_weight_name_mapper(
            MuseGlimmerForConditionalGeneration.hf_to_sglang_mapper
        )

        self.assertEqual(
            quant_config._resolve_quant_algo("model.layers.0.mlp.gate_up_proj"),
            "W4A16_NVFP4",
        )
        # Attention stays unfused whenever a quant_config is present, so q/k/v
        # resolve per shard; only the MLP goes through packed_modules_mapping.
        self.assertEqual(
            quant_config._resolve_quant_algo("model.layers.0.self_attn.q_proj"),
            "FP8",
        )
        self.assertEqual(
            quant_config._resolve_quant_algo(
                "model.layers.0.self_attn.output_gate_proj"
            ),
            "FP8",
        )
        self.assertEqual(quant_config._resolve_quant_algo("lm_head"), "W4A16_NVFP4")
        # The vision tower hangs off the entry class, not off ``model``.
        self.assertEqual(
            quant_config._resolve_quant_algo("vision_tower.layers.0.attn.q_proj"),
            "FP8",
        )

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

    def test_flat_hf_quant_config_without_quantization_key(self):
        """Diffusion/unified ModelOpt exports use a flat hf_quant_config.json.

        Regression for Cosmos3-style checkpoints that put quant_algo at the top
        level (no nested ``quantization`` key).
        """
        model_config = ModelConfig.__new__(ModelConfig)

        result = model_config._parse_modelopt_quant_config(
            {
                "quant_method": "modelopt",
                "quant_algo": "FP8",
                "quant_type": "FP8_FP8",
                "ignore": ["lm_head", "visual*"],
            }
        )

        self.assertEqual(result["quant_method"], "modelopt_fp8")
        self.assertEqual(result["quant_algo"], "FP8")

    def test_hf_quant_config_missing_quant_algo_returns_none(self):
        model_config = ModelConfig.__new__(ModelConfig)
        self.assertIsNone(
            model_config._parse_modelopt_quant_config(
                {"quant_method": "modelopt", "producer": {"name": "modelopt"}}
            )
        )

    def test_modelopt_quant_section_supports_nested_and_flat(self):
        nested = {"quantization": {"quant_algo": "FP8", "exclude_modules": ["lm_head"]}}
        self.assertEqual(
            _modelopt_quant_section(nested)["quant_algo"],
            "FP8",
        )

        flat = {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            "ignore": ["lm_head"],
            "producer": {"name": "modelopt"},
        }
        self.assertIs(_modelopt_quant_section(flat), flat)
        self.assertEqual(_modelopt_quant_section(flat)["quant_algo"], "FP8")

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
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )

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

    def test_lm_head_guard_accepts_modelopt_fp4_cutedsl_w4a16_runtime_state(self):
        lm_head = nn.Module()
        lm_head.weight = nn.Parameter(
            torch.empty(128, 1024, dtype=torch.uint8), requires_grad=False
        )
        lm_head.weight_scale_interleaved = nn.Parameter(torch.empty(1))
        lm_head.alpha = nn.Parameter(torch.empty(1))
        lm_head.input_size_per_partition = 2048
        lm_head.output_size_per_partition = 128
        quant_method = ModelOptFp4LinearMethod(ModelOptFp4Config())
        quant_method.quant_mode = "w4a16"

        self.assertTrue(should_apply_lm_head_quant_method(lm_head, quant_method))

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
