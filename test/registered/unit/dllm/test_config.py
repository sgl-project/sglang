"""
Unit tests for sglang.srt.dllm.config.DllmConfig

Tests cover:
- Direct instantiation of DllmConfig
- from_server_args() returns None when dllm_algorithm is None
- from_server_args() correctly parses known architectures (LLaDA2MoeModelLM, SDARForCausalLM, SDARMoeForCausalLM)
- from_server_args() raises RuntimeError for unknown architectures
- from_server_args() correctly parses YAML config file and overrides block_size
- from_server_args() defaults max_running_requests to 1 when None
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import types
import unittest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, mock_open, patch

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without the full SGLang stack
# ---------------------------------------------------------------------------


def _make_stub_modules():
    """Create lightweight stubs for all heavy dependencies pulled in by
    sglang.srt.dllm.config -> sglang.srt.configs.model_config -> …"""

    # Top-level sglang package stub
    sglang_stub = types.ModuleType("sglang")
    sys.modules.setdefault("sglang", sglang_stub)

    srt_stub = types.ModuleType("sglang.srt")
    srt_stub.__path__ = []
    sys.modules.setdefault("sglang.srt", srt_stub)

    # sglang.srt.server_args
    server_args_stub = types.ModuleType("sglang.srt.server_args")

    @dataclass
    class ServerArgs:
        model_path: str = "dummy"
        revision: Optional[str] = None
        dllm_algorithm: Optional[str] = None
        dllm_algorithm_config: Optional[str] = None
        dllm_fdfo: bool = False
        max_running_requests: Optional[int] = None

    server_args_stub.ServerArgs = ServerArgs
    sys.modules["sglang.srt.server_args"] = server_args_stub

    # sglang.srt.configs and sglang.srt.configs.model_config
    configs_stub = types.ModuleType("sglang.srt.configs")
    configs_stub.__path__ = []
    sys.modules["sglang.srt.configs"] = configs_stub

    model_config_stub = types.ModuleType("sglang.srt.configs.model_config")

    class ModelConfig:
        @staticmethod
        def from_server_args(server_args, model_path=None, model_revision=None):
            """Overridden by individual tests via patch."""
            raise NotImplementedError("Should be patched in tests")

    model_config_stub.ModelConfig = ModelConfig
    sys.modules["sglang.srt.configs.model_config"] = model_config_stub

    # Make sglang.srt.dllm a proper package stub so sub-modules are findable
    dllm_stub = types.ModuleType("sglang.srt.dllm")
    dllm_stub.__path__ = []
    sys.modules["sglang.srt.dllm"] = dllm_stub


_make_stub_modules()

# Now load the real source file directly to avoid any import chain issues
import importlib.util as _ilu
import os as _os

_config_path = _os.path.abspath(
    _os.path.join(
        _os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "srt",
        "dllm",
        "config.py",
    )
)
_spec = _ilu.spec_from_file_location("sglang.srt.dllm.config", _config_path)
_mod = _ilu.module_from_spec(_spec)
sys.modules["sglang.srt.dllm.config"] = _mod
_spec.loader.exec_module(_mod)

DllmConfig = _mod.DllmConfig
from sglang.srt.server_args import ServerArgs  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_args(**kwargs) -> ServerArgs:
    args = ServerArgs(model_path="dummy")
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def _make_mock_model_config(arch: str):
    mock = MagicMock()
    mock.hf_config.architectures = [arch]
    return mock


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestDllmConfigInit(unittest.TestCase):
    """Tests for direct __init__ / constructor."""

    def test_all_fields_stored(self):
        cfg = DllmConfig(
            algorithm="low_confidence",
            algorithm_config={"threshold": 0.5},
            block_size=32,
            mask_id=156895,
            max_running_requests=4,
            first_done_first_out_mode=True,
        )
        self.assertEqual(cfg.algorithm, "low_confidence")
        self.assertEqual(cfg.algorithm_config, {"threshold": 0.5})
        self.assertEqual(cfg.block_size, 32)
        self.assertEqual(cfg.mask_id, 156895)
        self.assertEqual(cfg.max_running_requests, 4)
        self.assertTrue(cfg.first_done_first_out_mode)

    def test_default_fdfo_is_false(self):
        cfg = DllmConfig(
            algorithm="x",
            algorithm_config={},
            block_size=1,
            mask_id=0,
            max_running_requests=1,
        )
        self.assertFalse(cfg.first_done_first_out_mode)


class TestDllmConfigFromServerArgs(unittest.TestCase):
    """Tests for the DllmConfig.from_server_args() factory."""

    # ------------------------------------------------------------------ #
    #  None algorithm -> returns None                                      #
    # ------------------------------------------------------------------ #

    def test_returns_none_when_algorithm_is_none(self):
        args = _make_server_args(dllm_algorithm=None)
        result = DllmConfig.from_server_args(args)
        self.assertIsNone(result)

    # ------------------------------------------------------------------ #
    #  Known architectures                                                 #
    # ------------------------------------------------------------------ #

    @patch("sglang.srt.dllm.config.ModelConfig")
    def test_llada2_architecture_defaults(self, mock_cls):
        mock_cls.from_server_args.return_value = _make_mock_model_config(
            "LLaDA2MoeModelLM"
        )
        args = _make_server_args(
            dllm_algorithm="low_confidence", max_running_requests=2
        )

        cfg = DllmConfig.from_server_args(args)

        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.block_size, 32)
        self.assertEqual(cfg.mask_id, 156895)
        self.assertEqual(cfg.algorithm, "low_confidence")
        self.assertEqual(cfg.max_running_requests, 2)

    @patch("sglang.srt.dllm.config.ModelConfig")
    def test_sdar_architecture_defaults(self, mock_cls):
        mock_cls.from_server_args.return_value = _make_mock_model_config(
            "SDARForCausalLM"
        )
        args = _make_server_args(dllm_algorithm="joint_threshold")

        cfg = DllmConfig.from_server_args(args)

        self.assertEqual(cfg.block_size, 4)
        self.assertEqual(cfg.mask_id, 151669)

    @patch("sglang.srt.dllm.config.ModelConfig")
    def test_sdar_moe_architecture_defaults(self, mock_cls):
        mock_cls.from_server_args.return_value = _make_mock_model_config(
            "SDARMoeForCausalLM"
        )
        args = _make_server_args(dllm_algorithm="joint_threshold")

        cfg = DllmConfig.from_server_args(args)

        self.assertEqual(cfg.block_size, 4)
        self.assertEqual(cfg.mask_id, 151669)

    # ------------------------------------------------------------------ #
    #  Unknown architecture                                                #
    # ------------------------------------------------------------------ #

    @patch("sglang.srt.dllm.config.ModelConfig")
    def test_unknown_arch_raises_runtime_error(self, mock_cls):
        mock_cls.from_server_args.return_value = _make_mock_model_config("GhostModel")
        args = _make_server_args(dllm_algorithm="some_algo")

        with self.assertRaises(RuntimeError) as ctx:
            DllmConfig.from_server_args(args)

        self.assertIn("GhostModel", str(ctx.exception))

    # ------------------------------------------------------------------ #
    #  max_running_requests defaults                                       #
    # ------------------------------------------------------------------ #

    @patch("sglang.srt.dllm.config.ModelConfig")
    def test_max_running_requests_defaults_to_1_when_none(self, mock_cls):
        mock_cls.from_server_args.return_value = _make_mock_model_config(
            "LLaDA2MoeModelLM"
        )
        args = _make_server_args(dllm_algorithm="x", max_running_requests=None)

        cfg = DllmConfig.from_server_args(args)

        self.assertEqual(cfg.max_running_requests, 1)

    # ------------------------------------------------------------------ #
    #  YAML config override                                                #
    # ------------------------------------------------------------------ #

    @patch("sglang.srt.dllm.config.ModelConfig")
    def test_yaml_config_overrides_block_size(self, mock_cls):
        mock_cls.from_server_args.return_value = _make_mock_model_config(
            "SDARForCausalLM"
        )
        args = _make_server_args(
            dllm_algorithm="x",
            dllm_algorithm_config="config.yaml",
            max_running_requests=None,
            dllm_fdfo=True,
        )

        yaml_content = "block_size: 16\ncustom_param: hello\n"
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            cfg = DllmConfig.from_server_args(args)

        # block_size from YAML (16) overrides model default (4)
        self.assertEqual(cfg.block_size, 16)
        self.assertEqual(cfg.algorithm_config["custom_param"], "hello")
        self.assertTrue(cfg.first_done_first_out_mode)

    @patch("sglang.srt.dllm.config.ModelConfig")
    def test_yaml_config_without_block_size_keeps_model_default(self, mock_cls):
        mock_cls.from_server_args.return_value = _make_mock_model_config(
            "SDARForCausalLM"
        )
        args = _make_server_args(dllm_algorithm="x", dllm_algorithm_config="cfg.yaml")

        yaml_content = "some_key: 42\n"
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            cfg = DllmConfig.from_server_args(args)

        # No block_size in YAML -> model default (4) preserved
        self.assertEqual(cfg.block_size, 4)


if __name__ == "__main__":
    unittest.main()
