"""Unit tests for --enable-cp-cache-layer-split ServerArgs guards."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_layer_split_args(**overrides):
    enable_layer_split = overrides.pop("enable_cp_cache_layer_split", True)
    model_path = overrides.pop("model_path", "dummy")
    kwargs = dict(
        model_path="dummy",
        enable_cp_cache_layer_split=False,
        enable_prefill_cp=True,
        cp_strategy="interleave",
        attn_cp_size=4,
        tp_size=8,
        disaggregation_mode="prefill",
    )
    kwargs.update(overrides)
    args = ServerArgs(**kwargs)
    args.model_path = model_path
    args.enable_cp_cache_layer_split = enable_layer_split
    return args


def _dsa_model_config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
        )
    )


class TestCpCacheLayerSplitServerArgs(CustomTestCase):
    def test_accepts_valid_dummy_model_args_when_cuda_graph_config_is_none(self):
        args = _make_layer_split_args()
        args.cuda_graph_config = None

        with patch("sglang.srt.server_args.is_hip", return_value=False):
            args._handle_cp_cache_layer_split()

        self.assertTrue(args.enable_cp_cache_layer_split)
        self.assertFalse(args.disable_cuda_graph)

    def test_rejects_unsupported_model_arch(self):
        args = _make_layer_split_args()
        args.model_path = "unsupported-model"
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"])
        )

        with patch.object(ServerArgs, "get_model_config", return_value=model_config):
            with self.assertRaisesRegex(ValueError, "not supported for model arch"):
                args._handle_cp_cache_layer_split()

    def test_rejects_unsupported_shared_configurations(self):
        cases = (
            ("prefill_cp", {"enable_prefill_cp": False}, "enable-prefill-cp"),
            ("cp_size", {"attn_cp_size": 1}, "attn-cp-size > 1"),
            ("cp_strategy", {"cp_strategy": "zigzag"}, "cp-strategy interleave"),
            ("pipeline_parallelism", {"pp_size": 2}, "pipeline parallelism"),
            (
                "disaggregation_mode",
                {"disaggregation_mode": "null"},
                "prefill-only servers",
            ),
            (
                "transfer_backend",
                {"disaggregation_transfer_backend": "nixl"},
                "mooncake",
            ),
            (
                "hicache_backend",
                {"hicache_storage_backend": "nixl"},
                "Supported backends",
            ),
        )

        for name, overrides, error in cases:
            args = _make_layer_split_args(**overrides)
            with self.subTest(case=name), patch(
                "sglang.srt.server_args.is_hip", return_value=False
            ), self.assertRaisesRegex(ValueError, error):
                args._handle_cp_cache_layer_split()

    def test_disables_only_prefill_cuda_graph_for_dsv4_and_dsa(self):
        cases = (
            ("dsv4", "dummy", None),
            ("dsa", "glm-dsa", _dsa_model_config()),
        )

        for name, model_path, model_config in cases:
            args = _make_layer_split_args(model_path=model_path)
            args.cuda_graph_config = CudaGraphConfig(
                decode=PhaseConfig(backend=Backend.FULL),
                prefill=PhaseConfig(backend=Backend.TC_PIECEWISE),
            )
            with self.subTest(model=name), patch.object(
                ServerArgs, "get_model_config", return_value=model_config
            ), patch("sglang.srt.server_args.is_hip", return_value=False):
                args._handle_cp_cache_layer_split()

            self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
            self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
            self.assertFalse(args.disable_cuda_graph)

    def test_dsv4_rejects_unsupported_runtime_paths(self):
        with envs.SGLANG_OPT_USE_COMPRESSOR_V2.override(False), patch(
            "sglang.srt.server_args.is_hip", return_value=False
        ), self.assertRaisesRegex(ValueError, "Compressor V2"):
            _make_layer_split_args()._handle_cp_cache_layer_split()

        with patch("sglang.srt.server_args.is_hip", return_value=False), patch(
            "sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=True,
        ), self.assertRaisesRegex(ValueError, "unified_kv_triton"):
            _make_layer_split_args()._handle_cp_cache_layer_split()

    def test_dsa_skips_dsv4_only_compressor_v2_guard(self):
        args = _make_layer_split_args(
            model_path="glm-dsa",
            disaggregation_transfer_backend="mooncake",
        )

        with patch.object(
            ServerArgs, "get_model_config", return_value=_dsa_model_config()
        ), envs.SGLANG_OPT_USE_COMPRESSOR_V2.override(False), patch(
            "sglang.srt.server_args.is_hip", return_value=False
        ):
            args._handle_cp_cache_layer_split()

        self.assertTrue(args.enable_cp_cache_layer_split)


if __name__ == "__main__":
    unittest.main()
