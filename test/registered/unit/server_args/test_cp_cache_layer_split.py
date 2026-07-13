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

    def test_disables_only_prefill_cuda_graph(self):
        args = _make_layer_split_args()
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL),
            prefill=PhaseConfig(backend=Backend.TC_PIECEWISE),
        )

        with patch("sglang.srt.server_args.is_hip", return_value=False):
            args._handle_cp_cache_layer_split()

        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
        self.assertFalse(args.disable_cuda_graph)

    def test_dsa_uses_same_prefill_cuda_graph_behavior(self):
        args = _make_layer_split_args(model_path="glm-dsa")
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL),
            prefill=PhaseConfig(backend=Backend.TC_PIECEWISE),
        )

        with patch.object(
            ServerArgs, "get_model_config", return_value=_dsa_model_config()
        ), patch("sglang.srt.server_args.is_hip", return_value=False):
            args._handle_cp_cache_layer_split()

        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
        self.assertFalse(args.disable_cuda_graph)

    def test_rejects_missing_prefill_cp(self):
        args = _make_layer_split_args(enable_prefill_cp=False)

        with self.assertRaisesRegex(ValueError, "enable-prefill-cp"):
            args._handle_cp_cache_layer_split()

    def test_rejects_non_prefill_server(self):
        args = _make_layer_split_args(disaggregation_mode="null")

        with self.assertRaisesRegex(ValueError, "prefill-only servers"):
            args._handle_cp_cache_layer_split()

    def test_rejects_attn_cp_size_one(self):
        args = _make_layer_split_args(attn_cp_size=1)

        with self.assertRaisesRegex(ValueError, "attn-cp-size > 1"):
            args._handle_cp_cache_layer_split()

    def test_rejects_non_interleave_cp_strategy(self):
        args = _make_layer_split_args(cp_strategy="zigzag")

        with self.assertRaisesRegex(ValueError, "cp-strategy interleave"):
            args._handle_cp_cache_layer_split()

    def test_rejects_pipeline_parallelism_for_dsv4_and_dsa(self):
        cases = (
            ("dsv4", _make_layer_split_args(pp_size=2), None),
            (
                "dsa",
                _make_layer_split_args(model_path="glm-dsa", pp_size=2),
                _dsa_model_config(),
            ),
        )

        for name, args, model_config in cases:
            with self.subTest(model=name), patch.object(
                ServerArgs,
                "get_model_config",
                return_value=model_config,
            ), self.assertRaisesRegex(ValueError, "pipeline parallelism"):
                args._handle_cp_cache_layer_split()

    def test_rejects_nixl_disaggregation_transfer_backend(self):
        args = _make_layer_split_args()
        args.disaggregation_transfer_backend = "nixl"

        with patch("sglang.srt.server_args.is_hip", return_value=False):
            with self.assertRaisesRegex(ValueError, "mooncake"):
                args._handle_cp_cache_layer_split()

    def test_rejects_unified_kv_backend(self):
        args = _make_layer_split_args()

        with patch("sglang.srt.server_args.is_hip", return_value=False), patch(
            "sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "unified_kv_triton"):
                args._handle_cp_cache_layer_split()

    def test_rejects_compressor_v1(self):
        args = _make_layer_split_args()

        with envs.SGLANG_OPT_USE_COMPRESSOR_V2.override(False), patch(
            "sglang.srt.server_args.is_hip", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "Compressor V2"):
                args._handle_cp_cache_layer_split()

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

    def test_dsa_rejects_unsupported_hicache_storage_backend(self):
        args = _make_layer_split_args(
            model_path="glm-dsa",
            disaggregation_transfer_backend="mooncake",
            hicache_storage_backend="nixl",
        )

        with patch.object(
            ServerArgs, "get_model_config", return_value=_dsa_model_config()
        ), patch("sglang.srt.server_args.is_hip", return_value=False):
            with self.assertRaisesRegex(ValueError, "Supported backends"):
                args._handle_cp_cache_layer_split()


if __name__ == "__main__":
    unittest.main()
