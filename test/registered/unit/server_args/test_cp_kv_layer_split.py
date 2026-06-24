"""Unit tests for --enable-cp-kv-layer-split ServerArgs guards."""

import unittest
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
    kwargs = dict(
        model_path="dummy",
        enable_cp_kv_layer_split=True,
        enable_dsa_prefill_context_parallel=True,
        attn_cp_size=4,
        tp_size=8,
        disaggregation_mode="prefill",
    )
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


class TestCpKvLayerSplitServerArgs(CustomTestCase):
    def test_accepts_valid_dummy_model_args_when_cuda_graph_config_is_none(self):
        args = _make_layer_split_args()
        args.cuda_graph_config = None

        with patch("sglang.srt.server_args.is_hip", return_value=False):
            args._handle_cp_kv_layer_split()

        self.assertTrue(args.enable_cp_kv_layer_split)
        self.assertTrue(args.disable_cuda_graph)

    def test_disables_cuda_graph_backends_when_config_is_present(self):
        args = _make_layer_split_args()
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL),
            prefill=PhaseConfig(backend=Backend.TC_PIECEWISE),
        )

        with patch("sglang.srt.server_args.is_hip", return_value=False):
            args._handle_cp_kv_layer_split()

        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
        self.assertTrue(args.disable_cuda_graph)

    def test_rejects_missing_dsa_prefill_cp(self):
        args = _make_layer_split_args(enable_dsa_prefill_context_parallel=False)

        with self.assertRaisesRegex(ValueError, "enable-dsa-prefill-context-parallel"):
            args._handle_cp_kv_layer_split()

    def test_rejects_non_prefill_server(self):
        args = _make_layer_split_args(disaggregation_mode="null")

        with self.assertRaisesRegex(ValueError, "prefill-only servers"):
            args._handle_cp_kv_layer_split()

    def test_rejects_attn_cp_size_one(self):
        args = _make_layer_split_args(attn_cp_size=1)

        with self.assertRaisesRegex(ValueError, "attn-cp-size > 1"):
            args._handle_cp_kv_layer_split()

    def test_rejects_non_round_robin_cp_mode(self):
        args = _make_layer_split_args(dsa_prefill_cp_mode="in-seq-split")

        with self.assertRaisesRegex(ValueError, "round-robin-split"):
            args._handle_cp_kv_layer_split()

    def test_rejects_unsupported_disaggregation_transfer_backend(self):
        args = _make_layer_split_args()
        args.disaggregation_transfer_backend = "mori"

        with patch("sglang.srt.server_args.is_hip", return_value=False):
            with self.assertRaisesRegex(ValueError, "mooncake or nixl"):
                args._handle_cp_kv_layer_split()

    def test_rejects_unified_kv_backend(self):
        args = _make_layer_split_args()

        with patch("sglang.srt.server_args.is_hip", return_value=False), patch(
            "sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "unified_kv_triton"):
                args._handle_cp_kv_layer_split()

    def test_rejects_compressor_v1(self):
        args = _make_layer_split_args()

        with envs.SGLANG_OPT_USE_COMPRESSOR_V2.override(False), patch(
            "sglang.srt.server_args.is_hip", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "Compressor V2"):
                args._handle_cp_kv_layer_split()


if __name__ == "__main__":
    unittest.main()
