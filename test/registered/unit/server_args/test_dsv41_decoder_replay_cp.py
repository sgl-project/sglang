"""Decoder tail replay accepts CP's single-DP-rank attention plumbing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.deepseek_v4_hook import (
    validate_deepseek_v4_cp,
    validate_deepseek_v41_features,
)
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecoderReplayCP(unittest.TestCase):
    def args(self, **kwargs):
        args = ServerArgs(
            model_path="dummy", enable_decoder_swa_bounded_replay=True, **kwargs
        )
        # The graph-resolution hook precedes model feature validation in the server.
        args.cuda_graph_config = SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED)
        )
        return args

    def setUp(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        )
        patcher = patch(
            "sglang.srt.arg_groups.deepseek_v4_hook.model_config_of",
            return_value=model_config,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cp_resolution_can_enable_decoder_replay(self):
        args = self.args(enable_prefill_cp=True, cp_strategy="interleave", tp_size=8)
        validate_deepseek_v4_cp(args)
        self.assertTrue(resolving_view(args).enable_dp_attention)
        self.assertEqual(resolving_view(args).attn_cp_size, 8)
        validate_deepseek_v41_features(args)

    def test_multiple_dp_ranks_remain_rejected(self):
        args = self.args(enable_dp_attention=True, dp_size=2, tp_size=8)
        with self.assertRaisesRegex(
            ValueError, "decoder-swa-bounded-replay.*DP attention"
        ):
            validate_deepseek_v41_features(args)


if __name__ == "__main__":
    unittest.main()
