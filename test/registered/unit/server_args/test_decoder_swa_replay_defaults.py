"""Automatic decoder replay must preserve explicit choices and unsupported setups."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_GATE = (
    "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton"
)


def make_args(backend=Backend.DISABLED, model_type="deepseek_v41", **kwargs):
    args = ServerArgs(
        model_path="local-dsv41",
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=backend, max_seq_len=16384)
        ),
        **kwargs,
    )
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model_type)
    )
    return args


class TestDecoderSWAReplayDefaults(CustomTestCase):
    def setUp(self):
        self.enterContext(patch(_GATE, return_value=False))
        self.enterContext(override_platform(is_cuda=True, is_hip=False))

    def resolve(self, **kwargs):
        args = make_args(**kwargs)
        validate_deepseek_v41_features(args)
        return resolving_view(args).enable_decoder_swa_bounded_replay

    def test_cuda_defaults_and_opt_out(self):
        for backend in (Backend.DISABLED, Backend.BREAKABLE):
            with self.subTest(backend=backend):
                self.assertTrue(self.resolve(backend=backend))
                self.assertFalse(
                    self.resolve(backend=backend, disable_decoder_bounded_replay=True)
                )
                self.assertTrue(
                    self.resolve(
                        backend=backend, enable_decoder_swa_bounded_replay=True
                    )
                )

    def test_full_prefill_graph_stays_disabled(self):
        self.assertFalse(self.resolve(backend=Backend.FULL))
        with self.assertRaisesRegex(ValueError, "prefill CUDA graph"):
            self.resolve(backend=Backend.FULL, enable_decoder_swa_bounded_replay=True)

    def test_supported_dp_and_incompatible_dp(self):
        self.assertTrue(
            self.resolve(tp_size=4, attn_dp_size=4, moe_a2a_backend="megamoe")
        )
        self.assertFalse(self.resolve(tp_size=4, attn_dp_size=2))
        with self.assertRaisesRegex(ValueError, "DP attention"):
            self.resolve(
                tp_size=4, attn_dp_size=2, enable_decoder_swa_bounded_replay=True
            )

    def test_other_platforms_and_model_remain_opt_in(self):
        with override_platform(is_cuda=False, is_hip=True):
            self.assertFalse(self.resolve())
            self.assertTrue(self.resolve(enable_decoder_swa_bounded_replay=True))
        self.assertFalse(self.resolve(model_type="deepseek_v4"))

    def test_disaggregation_and_cp_do_not_auto_enable(self):
        for kwargs in (
            dict(disaggregation_mode="prefill"),
            dict(disaggregation_mode="decode"),
            dict(enable_prefill_cp=True),
            dict(attn_cp_size=2),
            dict(dcp_size=2),
        ):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(self.resolve(**kwargs))

    def test_conflicting_flags_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.resolve(
                enable_decoder_swa_bounded_replay=True,
                disable_decoder_bounded_replay=True,
            )

    def test_cli_auto_enable_disable(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for flags, enabled, disabled in (
            ([], False, False),
            (["--enable-decoder-swa-bounded-replay"], True, False),
            (["--disable-decoder-bounded-replay"], False, True),
        ):
            with self.subTest(flags=flags):
                raw = parser.parse_args(["--model-path", "local-dsv41", *flags])
                args = ServerArgs.from_cli_args(raw)
                self.assertIs(args.enable_decoder_swa_bounded_replay, enabled)
                self.assertIs(args.disable_decoder_bounded_replay, disabled)


if __name__ == "__main__":
    unittest.main()
