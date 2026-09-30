"""V4.1 PD feature gate: static P, dynamic D, existing topology restrictions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import deepseek_v4_hook as mod
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestV41PdDynamicGate(CustomTestCase):
    def validate(self, role, mode, **overrides):
        cfg = SimpleNamespace(
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=False,
            speculative_algorithm="DSPARK",
            enable_hisparse=False,
            dsv4_attn_backend="flashmla",
            enable_two_batch_overlap=False,
            pp_size=1,
            disaggregation_mode=role,
            disaggregation_transfer_backend="mooncake",
            dp_size=1,
            enable_dp_attention=False,
            attn_cp_size=1,
            dcp_size=1,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled")
            ),
        )
        cfg.__dict__.update(overrides)
        with (
            patch.object(mod, "resolving_view", return_value=cfg),
            patch.object(
                mod,
                "model_config_of",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(model_type="deepseek_v41")
                ),
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
            patch(
                "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
                return_value=mode,
            ),
        ):
            mod.validate_deepseek_v41_features(object())

    def test_decode_all_modes(self):
        for mode in RaggedVerifyMode:
            self.validate("decode", mode)

    def test_prefill_remains_static(self):
        self.validate("prefill", RaggedVerifyMode.STATIC)
        for mode in (RaggedVerifyMode.COMPACT, RaggedVerifyMode.CAP_ACCEPT):
            with self.assertRaisesRegex(ValueError, "static verify on prefill"):
                self.validate("prefill", mode)

    def test_unsupported_topology_remains_rejected(self):
        for change in (
            {"dp_size": 2},
            {"attn_cp_size": 2},
            {"dcp_size": 2},
            {"enable_dp_attention": True},
            {"disaggregation_transfer_backend": "nixl"},
        ):
            with self.assertRaises(ValueError):
                self.validate("decode", RaggedVerifyMode.COMPACT, **change)


if __name__ == "__main__":
    unittest.main()
