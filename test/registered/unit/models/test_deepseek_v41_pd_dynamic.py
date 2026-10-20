"""PD verify modes must not regress upstream transport / parallelism gates."""

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
            ep_join_mode=None,
            tp_size=8,
            disaggregation_mode=role,
            disaggregation_transfer_backend="mooncake",
            dp_size=1,
            attn_dp_size=1,
            enable_dp_attention=False,
            enable_prefill_cp=False,
            cp_strategy="interleave",
            attn_cp_size=1,
            dcp_size=1,
            moe_a2a_backend="none",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=mod.Backend.DISABLED)
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
            patch.object(
                mod,
                "get_platform",
                return_value=SimpleNamespace(is_cuda=True, is_hip=False),
            ),
            patch(
                "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
                return_value=False,
            ),
            patch(
                "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
                return_value=mode,
            ),
        ):
            mod.validate_deepseek_v41_features(object())

    def test_decode_modes_keep_both_transports_and_attention_dp(self):
        for mode in RaggedVerifyMode:
            for backend in ("mooncake", "nixl"):
                for dp in (1, 8):
                    with self.subTest(mode=mode, backend=backend, dp=dp):
                        self.validate(
                            "decode",
                            mode,
                            disaggregation_transfer_backend=backend,
                            dp_size=dp,
                            attn_dp_size=dp,
                            enable_dp_attention=dp > 1,
                        )

    def test_prefill_stays_static_and_preserves_interleave_cp(self):
        for backend in ("mooncake", "nixl"):
            for cp in (1, 8):
                config = dict(
                    disaggregation_transfer_backend=backend,
                    enable_prefill_cp=cp > 1,
                    attn_cp_size=cp,
                )
                self.validate("prefill", RaggedVerifyMode.STATIC, **config)
                for mode in (RaggedVerifyMode.COMPACT, RaggedVerifyMode.CAP_ACCEPT):
                    with self.assertRaisesRegex(ValueError, "static verify on prefill"):
                        self.validate("prefill", mode, **config)

    def test_unsupported_transport_and_decode_cp_still_fail(self):
        for mode in RaggedVerifyMode:
            for change in (
                {"attn_cp_size": 2},
                {"dcp_size": 2},
                {"enable_prefill_cp": True},
                {"disaggregation_transfer_backend": "unsupported"},
            ):
                with self.subTest(mode=mode, change=change):
                    with self.assertRaises(ValueError):
                        self.validate("decode", mode, **change)

    def test_prefill_cp_other_than_interleave_still_fails(self):
        with self.assertRaises(ValueError):
            self.validate(
                "prefill",
                RaggedVerifyMode.STATIC,
                enable_prefill_cp=True,
                attn_cp_size=8,
                cp_strategy="round-robin",
            )


if __name__ == "__main__":
    unittest.main()
