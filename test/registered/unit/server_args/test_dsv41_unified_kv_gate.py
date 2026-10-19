"""DeepSeek-V4.1 on the fp8 unified KV layout admits DSpark, and only in the shapes its
draft block window and verify streams are wired for."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ENV_GATE = "sglang.srt.layers.attention.dsv4.unified_kv_gate"


class _PastFeatureChecks(Exception):
    """Validation read a field past the V4.1 feature checks."""


class _View(SimpleNamespace):
    def __getattr__(self, name):
        raise _PastFeatureChecks(name)


class TestUnifiedKvDSparkGate(CustomTestCase):
    def _validate(
        self, *, fp8=True, ragged=RaggedVerifyMode.STATIC, disaggregation_mode="null"
    ):
        view = _View(
            dsv4_attn_backend="dsv4",
            enable_encoder_swa_bounded_replay=False,
            speculative_algorithm="DSPARK",
            enable_hisparse=False,
            enable_hierarchical_cache=False,
            disaggregation_mode=disaggregation_mode,
            enable_two_batch_overlap=False,
            pp_size=1,
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        )
        with (
            patch.object(deepseek_v4_hook, "resolving_view", return_value=view),
            patch.object(
                deepseek_v4_hook, "model_config_of", return_value=model_config
            ),
            patch(f"{_ENV_GATE}.is_unified_kv_triton", return_value=True),
            patch(f"{_ENV_GATE}.is_unified_kv_fp8", return_value=fp8),
            patch(
                "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
                return_value=ragged,
            ),
        ):
            deepseek_v4_hook.validate_deepseek_v41_features(object())

    def test_dspark_passes_on_the_fp8_layout(self):
        with self.assertRaises(_PastFeatureChecks):
            self._validate()

    def test_rejects_what_the_draft_window_is_not_wired_for(self):
        for kwargs, message in (
            (dict(ragged=RaggedVerifyMode.COMPACT), "compact ragged verify"),
            (dict(disaggregation_mode="prefill"), "DSpark under PD"),
            (dict(fp8=False), "bf16 unified KV layout"),
        ):
            with (
                self.subTest(**kwargs),
                self.assertRaisesRegex(ValueError, message),
            ):
                self._validate(**kwargs)


if __name__ == "__main__":
    unittest.main()
