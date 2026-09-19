"""V4.1 language-model-only PD configuration validation."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.arg_groups import model_hook
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV41TextOnlyPDPolicy(CustomTestCase):
    def validate(
        self,
        model_type="deepseek_v41",
        mode="prefill",
        arch="DeepseekV4ForCausalLM",
        **flags,
    ):
        cfg = NS(
            language_model_only=True,
            encoder_only=False,
            language_only=False,
            enable_prefix_mm_cache=False,
            enable_broadcast_mm_inputs_process=False,
            mm_enable_dp_encoder=False,
            disaggregation_mode=mode,
        )
        for name, value in flags.items():
            setattr(cfg, name, value)
        model = NS(hf_config=NS(model_type=model_type, architectures=[arch]))
        args = NS(
            LANGUAGE_MODEL_ONLY_ARCHITECTURES=ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES
        )
        with (
            patch.object(model_hook, "resolving_view", return_value=cfg),
            patch.object(model_hook, "model_config_of", return_value=model),
        ):
            model_hook.handle_language_model_only(args)

    def test_v41_modes(self):
        for mode in ("null", "prefill", "decode"):
            with self.subTest(mode=mode):
                self.validate(mode=mode)

    def test_other_models_still_reject_pd(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.validate(model_type="cosmos3", arch="Cosmos3ForConditionalGeneration")

    def test_encoder_options_still_rejected(self):
        for flag in (
            "encoder_only",
            "language_only",
            "enable_prefix_mm_cache",
            "enable_broadcast_mm_inputs_process",
            "mm_enable_dp_encoder",
        ):
            with (
                self.subTest(flag=flag),
                self.assertRaisesRegex(ValueError, "cannot be combined"),
            ):
                self.validate(**{flag: True})

    def test_unknown_arch_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            self.validate(arch="UnknownArchitecture")


if __name__ == "__main__":
    unittest.main()
