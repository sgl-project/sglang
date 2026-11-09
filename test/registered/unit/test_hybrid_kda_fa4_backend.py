"""Unit tests for the KDA-hybrid fa4 constraint in
srt/arg_groups/overrides.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.model_override_base import ResolvedView
from sglang.srt.arg_groups.overrides import (
    POST_PROCESS_PASSES,
    _hybrid_kda_fa4_backend_constraint,
)
from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
from sglang.srt.configs.glm5_next import Glm5NextConfig
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.runtime_context import override_platform
from sglang.test.test_utils import CustomTestCase

# The three config shapes `hybrid_kda_config` matches, each with the architecture
# name the refusal quotes.
KDA_CONFIGS = (
    (
        KimiLinearConfig(architectures=["KimiLinearForCausalLM"]),
        "KimiLinearForCausalLM",
    ),
    (
        BailingHybridConfig(use_kda=True, architectures=["BailingMoeV3ForCausalLM"]),
        "BailingMoeV3ForCausalLM",
    ),
    (
        Glm5NextConfig(
            text_config={"gate_lower_bound": -5},
            architectures=["Glm5NextForConditionalGeneration"],
        ),
        "Glm5NextForConditionalGeneration",
    ),
)


def _view(hf_config, **overrides):
    """A fixture-supplied `_model_config` is what `model_config_of` hands back,
    so this reads no checkpoint. The three backend fields are all the pass
    looks at besides the configuration.
    """
    fields = dict(
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        _model_config=SimpleNamespace(hf_config=hf_config, is_draft_model=False),
    )
    fields.update(overrides)
    return ResolvedView(SimpleNamespace(**fields))


def _sm120():
    return override_platform(is_sm100=False, is_sm110=False, is_sm120=True)


def _run(view):
    """Return the pass's declaration; it either refuses or declares nothing."""
    return _hybrid_kda_fa4_backend_constraint(view)


class TestHybridKdaFa4BackendConstraint(CustomTestCase):
    def test_sm120_refuses_fa4_for_every_kda_shape(self):
        # Red if the family predicate drops an arm: KimiLinear, Bailing with
        # use_kda and GLM-5-Next are the three `hybrid_kda_config` accepts.
        for hf_config, arch in KDA_CONFIGS:
            with self.subTest(arch=arch):
                with _sm120():
                    with self.assertRaises(ValueError) as ctx:
                        _run(_view(hf_config, attention_backend="fa4"))
                    self.assertIn(arch, str(ctx.exception))

    def test_refusal_names_the_side_that_resolved_to_fa4(self):
        # One report row ran `--prefill-attention-backend fa4` with triton
        # behind it, so a refusal that only looked at `attention_backend` would
        # let that through, and the message has to say which side it blamed.
        cases = (
            ("prefill", "decode", "fa4", "triton"),
            ("decode", "prefill", "fa4", "triton"),
        )
        with _sm120():
            for blamed, other, blamed_value, other_value in cases:
                with self.subTest(blamed=blamed):
                    with self.assertRaises(ValueError) as ctx:
                        _run(
                            _view(
                                KDA_CONFIGS[0][0],
                                attention_backend="triton",
                                **{f"{blamed}_attention_backend": blamed_value},
                            )
                        )
                    message = str(ctx.exception)
                    self.assertIn(f"{blamed}={blamed_value}", message)
                    self.assertIn(f"{other}={other_value}", message)

    def test_refusal_points_at_the_backend_the_report_served_with(self):
        hf_config, _ = KDA_CONFIGS[0]
        with _sm120():
            with self.assertRaises(ValueError) as ctx:
                _run(_view(hf_config, attention_backend="fa4"))
        message = str(ctx.exception)
        self.assertIn("triton", message)
        # The names the hybrid-GDN whitelist allows here are not a promise for
        # KDA hybrids, so the refusal must not read as recommending them.
        self.assertIn("were not validated for KDA hybrids", message)

    def test_other_hybrid_families_keep_their_answers(self):
        # use_kda=False is the lightning arm, and hybrid GDN already has its own
        # Blackwell whitelist in attention_registry.py: neither is this pass's
        # business, and refusing them would be a behavioural change the report
        # does not cover.
        cases = {
            "bailing_lightning": BailingHybridConfig(
                use_kda=False, architectures=["MiniCPMForCausalLM"]
            ),
            "hybrid_gdn": Qwen3NextConfig(
                full_attention_interval=4, architectures=["Qwen3NextForCausalLM"]
            ),
        }
        with _sm120():
            for name, hf_config in cases.items():
                with self.subTest(name=name):
                    self.assertEqual(
                        _run(_view(hf_config, attention_backend="fa4")), {}
                    )

    def test_non_fa4_backends_are_not_refused(self):
        # triton is the remedy the message gives, and the other names have their
        # own checks elsewhere: this pass must not be what stops them.
        hf_config, _ = KDA_CONFIGS[0]
        with _sm120():
            for backend in ("triton", "flashinfer", "trtllm_mha", "fa3", None):
                with self.subTest(backend=backend):
                    self.assertEqual(
                        _run(_view(hf_config, attention_backend=backend)), {}
                    )

    def test_other_archs_are_untouched(self):
        # fa4 under a KDA hybrid is the SM100 default path today
        # (HYBRID_GDN_SM100_BACKENDS and the SM100 fa4 defaults), so widening
        # this refusal past compute capability 12 would break working launches.
        hf_config, _ = KDA_CONFIGS[0]
        for facts in (
            dict(is_sm100=True, is_sm110=False, is_sm120=False),
            dict(is_sm100=False, is_sm110=True, is_sm120=False),
            dict(is_sm100=False, is_sm110=False, is_sm120=False, is_hip=True),
        ):
            with self.subTest(**facts):
                with override_platform(**facts):
                    self.assertEqual(
                        _run(_view(hf_config, attention_backend="fa4")), {}
                    )

    def test_message_builds_without_an_architecture_name(self):
        # `architectures` is optional in a config.json, and the codebase reads it
        # defensively elsewhere (`(config.architectures or [None])[0]`).
        with _sm120():
            with self.assertRaises(ValueError) as ctx:
                _run(_view(KimiLinearConfig(), attention_backend="fa4"))
        self.assertIn("this model", str(ctx.exception))

    def test_pass_is_registered(self):
        # The decorator is the only wiring: dropping it leaves every assertion
        # above green while the server keeps dying at graph capture.
        self.assertIn(_hybrid_kda_fa4_backend_constraint, POST_PROCESS_PASSES)


if __name__ == "__main__":
    unittest.main(verbosity=3)
