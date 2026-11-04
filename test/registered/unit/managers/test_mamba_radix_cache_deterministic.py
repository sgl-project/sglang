"""Deterministic inference must not restore Mamba state from the radix cache.

``extra_buffer`` restores a released Mamba checkpoint that was computed by an
earlier, differently shaped prefill batch. The restored state is not bit-exact
with a cold prefill, so a cache hit can change greedy logprobs/tokens even under
``--enable-deterministic-inference`` (hybrid GDN drift, SGLang issue #41351).
When the strategy is left on ``auto``, deterministic inference should replay the
prefix with ``no_buffer`` instead of restoring the donated state.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.arg_groups.overrides as ov
import sglang.srt.configs.linear_attn_model_registry as registry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _view(deterministic: bool, strategy: str = "auto", disable_radix: bool = False):
    hf_config = SimpleNamespace(
        architectures=["Qwen3_5ForCausalLM"], layer_types=[]
    )
    return SimpleNamespace(
        enable_deterministic_inference=deterministic,
        mamba_radix_cache_strategy=strategy,
        disable_radix_cache=disable_radix,
        disable_overlap_schedule=False,
        page_size=1,
        linear_attn_backend="triton",
        model_config=SimpleNamespace(hf_config=hf_config),
    )


def _resolve(view):
    with (
        patch.object(ov, "model_config_of", lambda v: v.model_config),
        patch.object(registry, "get_linear_attn_spec", lambda hf: None),
        patch.object(ov, "supports_mamba_cache_extra_buffer", lambda v, hf: True),
    ):
        return ov._mamba_radix_cache_resolution(view)


class TestDeterministicMambaRadixCache(CustomTestCase):
    def test_deterministic_auto_switches_to_no_buffer(self):
        declared = _resolve(_view(True))
        self.assertEqual(declared["mamba_radix_cache_strategy"], "no_buffer")
        self.assertTrue(declared["disable_overlap_schedule"])

    def test_non_deterministic_auto_keeps_extra_buffer(self):
        declared = _resolve(_view(False))
        self.assertEqual(declared["mamba_radix_cache_strategy"], "extra_buffer")

    def test_explicit_strategy_is_not_overridden(self):
        declared = _resolve(_view(True, strategy="extra_buffer"))
        self.assertNotIn("mamba_radix_cache_strategy", declared)

    def test_explicit_extra_buffer_warns_under_deterministic(self):
        # Explicitly opting into extra_buffer keeps the user's choice but must
        # not silently violate the deterministic contract (#47213).
        with self.assertLogs(
            "sglang.srt.arg_groups.overrides", level="WARNING"
        ) as logs:
            _resolve(_view(True, strategy="extra_buffer"))
        self.assertTrue(any("extra_buffer" in line for line in logs.output))

    def test_disable_radix_cache_takes_precedence(self):
        self.assertEqual(_resolve(_view(True, disable_radix=True)), {})


if __name__ == "__main__":
    unittest.main()
