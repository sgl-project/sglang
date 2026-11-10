"""Deterministic inference must not restore hybrid Mamba state from the radix cache.

``extra_buffer`` restores a released Mamba checkpoint that was produced by an
earlier, differently shaped prefill batch, so a cache hit is not bit-exact with a
cold prefill: greedy logprobs and tokens can drift even under
``--enable-deterministic-inference`` (hybrid GDN drift, issue #47213). With the
strategy left on ``auto``, deterministic inference therefore has to replay the
prefix with ``no_buffer`` instead of restoring donated state.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.arg_groups.overrides as ov
import sglang.srt.configs.linear_attn_model_registry as registry
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# Qwen3.5 text stack: hybrid GDN, i.e. a model whose extra_buffer radix cache
# restores Mamba state. The shape fields keep the config loadable on CPU CI.
_MINI_QWEN3_5_CONFIG = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "vocab_size": 512,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "torch_dtype": "bfloat16",
}


class TestDeterministicMambaRadixCacheResolution(CustomTestCase):
    """Resolution contract with the model and backend probes stubbed."""

    def _view(self, deterministic, strategy="auto", disable_radix=False):
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

    def _resolve(self, view):
        with (
            patch.object(ov, "model_config_of", lambda v: v.model_config),
            patch.object(registry, "get_linear_attn_spec", lambda hf: None),
            patch.object(ov, "supports_mamba_cache_extra_buffer", lambda v, hf: True),
        ):
            return ov._mamba_radix_cache_resolution(view)

    def test_deterministic_auto_switches_to_no_buffer(self):
        declared = self._resolve(self._view(True))
        self.assertEqual(declared["mamba_radix_cache_strategy"], "no_buffer")
        self.assertTrue(declared["disable_overlap_schedule"])

    def test_non_deterministic_auto_keeps_extra_buffer(self):
        declared = self._resolve(self._view(False))
        self.assertEqual(declared["mamba_radix_cache_strategy"], "extra_buffer")

    def test_explicit_strategy_is_not_overridden(self):
        for strategy in ("extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                declared = self._resolve(self._view(True, strategy=strategy))
                self.assertNotIn("mamba_radix_cache_strategy", declared)

    def test_explicit_extra_buffer_warns_under_deterministic(self):
        # Explicitly opting into extra_buffer keeps the user's choice but must
        # not silently violate the deterministic contract (#47213).
        for strategy in ("extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                with self.assertLogs(
                    "sglang.srt.arg_groups.overrides", level="WARNING"
                ) as logs:
                    self._resolve(self._view(True, strategy=strategy))
                self.assertTrue(
                    any(strategy in line for line in logs.output), logs.output
                )

    def test_disable_radix_cache_takes_precedence(self):
        self.assertEqual(self._resolve(self._view(True, disable_radix=True)), {})


class TestDeterministicMambaRadixCacheEndToEnd(CustomTestCase):
    """Same contract through the real ModelConfig and extra_buffer support probe."""

    def _server_args(self, **kwargs):
        from sglang.srt.server_args import ServerArgs

        config_dir = tempfile.mkdtemp(prefix="det_mamba_cache_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        config = dict(
            _MINI_QWEN3_5_CONFIG,
            architectures=["Qwen3_5ForCausalLM"],
            model_type="qwen3_5_text",
        )
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(config, f)
        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("linear_attn_backend", "triton")
        return ServerArgs(model_path=config_dir, **kwargs)

    def _resolve(self, **kwargs):
        return ov._mamba_radix_cache_resolution(
            resolving_view(self._server_args(**kwargs))
        )

    def test_deterministic_auto_replays_the_prefix(self):
        declared = self._resolve(enable_deterministic_inference=True)
        self.assertEqual(declared["mamba_radix_cache_strategy"], "no_buffer")
        self.assertTrue(declared["disable_overlap_schedule"])

    def test_non_deterministic_auto_keeps_state_reuse(self):
        declared = self._resolve(enable_deterministic_inference=False)
        self.assertEqual(declared["mamba_radix_cache_strategy"], "extra_buffer")
        self.assertNotIn("disable_overlap_schedule", declared)

    def test_explicit_extra_buffer_under_deterministic_warns_and_is_kept(self):
        with self.assertLogs("sglang.srt.arg_groups.overrides", level="WARNING"):
            declared = self._resolve(
                enable_deterministic_inference=True,
                mamba_radix_cache_strategy="extra_buffer",
            )
        self.assertNotIn("mamba_radix_cache_strategy", declared)


if __name__ == "__main__":
    unittest.main()
