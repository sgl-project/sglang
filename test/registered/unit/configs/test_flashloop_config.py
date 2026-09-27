"""CPU-only guards for opt-in recurrence configuration and independent switches."""

import itertools
import unittest
from types import SimpleNamespace

from sglang.srt.configs.flashloop import (
    adapt_config,
    component_options,
    validate_runtime_options,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def checkpoint():
    return dict(
        model_type="ouro",
        total_ut_steps=4,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_size=2048,
    )


class TestFlashLoopConfig(CustomTestCase):
    def test_logical_layers_do_not_duplicate_weights(self):
        original = checkpoint()
        adapted = adapt_config(original)
        self.assertEqual(original["num_hidden_layers"], 24)
        self.assertEqual(adapted["flashloop_physical_layers"], 24)
        self.assertEqual(adapted["num_hidden_layers"], 96)
        self.assertEqual(adapt_config(adapted)["num_hidden_layers"], 96)

    def test_independent_switches_after_json_override(self):
        for prefill, decode, quant in itertools.product((False, True), repeat=3):
            with self.subTest(prefill=prefill, decode=decode, quant=quant):
                cfg = SimpleNamespace(**adapt_config(checkpoint(), 0.1, (0.25, 0.1)))
                cfg.flashloop_components = dict(
                    token_sparse_prefill=prefill,
                    sparse_decode=decode,
                    kv_residual_quantization=quant,
                )
                self.assertEqual(
                    component_options(cfg),
                    (
                        (0.25, 0.1) if prefill else (1.0, 1.0),
                        0.1 if decode else 1.0,
                        quant,
                    ),
                )

    def test_parser_defaults_dense(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from sglang.srt.configs.model_config_parser_registry import (
            get_model_config_parser,
        )

        with TemporaryDirectory() as folder:
            Path(folder, "config.json").write_text(json.dumps(checkpoint()))
            cfg = get_model_config_parser("flashloop").parse(
                folder, trust_remote_code=False
            )
        self.assertEqual(component_options(cfg), ((1.0, 1.0), 1.0, False))

    def test_invalid_component_configuration(self):
        cfg = SimpleNamespace(**adapt_config(checkpoint()))
        for components in ({"sparse_decode": "false"}, {"unknown": True}):
            cfg.flashloop_components = components
            with self.assertRaises(ValueError):
                component_options(cfg)
        for fraction in (0.0, -1.0, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                adapt_config(checkpoint(), fraction)

    def test_reject_unsupported_execution(self):
        for options in (
            dict(tp_size=2),
            dict(kv_cache_dtype="fp8_e4m3"),
            dict(chunked_prefill_size=1024),
            dict(disable_radix_cache=False),
            dict(enable_unified_memory=True),
            dict(enable_hierarchical_cache=True),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_runtime_options(options)


if __name__ == "__main__":
    unittest.main()
