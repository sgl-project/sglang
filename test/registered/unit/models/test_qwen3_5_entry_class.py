"""Regression test for Qwen3.5 text-only entry-class registration.

A text-only SFT of Qwen3.5 (e.g. trained via ``AutoModelForCausalLM``)
writes ``architectures: ["Qwen3_5ForCausalLM"]`` to ``config.json``. The
``Qwen3_5ForCausalLM`` / ``Qwen3_5MoeForCausalLM`` classes existed but were
missing from the module's ``EntryClass`` list, so SGLang could not resolve
the architecture and refused to serve the checkpoint. See issue #27872.
"""

from sglang.srt.models.registry import ModelRegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestQwen35EntryClass(CustomTestCase):
    def test_text_only_for_causal_lm_archs_registered(self):
        archs = ModelRegistry.get_supported_archs()
        self.assertIn("Qwen3_5ForCausalLM", archs)
        self.assertIn("Qwen3_5MoeForCausalLM", archs)

    def test_for_causal_lm_archs_resolve_to_classes(self):
        from sglang.srt.models.qwen3_5 import (
            Qwen3_5ForCausalLM,
            Qwen3_5MoeForCausalLM,
        )

        dense_cls, _ = ModelRegistry.resolve_model_cls("Qwen3_5ForCausalLM")
        moe_cls, _ = ModelRegistry.resolve_model_cls("Qwen3_5MoeForCausalLM")
        self.assertIs(dense_cls, Qwen3_5ForCausalLM)
        self.assertIs(moe_cls, Qwen3_5MoeForCausalLM)

        # The resolved classes must expose the serving contract (a runnable
        # forward plus weight loading), not just be name-resolvable.
        for cls in (dense_cls, moe_cls):
            self.assertTrue(callable(getattr(cls, "forward", None)))
            self.assertTrue(callable(getattr(cls, "load_weights", None)))


if __name__ == "__main__":
    import unittest

    unittest.main()
