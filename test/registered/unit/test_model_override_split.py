"""The per-model override modules register in a load-bearing order.

``arg_groups/model_overrides/__init__.py`` is a list of imports, and importing
a module is what registers its declarations. The gate applies every matching
declaration in registration order, last writer winning, so the order of that
list decides which declaration wins for any architecture claimed by more than
one family module.

That makes the list a behavioural statement dressed as an import block, and
tools treat import blocks as free to reorder -- isort alphabetised it once
during this split and flipped ``InternS2MobiusForConditionalGeneration``'s two
declarations. The file carries ``# isort: skip_file``; this pins the outcome.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.arg_groups.model_override_base import _MODEL_OVERRIDE_FNS
from sglang.test.test_utils import CustomTestCase

# Every architecture whose declarations come from more than one family module.
# Order is the order overrides.py had before the split.
CONTESTED = {
    "InternS2MobiusForConditionalGeneration": [
        "_qwen3_5_hybrid_overrides",
        "_interns2_mobius_baseline_overrides",
    ],
    "KimiK3ForConditionalGeneration": [
        "_kimi_k3_overrides",
        "_kimi_k3_moe_runner_overrides",
    ],
    "Qwen3NextForCausalLM": [
        "_qwen3_5_hybrid_overrides",
        "_qwen3_moe_family_overrides",
    ],
    "Qwen3_5MoeForConditionalGeneration": [
        "_qwen3_5_hybrid_overrides",
        "_qwen3_moe_family_overrides",
    ],
    "Qwen3_5ForConditionalGeneration": [
        "_qwen3_5_hybrid_overrides",
        "_qwen3_moe_family_overrides",
    ],
    "InternS2PreviewForConditionalGeneration": [
        "_qwen3_5_hybrid_overrides",
        "_qwen3_moe_family_overrides",
    ],
}


class TestModelOverrideSplit(CustomTestCase):
    def test_contested_architectures_keep_their_order(self):
        for architecture, expected in CONTESTED.items():
            with self.subTest(architecture=architecture):
                got = [fn.__name__ for fn in _MODEL_OVERRIDE_FNS[architecture]]
                self.assertEqual(expected, got)

    def test_the_list_names_every_architecture_with_two_claimants(self):
        """So a newly contested architecture has to be added here, rather than
        acquiring an order nothing is watching."""
        contested = {arch for arch, fns in _MODEL_OVERRIDE_FNS.items() if len(fns) > 1}
        self.assertEqual(set(CONTESTED), contested)

    def test_every_declaration_comes_from_its_own_family_module(self):
        """The split is what this test is about: nothing is left behind in
        overrides.py."""
        for architecture, fns in _MODEL_OVERRIDE_FNS.items():
            for fn in fns:
                with self.subTest(architecture=architecture, fn=fn.__name__):
                    self.assertTrue(
                        fn.__module__.startswith(
                            "sglang.srt.arg_groups.model_overrides."
                        ),
                        f"{fn.__name__} still lives in {fn.__module__}",
                    )


if __name__ == "__main__":
    unittest.main()
