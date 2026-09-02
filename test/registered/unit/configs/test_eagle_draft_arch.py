"""Unit tests for EAGLE draft architecture resolution."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    EAGLE_DRAFT_BASE_ARCHS,
    resolve_eagle_draft_arch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cfg(**kwargs):
    return SimpleNamespace(**kwargs)


class TestResolveEagleDraftArch(CustomTestCase):
    def test_llama_draft_without_draft_vocab_is_eagle(self):
        # yuhuili/EAGLE-LLaMA3.1-Instruct-8B, linborui/EAGLE-Llama-3.2-3B-Instruct
        self.assertEqual(
            resolve_eagle_draft_arch("LlamaForCausalLM", _cfg()),
            "LlamaForCausalLMEagle",
        )

    def test_llama_draft_with_draft_vocab_is_eagle3(self):
        # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B reports draft_vocab_size=32000
        self.assertEqual(
            resolve_eagle_draft_arch("LlamaForCausalLM", _cfg(draft_vocab_size=32000)),
            "LlamaForCausalLMEagle3",
        )

    def test_layer_count_is_not_used_to_classify(self):
        # yuhuili/EAGLE3-Vicuna1.3-13B reports num_hidden_layers=40, so the
        # draft is not identifiable by having a single decoder layer.
        self.assertEqual(
            resolve_eagle_draft_arch(
                "LlamaForCausalLM", _cfg(num_hidden_layers=40, draft_vocab_size=32000)
            ),
            "LlamaForCausalLMEagle3",
        )

    def test_qwen2_draft_is_eagle(self):
        # yuhuili/EAGLE-Qwen2-7B-Instruct
        self.assertEqual(
            resolve_eagle_draft_arch("Qwen2ForCausalLM", _cfg()),
            "Qwen2ForCausalLMEagle",
        )

    def test_family_without_eagle3_class_is_left_alone(self):
        # There is no Qwen2ForCausalLMEagle3 entry class to route to.
        self.assertIsNone(
            resolve_eagle_draft_arch("Qwen2ForCausalLM", _cfg(draft_vocab_size=32000))
        )

    def test_repackaged_checkpoints_are_untouched(self):
        # lmsys/sglang-EAGLE-* already name the Eagle class.
        for arch in ("LlamaForCausalLMEagle", "LlamaForCausalLMEagle3"):
            with self.subTest(arch=arch):
                self.assertIsNone(resolve_eagle_draft_arch(arch, _cfg()))

    def test_unknown_architecture_is_untouched(self):
        self.assertIsNone(resolve_eagle_draft_arch("DeepseekV3ForCausalLM", _cfg()))

    def test_explicit_none_draft_vocab_counts_as_absent(self):
        self.assertEqual(
            resolve_eagle_draft_arch("LlamaForCausalLM", _cfg(draft_vocab_size=None)),
            "LlamaForCausalLMEagle",
        )

    def test_mapping_targets_are_registered_entry_classes(self):
        """Every architecture the table maps to must be loadable."""
        from sglang.srt.models.registry import ModelRegistry

        supported = set(ModelRegistry.get_supported_archs())
        for base, (eagle_arch, eagle3_arch) in EAGLE_DRAFT_BASE_ARCHS.items():
            for target in (eagle_arch, eagle3_arch):
                if target is None:
                    continue
                with self.subTest(base=base, target=target):
                    self.assertIn(target, supported)


if __name__ == "__main__":
    unittest.main()
