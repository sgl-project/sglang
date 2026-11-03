"""Draft and target entry classes must expose compatible fusion hooks."""

import re
import unittest

from sglang.srt.models.registry import ModelRegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# The hooks the loader resolves on the entry class where a draft/target
# disagreement is a defect rather than a difference. Weight-name maps
# (`packed_modules_mapping`, `hf_to_sglang_mapper`) are deliberately absent: a
# draft's checkpoint has its own names, so those differ by design.
PARITY_HOOKS = ("shared_experts_fusion_disable_reason", "num_fusable_shared_experts")

# Gated entry classes that declare no count, on purpose: HYV4's gate always
# disables fusion, and the Qwen3.5 family (with the models built on it) fuses a
# gated shared expert inside its own MoE block, outside the loader's count.
GATE_WITHOUT_COUNT = frozenset(
    {
        "HYV4ForCausalLM",
        "HYV4ForCausalLMNextN",
        "InternS2MobiusForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "MiniCPMV",
        "MiniCPMV4_6ForConditionalGeneration",
        "MiniCPMV4_7ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForCausalLMMTP",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen4ExpForCausalLMMTP",
        "Qwen4ExpForConditionalGeneration",
    }
)

# `...Eagle`/`...Eagle3` drafts are standalone checkpoints with their own
# architecture; the rest mirror a stage of their target.
DRAFT_SUFFIX = re.compile(r"(NextN|MTP|DSpark|DFlash|Standalone)$")


def _target_of(arch: str, archs: dict):
    """The arch a draft entry class drafts for, if it is named after it."""
    # Bailing shares its NextN entry class across generations. Its loader
    # fusion hooks belong to V3; legacy checkpoints return a fused count of 0
    # (covered by TestBailingMoeV3Gate in test_shared_experts_fusion_gates.py).
    if arch == "BailingMoeForCausalLMNextN":
        return "BailingMoeV3ForCausalLM"
    match = DRAFT_SUFFIX.search(arch)
    if not match:
        return None
    base = arch[: match.start()]
    for candidate in (base, f"{base}ForCausalLM"):
        if candidate in archs and candidate != arch:
            return candidate
    return None


def _entry_classes() -> dict:
    archs = {}
    for arch in sorted(ModelRegistry.get_supported_archs()):
        try:
            archs[arch] = ModelRegistry.resolve_model_cls(arch)[0]
        except Exception:
            continue  # an arch this environment cannot import
    return archs


class TestDraftEntryHookParity(CustomTestCase):
    def test_a_draft_resolves_a_gate_exactly_when_its_target_does(self):
        archs = _entry_classes()

        checked, offenders = 0, []
        for arch, cls in archs.items():
            target = _target_of(arch, archs)
            if target is None:
                continue
            for hook in PARITY_HOOKS:
                checked += 1
                draft_has = hasattr(cls, hook)
                target_has = hasattr(archs[target], hook)
                if draft_has == target_has:
                    continue
                which = "the draft" if target_has else "the target"
                offenders.append(
                    f"{arch} vs {target}: {hook} missing on {which} "
                    f"({cls.__module__} / {archs[target].__module__})"
                )

        self.assertGreater(checked, 10, "the draft/target pairing found nothing")
        self.assertEqual(
            [],
            offenders,
            "a draft entry class and the target it drafts for must resolve the "
            "same loader hooks; otherwise the loader installs one decision for "
            "the target and another for the draft, and the draft's weights are "
            "laid out for the wrong one:\n  " + "\n  ".join(offenders),
        )


class TestFusionHookPairing(CustomTestCase):
    def test_a_fusion_gate_comes_with_a_count(self):
        archs = _entry_classes()
        gated = {
            arch
            for arch, cls in archs.items()
            if hasattr(cls, "shared_experts_fusion_disable_reason")
        }
        missing = sorted(
            arch
            for arch in gated - GATE_WITHOUT_COUNT
            if not hasattr(archs[arch], "num_fusable_shared_experts")
        )
        self.assertEqual(
            [],
            missing,
            "these entry classes expose a fusion gate but no "
            "num_fusable_shared_experts, so the loader runs them unfused; declare "
            "the count (delegating like the gate), or list the class in "
            "GATE_WITHOUT_COUNT with the reason it never fuses routed slots",
        )
        self.assertEqual(
            [],
            sorted(GATE_WITHOUT_COUNT - gated),
            "GATE_WITHOUT_COUNT names entry classes that are gone or lost their gate",
        )


if __name__ == "__main__":
    unittest.main()
