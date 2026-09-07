"""A draft entry class answers the loader's questions exactly when its target does.

The loader asks the *entry class* it instantiates for the shared-experts-fusion
decision (`install_shared_experts_fusion_decision`). A draft is its own entry
class -- `...NextN`, `...MTP`, `...DSpark`, `...Eagle3` -- so when the target
family carries auto-disable conditions and the draft's class does not expose
them, the draft resolves a *different* decision than the target it drafts for,
and its weights are laid out for the other layout. That shipped once: the DSV4
DSpark draft skipped its bundled shared-expert tensors until #33312 gave it the
gate.

`test_fusion_gate_coverage.py` walks the same registry but asks a different
question: does this entry class *touch* the decision (read the flag, name a gated
class)? That catches a class after it starts consuming the decision. It cannot
catch one that should consume it and does not, which is what the DSpark class
looked like: it built the family's *layer* classes, so the flag reader lived in
another module and its own source named no gated class.

This case asks the invariant directly instead: **presence parity between a draft
entry class and its target**. Identity is deliberately not required -- a draft
that delegates with adapted arguments (the Qwen3.5 MTP unwraps `text_config` and
substitutes the MTP quantization config) is exactly right.
"""

import re
import unittest

from sglang.srt.models.registry import ModelRegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=14, suite="base-a-test-cpu")

# The hooks the loader resolves on the entry class where a draft/target
# disagreement is a defect rather than a difference. Weight-name maps
# (`packed_modules_mapping`, `hf_to_sglang_mapper`) are deliberately absent: a
# draft's checkpoint has its own names, so those differ by design.
PARITY_HOOKS = ("shared_experts_fusion_disable_reason",)

# `...Eagle`/`...Eagle3` drafts are standalone checkpoints with their own
# architecture; the rest mirror a stage of their target.
DRAFT_SUFFIX = re.compile(r"(NextN|MTP|DSpark|DFlash|Standalone)$")


def _target_of(arch: str, archs: dict):
    """The arch a draft entry class drafts for, if it is named after it."""
    match = DRAFT_SUFFIX.search(arch)
    if not match:
        return None
    base = arch[: match.start()]
    for candidate in (base, f"{base}ForCausalLM"):
        if candidate in archs and candidate != arch:
            return candidate
    return None


class TestDraftEntryHookParity(CustomTestCase):
    def test_a_draft_resolves_a_gate_exactly_when_its_target_does(self):
        archs = {}
        for arch in sorted(ModelRegistry.get_supported_archs()):
            try:
                archs[arch] = ModelRegistry.resolve_model_cls(arch)[0]
            except Exception:
                continue  # an arch this environment cannot import

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


if __name__ == "__main__":
    unittest.main()
