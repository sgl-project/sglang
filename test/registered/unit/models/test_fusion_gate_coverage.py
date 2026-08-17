"""Every loader entry class that can reach a fusion-gated family answers for it.

The loader installs the shared-experts-fusion decision for the class it
instantiates (`install_shared_experts_fusion_decision`). A model whose layers
read `is_shared_experts_fusion_disabled()` therefore gets whatever answer that
*entry* class produced — and an entry class with no
`shared_experts_fusion_disable_reason` falls back to the user's intent, silently
skipping the family's auto-disable conditions.

That is easy to miss for a wrapper: `KimiVLForConditionalGeneration` is the
registered arch, but a DeepSeek body is built inside it, and the DeepSeek
conditions used to be evaluated during that nested construction. This case walks
the registry so a new wrapper (or a new MTP/nextn entry) cannot reintroduce the
gap.
"""

import ast
import importlib
import inspect
import os
import sys
import unittest

from sglang.srt.models.registry import ModelRegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=28, suite="base-a-test-cpu")

GATE = "shared_experts_fusion_disable_reason"
FLAG_READERS = (
    "is_shared_experts_fusion_disabled",
    "determine_num_fused_shared_experts",
)

# Archs that read the fusion flag but deliberately have no gate: nothing in
# their lineage carries auto-disable conditions, so they follow the user's
# intent — the behavior they had before the decision moved to the loader.
GATELESS_BY_DESIGN = {
    # The in-tree class has no ``determine_num_fused_shared_experts`` at all
    # (the call is guarded by ``hasattr`` for a downstream variant).
    "BailingMoeForCausalLMNextN",
    # Its target family (Glm4v) is dense; there is no gate to inherit.
    "GlmOcrForConditionalGenerationNextN",
    # The vision tower registered on its own: it shares a module with
    # PixtralForConditionalGeneration (which does answer) but builds no language
    # model, so there is nothing for a gate to decide.
    "PixtralVisionModel",
}


def _gated_classes(source: str) -> set:
    """Classes in this module that define or receive a fusion gate."""
    names = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for body in node.body:
                if (
                    isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and body.name == GATE
                ):
                    names.add(node.name)
                if isinstance(body, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == GATE for t in body.targets
                ):
                    names.add(node.name)
        # ``for cls in (A, B): cls.<GATE> = ...``
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, (ast.Tuple, ast.List))
            and GATE in ast.dump(node)
        ):
            names |= {e.id for e in node.iter.elts if isinstance(e, ast.Name)}
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == GATE
                    and isinstance(target.value, ast.Name)
                ):
                    names.add(target.value.id)
    return names


def gated_class_names() -> set:
    """Every model class that *resolves* a gate, inherited ones included.

    A subclass like `DeepseekV3ForCausalLM` inherits the gate without naming it,
    so collecting names from class bodies alone would let a wrapper that builds
    the subclass slip through.
    """
    names = set()
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("sglang.srt.models.") or module is None:
            continue
        for member in vars(module).values():
            # transformers re-exports lazy placeholders that raise on any
            # attribute access when their optional backend is missing.
            try:
                if (
                    inspect.isclass(member)
                    and (member.__module__ or "").startswith("sglang.srt.models.")
                    and hasattr(member, GATE)
                ):
                    names.add(member.__name__)
            except Exception:
                continue
    return names


class TestFusionGateCoverage(CustomTestCase):
    def test_every_entry_class_reaching_a_gated_family_has_a_gate(self):
        models_dir = list(importlib.import_module("sglang.srt.models").__path__)[0]
        gates_by_module = {}
        for name in sorted(os.listdir(models_dir)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(models_dir, name), encoding="utf-8") as f:
                try:
                    gates_by_module[f"sglang.srt.models.{name[:-3]}"] = _gated_classes(
                        f.read()
                    )
                except SyntaxError:
                    continue

        missing = []
        all_gated = None
        for arch in sorted(ModelRegistry.get_supported_archs()):
            try:
                model_class, _ = ModelRegistry.resolve_model_cls(arch)
            except Exception:
                continue
            if hasattr(model_class, GATE) or arch in GATELESS_BY_DESIGN:
                continue
            module = importlib.import_module(model_class.__module__)
            try:
                source = inspect.getsource(module)
            except OSError:
                continue
            reasons = []
            if any(reader in source for reader in FLAG_READERS):
                reasons.append("reads the fusion flag")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                tree = None
            if tree is not None:
                # Any *use* of a gated class counts, whatever the shape: a
                # direct call (`DeepseekV2ForCausalLM(...)`), a module attribute
                # (`qwen3_5.Qwen3_5MoeForCausalLM`), or a class attribute the
                # constructor later calls (`body_cls = qwen3_5.Qwen3_5...`).
                # Only matching calls would miss the last two.
                if all_gated is None:
                    all_gated = gated_class_names()
                used = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id in all_gated:
                        used.add(node.id)
                    elif isinstance(node, ast.Attribute) and node.attr in all_gated:
                        used.add(node.attr)
                for name in sorted(used):
                    if name != model_class.__name__:
                        reasons.append(f"references {name}")
            if reasons:
                missing.append(
                    f"{arch} ({model_class.__module__}): {', '.join(reasons)}"
                )

        self.assertEqual(
            [],
            missing,
            "these entry classes reach a fusion-gated family but resolve no "
            f"{GATE}, so the loader falls back to the user's intent for them and "
            "the family's auto-disable conditions never run:\n  "
            + "\n  ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
