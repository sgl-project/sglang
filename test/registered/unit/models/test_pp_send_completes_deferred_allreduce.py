"""A decoder layer may leave its FFN all-reduce to the next layer's input norm.
The last layer on a non-last pipeline rank has no next layer there, so every
model forward that sends hidden states produced by such layers must complete
the reduction first. Both sets are derived from the model sources."""

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODELS_DIR = Path(sglang.__file__).resolve().parent / "srt" / "models"
MARKER = "_sglang_needs_allreduce_fusion"
COMPLETE = "complete_deferred_allreduce"


def base_names(node):
    return [
        base.id if isinstance(base, ast.Name) else base.attr
        for base in node.bases
        if isinstance(base, (ast.Name, ast.Attribute))
    ]


def method(node, name):
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def calls(node, name):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            called = (
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            )
            if called == name:
                yield sub


def defers_in_own_body(node):
    """Asks whether to leave the reduction to the next layer (directly or
    through ffn_exit), or sets the marker itself."""
    if any(calls(node, "ffn_exit")) or any(
        calls(node, "should_fuse_mlp_allreduce_with_next_layer")
    ):
        return True
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Constant):
            for target in sub.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == MARKER
                    and sub.value.value is True
                ):
                    return True
    return False


def load_classes():
    classes = {}
    for path in sorted(MODELS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, []).append((path, tree, node))
    return classes


def deferring_layer_classes(classes):
    deferring = {
        name
        for name, defs in classes.items()
        if any(defers_in_own_body(node) for _, _, node in defs)
    }
    # A subclass defers through its base unless it replaces forward without
    # delegating to it.
    while True:
        inherited = set()
        for name, defs in classes.items():
            if name in deferring:
                continue
            for _, _, node in defs:
                forward = method(node, "forward")
                delegates = forward is None or any(
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Call)
                    and getattr(call.func.value.func, "id", None) == "super"
                    for call in calls(forward, "forward")
                )
                if delegates and set(base_names(node)) & deferring:
                    inherited.add(name)
        if not inherited:
            return deferring
        deferring |= inherited


def uses_deferring_layer(tree, deferring):
    """References a deferring layer class other than to subclass it."""
    in_bases = {
        id(base)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for base in node.bases
    }
    for node in ast.walk(tree):
        if id(node) in in_bases:
            continue
        if isinstance(node, ast.Name) and node.id in deferring:
            return True
        if isinstance(node, ast.Attribute) and node.attr in deferring:
            return True
    return False


def pp_senders(classes):
    """forward methods, of models built from deferring layers, that construct
    PPProxyTensors -- including ones inherited from a base model class."""
    deferring = deferring_layer_classes(classes)
    senders = {}
    pending = []
    for defs in classes.values():
        for path, tree, node in defs:
            if uses_deferring_layer(tree, deferring):
                pending.append((path, node))
    seen = set()
    while pending:
        path, node = pending.pop()
        key = (path, node.name)
        if key in seen:
            continue
        seen.add(key)
        forward = method(node, "forward")
        if forward is not None and any(calls(forward, "PPProxyTensors")):
            senders[f"{path.relative_to(MODELS_DIR)}:{node.name}"] = forward
        for base in base_names(node):
            for base_path, _, base_node in classes.get(base, ()):
                pending.append((base_path, base_node))
    return senders


class TestPPSendCompletesDeferredAllreduce(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.classes = load_classes()
        cls.senders = pp_senders(cls.classes)

    def test_derivation_finds_the_known_paths(self):
        # One model whose own layers defer, one that inherits its forward.
        self.assertIn("glm4_moe_lite.py:Glm4MoeLiteModel", self.senders)
        self.assertIn("qwen2_moe.py:Qwen2MoeModel", self.senders)

    def test_each_pp_send_completes_the_deferred_reduction_first(self):
        missing = []
        for name, forward in sorted(self.senders.items()):
            completes = [call.lineno for call in calls(forward, COMPLETE)]
            for send in calls(forward, "PPProxyTensors"):
                if not any(line < send.lineno for line in completes):
                    missing.append(f"{name} (line {send.lineno})")
        self.assertEqual(
            missing,
            [],
            f"call {COMPLETE}(hidden_states) before sending to the next pipeline rank",
        )


if __name__ == "__main__":
    unittest.main()
