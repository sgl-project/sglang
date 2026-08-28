"""The inherited capture loop must not read state a subclass never set.

Four speculative runners subclass ``DecodeCudaGraphRunner`` and inherit its
``capture()`` / ``_capture_one_stream()`` while deliberately NOT calling its
``__init__`` -- they hand-initialize the fields the inherited loop reads. That
is a contract with no compiler behind it: adding one ``self.new_field`` read to
``_capture_one_stream`` takes down every one of those runners at startup with
``AttributeError``, and it does so on the shared full-draft-graph path, so a
feature that only the target runner can use still stops a service that has the
feature switched off.

That has already happened once: ``_capture_one_stream`` grew an unconditional
``if self.ragged_verify_mode:`` dispatch, and all four runners lost the ability
to capture. The fix declared the ragged capture fields as class-level defaults
on ``DecodeCudaGraphRunner``; this test is what keeps the next field from
repeating it.

It is a source-level (AST) contract check rather than a live capture: capturing
needs a model, a device and a real attention backend, so a runnable test would
have to be a GPU test, and a GPU test is not where a missing attribute should
first be noticed. What it computes:

  required = attribute names read off ``self`` inside the capture loop that graph-tier mode
             changed (``_capture_one_stream``, ``_capture_ragged_tiers``, plus
             the parent helpers they call on ``self``)
  provided = names each subclass assigns to ``self`` in ``__init__`` (following
             the helpers ``__init__`` calls) + class-level defaults on the
             subclass or on ``DecodeCudaGraphRunner`` + method/property names
             visible on either class

and requires ``required <= provided`` for every subclass.

Scope, stated so the next reader does not assume more: the closure stops at the
methods the subclasses override (``capture_one_shape``) and does not walk
``capture()`` itself. Those read state that ``capture()`` installs at capture
time (``self.stream``, ``self.bs``, ...), never ``__init__`` state, so widening
the closure would turn this into a list of exceptions rather than a contract.
"""

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Read the sources from the installed/importable sglang, not from a path
# relative to this test file: the same contract has to hold for whichever tree
# CI actually runs.
_SRT = Path(sglang.__file__).resolve().parent / "srt"
_PARENT_FILE = _SRT / "model_executor" / "runner" / "decode_cuda_graph_runner.py"
_BASE_FILES = (
    _SRT / "model_executor" / "runner" / "base_cuda_graph_runner.py",
    _SRT / "model_executor" / "runner" / "base_runner.py",
)
# Every direct subclass that inherits the capture loop while skipping
# DecodeCudaGraphRunner.__init__. NPUGraphRunner / XPUGraphRunner also subclass
# it but call super().__init__(), so the contract holds for them by
# construction.
_SUBCLASSES = (
    (
        "EAGLEDraftCudaGraphRunner",
        _SRT / "speculative" / "eagle_draft_cuda_graph_runner.py",
    ),
    (
        "EAGLEDraftExtendCudaGraphRunner",
        _SRT / "speculative" / "eagle_draft_extend_cuda_graph_runner.py",
    ),
    (
        "FrozenKVMTPCudaGraphRunner",
        _SRT / "speculative" / "frozen_kv_mtp_cuda_graph_runner.py",
    ),
    (
        "MultiLayerEagleDraftExtendCudaGraphRunner",
        _SRT / "speculative" / "multi_layer_eagle_draft_extend_cuda_graph_runner.py",
    ),
)
# The capture loop graph-tier mode rewrote. `capture_one_shape` is excluded on purpose: all
# four subclasses override it, so their own version is what runs.
_CAPTURE_ENTRYPOINTS = ("_capture_one_stream", "_capture_ragged_tiers")


def _class_def(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{path}: no class {name}")


def _methods(cls: ast.ClassDef) -> set:
    return {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_level_names(cls: ast.ClassDef) -> set:
    """Names bound in the class body itself (the declared defaults)."""
    names = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _method_node(cls: ast.ClassDef, name: str):
    for node in cls.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    return None


def _self_reads(node: ast.AST) -> set:
    """`self.x` in a load context -> {"x"}.

    `getattr(self, "x", default)` is deliberately NOT collected: it does not
    produce a `self.x` attribute node, and a caller that spells the read that
    way has already declared the field optional.
    """
    reads = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "self"
            and isinstance(sub.ctx, ast.Load)
        ):
            reads.add(sub.attr)
    return reads


def _self_writes(node: ast.AST) -> set:
    writes = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
            for target in targets:
                for leaf in ast.walk(target):
                    if (
                        isinstance(leaf, ast.Attribute)
                        and isinstance(leaf.value, ast.Name)
                        and leaf.value.id == "self"
                    ):
                        writes.add(leaf.attr)
    return writes


def _self_calls(node: ast.AST) -> set:
    calls = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "self"
        ):
            calls.add(sub.func.attr)
    return calls


def _closure(cls: ast.ClassDef, entrypoints, *, skip: set):
    """Fixpoint over `self.method()` calls, returning the reads and the methods walked."""
    pending = [name for name in entrypoints]
    walked = set()
    reads = set()
    while pending:
        name = pending.pop()
        if name in walked or name in skip:
            continue
        node = _method_node(cls, name)
        if node is None:
            continue
        walked.add(name)
        reads |= _self_reads(node)
        pending.extend(_self_calls(node))
    return reads, walked


class TestDecodeCaptureContract(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = _class_def(_PARENT_FILE, "DecodeCudaGraphRunner")
        cls.parent_methods = _methods(cls.parent)
        for path in _BASE_FILES:
            for base in ("BaseCudaGraphRunner", "BaseRunner"):
                try:
                    cls.parent_methods |= _methods(_class_def(path, base))
                except AssertionError:
                    continue
        cls.required, cls.walked = _closure(
            cls.parent, _CAPTURE_ENTRYPOINTS, skip={"capture_one_shape"}
        )

    def test_capture_entrypoints_still_exist(self):
        """If the loop is renamed, the rest of this file silently checks nothing."""
        self.assertEqual(
            set(_CAPTURE_ENTRYPOINTS), self.walked & set(_CAPTURE_ENTRYPOINTS)
        )
        self.assertIn("ragged_verify_mode", self.required)
        self.assertIn("capture_num_tokens", self.required)

    def test_ragged_capture_fields_have_class_level_defaults(self):
        """The two fields the loop reads before any subclass state exists."""
        declared = _class_level_names(self.parent)
        self.assertIn("ragged_verify_mode", declared)
        self.assertIn("capture_num_tokens", declared)

    def test_every_bypassing_subclass_satisfies_the_capture_contract(self):
        parent_class_level = _class_level_names(self.parent)
        for name, path in _SUBCLASSES:
            with self.subTest(runner=name):
                sub = _class_def(path, name)
                init = _method_node(sub, "__init__")
                self.assertIsNotNone(init, f"{name} has no __init__")
                # Follow the helpers __init__ calls on self, so a runner that
                # sets its fields in a private setup method still counts.
                provided = set()
                pending = ["__init__"]
                seen = set()
                while pending:
                    method_name = pending.pop()
                    if method_name in seen:
                        continue
                    node = _method_node(sub, method_name)
                    if node is None:
                        continue
                    seen.add(method_name)
                    provided |= _self_writes(node)
                    pending.extend(_self_calls(node))
                provided |= _class_level_names(sub)
                provided |= parent_class_level
                provided |= _methods(sub)
                provided |= self.parent_methods
                missing = sorted(self.required - provided)
                self.assertEqual(
                    missing,
                    [],
                    f"{name} inherits the capture loop but never provides "
                    f"{missing}; declare a class-level default on "
                    f"DecodeCudaGraphRunner or set it in {name}.__init__",
                )

    def test_subclasses_really_do_bypass_the_parent_init(self):
        """The premise. If a runner starts calling super().__init__(), it no
        longer needs the class-level defaults and should be dropped from
        _SUBCLASSES rather than left here asserting nothing."""
        for name, path in _SUBCLASSES:
            with self.subTest(runner=name):
                init = _method_node(_class_def(path, name), "__init__")
                super_init = [
                    node
                    for node in ast.walk(init)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "__init__"
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == "super"
                ]
                self.assertEqual(super_init, [], f"{name} now calls super().__init__()")


if __name__ == "__main__":
    unittest.main()
