from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path
from typing import Callable, Optional

import sglang.srt
from sglang.test.test_utils import CustomTestCase

_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))

_METHOD = "cache_finished_req"
_RESULT_TYPE = "CacheFinishedReqResult"
_ROOT_CLASS = "BasePrefixCache"

_WRAPPER_CLASS = "StreamingSession"
_DELEGATION_ATTR = "inner"

_SUPER_DELEGATING_CLASSES = frozenset({"LMCRadixCache", "FlexKVRadixCache"})
_SUPER_RESULT_NAME = "cache_finished_req_result"

_PINNED_PROVIDERS = {
    "BasePrefixCache": "BasePrefixCache",
    "ChunkCache": "ChunkCache",
    "SWAChunkCache": "ChunkCache",
    "PureSWAChunkCache": "PureSWAChunkCache",
    "MambaRadixCache": "MambaRadixCache",
    "HiMambaRadixCache": "MambaRadixCache",
    "RadixCache": "RadixCache",
    "HiRadixCache": "RadixCache",
    "PureSWARadixCache": "PureSWARadixCache",
    "FlexKVRadixCache": "FlexKVRadixCache",
    "LMCRadixCache": "LMCRadixCache",
    "RadixCacheCpp": "RadixCacheCpp",
    "SWARadixCache": "SWARadixCache",
    "UnifiedRadixCache": "UnifiedRadixCache",
    "StreamingSession": "StreamingSession",
}


def _collect_classes() -> dict[str, list[ast.ClassDef]]:
    classes: dict[str, list[ast.ClassDef]] = {}
    for path in sorted(_SRT_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, []).append(node)
    return classes


def _base_names(class_node: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _linearize(
    name: str,
    classes: dict[str, list[ast.ClassDef]],
    seen: Optional[set[str]] = None,
) -> list[str]:
    # Names are resolved across the whole tree, so a subclass that shadows the
    # name of its own base looks like a cycle. Guard rather than recurse.
    seen = set() if seen is None else seen
    if name in seen:
        return []
    seen.add(name)

    order = [name]
    for class_node in classes.get(name, []):
        for base in _base_names(class_node):
            for entry in _linearize(base, classes, seen):
                if entry not in order:
                    order.append(entry)
    return order


def _own_method(class_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for node in class_node.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == _METHOD
        ):
            return node
    return None


def _is_result_construction(node: Optional[ast.expr]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _RESULT_TYPE
    )


def _is_method_call_on(
    node: Optional[ast.expr], *, receiver_matches: Callable[[ast.expr], bool]
) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == _METHOD
        and receiver_matches(node.func.value)
    )


def _is_super_call(node: Optional[ast.expr]) -> bool:
    def receiver_matches(value: ast.expr) -> bool:
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "super"
        )

    return _is_method_call_on(node, receiver_matches=receiver_matches)


def _is_inner_call(node: Optional[ast.expr]) -> bool:
    def receiver_matches(value: ast.expr) -> bool:
        return (
            isinstance(value, ast.Attribute)
            and value.attr == _DELEGATION_ATTR
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
        )

    return _is_method_call_on(node, receiver_matches=receiver_matches)


class TestCacheFinishedReqContractRatchet(CustomTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.classes = _collect_classes()
        self.subclasses = {
            name for name in self.classes if _ROOT_CLASS in _linearize(name, self.classes)
        }

    def _class_node(self, name: str) -> ast.ClassDef:
        nodes = self.classes[name]
        self.assertEqual(
            len(nodes), 1, f"{name} is defined more than once; the pin is ambiguous."
        )
        return nodes[0]

    def _provider(self, name: str) -> Optional[str]:
        for entry in _linearize(name, self.classes):
            for class_node in self.classes.get(entry, []):
                if _own_method(class_node) is not None:
                    return entry
        return None

    def _definitions(self) -> dict[str, ast.FunctionDef]:
        definitions: dict[str, ast.FunctionDef] = {}
        for name in sorted(set(_PINNED_PROVIDERS.values())):
            method = _own_method(self._class_node(name))
            self.assertIsNotNone(method, f"{name} no longer defines {_METHOD}.")
            definitions[name] = method
        return definitions

    def test_every_prefix_cache_subclass_is_pinned_to_its_provider(self):
        """Enumerate subclasses structurally, so a new backend cannot arrive unnoticed."""
        grown = self.subclasses - set(_PINNED_PROVIDERS)
        self.assertFalse(
            grown,
            f"New {_ROOT_CLASS} subclasses {sorted(grown)} are unpinned. Each must "
            f"return {_RESULT_TYPE}(unhandled_kv_start=...) rather than free the "
            "tail KV range itself (see release_kv_cache in mem_cache/common.py), "
            "then be added to _PINNED_PROVIDERS.",
        )
        shrunk = set(_PINNED_PROVIDERS) - self.subclasses
        self.assertFalse(
            shrunk, f"Pinned classes {sorted(shrunk)} are gone; drop them from the pin."
        )

    def test_inherited_definitions_resolve_to_the_pinned_provider(self):
        """A subclass silently gaining or dropping an override moves its provider."""
        for name in sorted(_PINNED_PROVIDERS):
            with self.subTest(cls=name):
                self.assertEqual(self._provider(name), _PINNED_PROVIDERS[name])

    def test_no_subclass_overrides_the_method_by_assignment(self):
        """`cache_finished_req = legacy_fn` is invisible to a def-based scan, so ban it."""
        for name in sorted(_PINNED_PROVIDERS):
            with self.subTest(cls=name):
                for node in self._class_node(name).body:
                    targets: list[ast.expr] = []
                    if isinstance(node, ast.Assign):
                        targets = list(node.targets)
                    elif isinstance(node, ast.AnnAssign):
                        targets = [node.target]
                    for target in targets:
                        self.assertFalse(
                            isinstance(target, ast.Name) and target.id == _METHOD,
                            f"{name} overrides {_METHOD} by assignment; define it so "
                            "the return-shape checks can see it.",
                        )

    def test_every_return_constructs_a_result_or_forwards_an_approved_delegation(self):
        """Only the struct or an audited delegation may be returned; None in any form is out."""
        for name, method in sorted(self._definitions().items()):
            if name == _ROOT_CLASS:
                continue
            with self.subTest(cls=name):
                for node in ast.walk(method):
                    if not isinstance(node, ast.Return):
                        continue
                    allowed = _is_result_construction(node.value)
                    if name == _WRAPPER_CLASS:
                        allowed = allowed or _is_inner_call(node.value)
                    elif name in _SUPER_DELEGATING_CLASSES:
                        allowed = allowed or (
                            isinstance(node.value, ast.Name)
                            and node.value.id == _SUPER_RESULT_NAME
                        )
                    returned = "None" if node.value is None else ast.unparse(node.value)
                    self.assertTrue(
                        allowed,
                        f"{name}.{_METHOD} returns `{returned}`. release_kv_cache "
                        f"reads anything but a {_RESULT_TYPE} as the deprecated "
                        "legacy contract and silently leaks the KV tail, so a bare "
                        "`return`, `return None`, or a conditional None is a leak.",
                    )

    def test_every_definition_ends_in_an_explicit_return(self):
        """Falling off the end returns None, which release_kv_cache reads as legacy."""
        for name, method in sorted(self._definitions().items()):
            if name == _ROOT_CLASS:
                continue
            with self.subTest(cls=name):
                self.assertIsInstance(
                    method.body[-1],
                    ast.Return,
                    f"{name}.{_METHOD} must end in an explicit return, so no control "
                    "flow can implicitly return None past the last statement.",
                )

    def test_super_delegating_backends_bind_their_result_from_super(self):
        """The forwarded name must come from super(), not from an unrelated local."""
        definitions = self._definitions()
        for name in sorted(_SUPER_DELEGATING_CLASSES):
            with self.subTest(cls=name):
                bindings = [
                    node
                    for node in ast.walk(definitions[name])
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == _SUPER_RESULT_NAME
                        for target in node.targets
                    )
                ]
                self.assertEqual(len(bindings), 1)
                self.assertTrue(_is_super_call(bindings[0].value))

    def test_the_wrapper_forwards_the_inner_cache_result_untouched(self):
        """The wrapper must hand back inner's value, including an external legacy None."""
        method = self._definitions()[_WRAPPER_CLASS]
        self.assertTrue(
            any(
                isinstance(node, ast.Return) and _is_inner_call(node.value)
                for node in ast.walk(method)
            )
        )

    def test_only_the_abstract_signature_and_the_wrapper_return_optional(self):
        """In-tree backends promise a struct; Optional exists only for legacy externals."""
        for name, method in sorted(self._definitions().items()):
            with self.subTest(cls=name):
                expected = (
                    f"Optional[{_RESULT_TYPE}]"
                    if name in (_ROOT_CLASS, _WRAPPER_CLASS)
                    else _RESULT_TYPE
                )
                annotation = ast.unparse(method.returns) if method.returns else None
                self.assertEqual(annotation, expected)


if __name__ == "__main__":
    unittest.main()
