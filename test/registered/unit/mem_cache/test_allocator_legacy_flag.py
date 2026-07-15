from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt  # noqa: E402
from sglang.srt.mem_cache.allocator import (  # noqa: E402
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (  # noqa: E402
    SWATokenToKVPoolAllocator,
)

_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))

_FLAG = "uses_legacy_real_length_alloc"

_KNOWN_WRAPPERS = {"DeepSeekV4HiSparseTokenToKVPoolAllocator"}


def _is_allocator_param(arg: ast.arg) -> bool:
    if "allocator" in arg.arg:
        return True
    return arg.annotation is not None and "Allocator" in ast.unparse(arg.annotation)


def _wrapped_allocator_params(class_node: ast.ClassDef) -> list[str]:
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        args = node.args.args + node.args.kwonlyargs
        return [a.arg for a in args if a.arg != "self" and _is_allocator_param(a)]
    return []


def _find_allocator_wrappers() -> dict[str, ast.ClassDef]:
    wrappers = {}
    for path in sorted(_SRT_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any("Allocator" in ast.unparse(b) for b in node.bases):
                continue
            if _wrapped_allocator_params(node):
                wrappers[node.name] = node
    return wrappers


def _forwarding_source(class_node: ast.ClassDef) -> str:
    sources = []
    for node in ast.walk(class_node):
        if isinstance(node, ast.FunctionDef) and node.name == _FLAG:
            sources.append(ast.unparse(node))
        elif isinstance(node, ast.Assign):
            targets = [ast.unparse(t) for t in node.targets]
            if f"self.{_FLAG}" in targets:
                sources.append(ast.unparse(node))
    return "\n".join(sources)


class TestUsesLegacyRealLengthAlloc(CustomTestCase):
    def test_base_allocator_declares_the_page_aligned_contract(self):
        """The default must be the aligned contract, so only opt-outs carry the burden."""
        self.assertIs(BaseTokenToKVPoolAllocator.uses_legacy_real_length_alloc, False)

    def test_in_tree_allocators_inherit_the_page_aligned_contract(self):
        """An in-tree allocator silently flipping to legacy would disable the guardrail."""
        for allocator_cls in (
            TokenToKVPoolAllocator,
            PagedTokenToKVPoolAllocator,
            SWATokenToKVPoolAllocator,
        ):
            with self.subTest(allocator=allocator_cls.__name__):
                self.assertIs(allocator_cls.uses_legacy_real_length_alloc, False)

    def test_the_wrapper_census_matches_the_pin(self):
        """A new allocator-wrapping allocator must be pinned, so it cannot skip the check below."""
        found = set(_find_allocator_wrappers())
        self.assertEqual(
            found,
            _KNOWN_WRAPPERS,
            f"the set of allocators taking another allocator changed: "
            f"added={sorted(found - _KNOWN_WRAPPERS)}, "
            f"removed={sorted(_KNOWN_WRAPPERS - found)}. Every such wrapper must "
            f"forward {_FLAG} from the allocator it wraps; pin it here once it does.",
        )

    def test_every_wrapper_forwards_the_flag_from_the_allocator_it_wraps(self):
        """Inheriting the base default instead of forwarding silently loses the wrapped declaration."""
        for name, class_node in sorted(_find_allocator_wrappers().items()):
            with self.subTest(allocator=name):
                source = _forwarding_source(class_node)
                self.assertTrue(
                    source, f"{name} never defines {_FLAG}; it inherits the base default"
                )
                wrapped = _wrapped_allocator_params(class_node)
                self.assertTrue(
                    any(f"{param}.{_FLAG}" in source for param in wrapped),
                    f"{name} defines {_FLAG} without reading it off {wrapped}:\n{source}",
                )


if __name__ == "__main__":
    unittest.main()
