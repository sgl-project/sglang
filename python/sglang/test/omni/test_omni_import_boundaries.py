# SPDX-License-Identifier: Apache-2.0

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON_ROOT = REPO_ROOT / "python"


class TestOmniImportBoundaries(unittest.TestCase):
    def test_runtime_layers_keep_omni_import_boundaries(self):
        cases = [
            (
                PYTHON_ROOT / "sglang" / "multimodal_gen",
                (
                    "sglang.omni.backends.ar.srt",
                    "sglang.srt.omni_session",
                ),
            ),
            (
                PYTHON_ROOT / "sglang" / "srt",
                (
                    "sglang.omni.configs",
                    "sglang.omni.core.coordinator",
                    "sglang.omni.model_adapters.sensenova_u1.session_adapter",
                ),
            ),
        ]

        for root, forbidden_prefixes in cases:
            with self.subTest(root=root.relative_to(REPO_ROOT)):
                violations = _find_imports(
                    root,
                    forbidden_prefixes=forbidden_prefixes,
                )

                self.assertEqual([], _format_violations(violations))

    def test_omni_runtime_is_model_agnostic(self):
        root = PYTHON_ROOT / "sglang" / "omni" / "runtime"
        forbidden_patterns = ("sensenova", "SenseNova", "U1", "u1")
        violations = []
        for path in sorted(root.rglob("*.py")):
            if _is_test_or_cache_path(path):
                continue
            text = path.read_text(encoding="utf-8-sig")
            for pattern in forbidden_patterns:
                if pattern in text:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)} contains {pattern}"
                    )

        self.assertEqual([], violations)


def _find_imports(root: Path, *, forbidden_prefixes: tuple[str, ...]):
    violations = []
    for path in sorted(root.rglob("*.py")):
        if _is_test_or_cache_path(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            raise AssertionError(f"Failed to parse {path}: {exc}") from exc
        for node in ast.walk(tree):
            modules = _imported_modules(node)
            for module in modules:
                if any(
                    module == forbidden or module.startswith(f"{forbidden}.")
                    for forbidden in forbidden_prefixes
                ):
                    violations.append((path, getattr(node, "lineno", 0), module))
    return violations


def _imported_modules(node: ast.AST) -> list[str]:
    if isinstance(node, ast.ImportFrom):
        return [node.module] if node.module else []
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return []


def _is_test_or_cache_path(path: Path) -> bool:
    parts = set(path.parts)
    return (
        "__pycache__" in parts
        or "test" in parts
        or "tests" in parts
        or path.name.startswith("test_")
    )


def _format_violations(violations):
    return [
        f"{path.relative_to(REPO_ROOT)}:{lineno} imports {module}"
        for path, lineno, module in violations
    ]


if __name__ == "__main__":
    unittest.main()
