# SPDX-License-Identifier: Apache-2.0

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON_ROOT = REPO_ROOT / "python"

MIGRATION_ALLOWED_MULTIMODAL_GEN_IMPORTS: set[tuple[str, int, str]] = set()


class TestOmniSessionImportBoundaries(unittest.TestCase):
    def test_multimodal_gen_does_not_import_srt_omni_session(self):
        violations = _find_imports(
            PYTHON_ROOT / "sglang" / "multimodal_gen",
            forbidden_prefixes=("sglang.srt.omni_session",),
        )
        violations = _without_allowed(
            violations,
            allowed=MIGRATION_ALLOWED_MULTIMODAL_GEN_IMPORTS,
        )

        self.assertEqual([], _format_violations(violations))

    def test_ordinary_srt_does_not_import_omni_session_policy(self):
        violations = _find_imports(
            PYTHON_ROOT / "sglang" / "srt",
            forbidden_prefixes=(
                "sglang.srt.omni_session.coordinator",
                "sglang.srt.omni_session.sensenova_u1",
            ),
            skip_dirs={PYTHON_ROOT / "sglang" / "srt" / "omni_session"},
        )

        self.assertEqual([], _format_violations(violations))

    def test_srt_omni_session_does_not_import_multimodal_gen(self):
        violations = _find_imports(
            PYTHON_ROOT / "sglang" / "srt" / "omni_session",
            forbidden_prefixes=("sglang.multimodal_gen",),
        )

        self.assertEqual([], _format_violations(violations))

    def test_multimodal_gen_u1_uses_only_context_ops(self):
        root = (
            PYTHON_ROOT
            / "sglang"
            / "multimodal_gen"
            / "runtime"
            / "pipelines_core"
            / "stages"
            / "model_specific_stages"
        )
        files = [
            PYTHON_ROOT
            / "sglang"
            / "multimodal_gen"
            / "configs"
            / "sample"
            / "sensenova_u1.py",
            PYTHON_ROOT
            / "sglang"
            / "multimodal_gen"
            / "runtime"
            / "pipelines"
            / "sensenova_u1.py",
            root / "sensenova_u1_executor.py",
            root / "sensenova_u1_prepare.py",
            root / "sensenova_u1_denoise.py",
            root / "sensenova_u1_decode.py",
        ]
        forbidden_patterns = (
            "bridge.runtime",
            "srt_request_executor",
            "sensenova_u1_bridge",
            "sensenova_u1_contexts",
            "UGSessionRuntime",
            "forward_vlm",
            "VLM",
            "vlm",
            "think_max_new_tokens",
        )
        violations = []
        for path in files:
            text = path.read_text(encoding="utf-8-sig")
            for pattern in forbidden_patterns:
                if pattern in text:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)} contains {pattern}"
                    )

        self.assertEqual([], violations)


def _find_imports(
    root: Path,
    *,
    forbidden_prefixes: tuple[str, ...],
    skip_dirs: set[Path] | None = None,
) -> list[tuple[Path, int, str]]:
    skip_dirs = {path.resolve() for path in (skip_dirs or set())}
    violations: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        if _is_skipped(path, skip_dirs):
            continue
        if _is_test_or_cache_path(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            raise AssertionError(f"Failed to parse {path}: {exc}") from exc

        for node in ast.walk(tree):
            module = _imported_module(node)
            if module is None:
                continue
            if any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in forbidden_prefixes
            ):
                violations.append((path, getattr(node, "lineno", 0), module))
    return violations


def _imported_module(node: ast.AST) -> str | None:
    if isinstance(node, ast.ImportFrom):
        return node.module
    if isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names]
        return ",".join(modules)
    return None


def _is_test_or_cache_path(path: Path) -> bool:
    parts = set(path.parts)
    return (
        "__pycache__" in parts
        or "test" in parts
        or "tests" in parts
        or path.name.startswith("test_")
    )


def _is_skipped(path: Path, skip_dirs: set[Path]) -> bool:
    resolved = path.resolve()
    return any(
        resolved == skip_dir or skip_dir in resolved.parents for skip_dir in skip_dirs
    )


def _format_violations(violations: list[tuple[Path, int, str]]) -> list[str]:
    return [
        f"{path.relative_to(REPO_ROOT)}:{lineno} imports {module}"
        for path, lineno, module in violations
    ]


def _without_allowed(
    violations: list[tuple[Path, int, str]],
    *,
    allowed: set[tuple[str, int, str]],
) -> list[tuple[Path, int, str]]:
    return [
        violation
        for violation in violations
        if (
            str(violation[0].relative_to(REPO_ROOT)),
            violation[1],
            violation[2],
        )
        not in allowed
    ]


if __name__ == "__main__":
    unittest.main()
