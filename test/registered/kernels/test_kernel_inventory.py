"""CPU-only structural checks for the unified kernel tree."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

import sglang.kernels as kernels
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
KERNELS_ROOT = REPO_ROOT / "python" / "sglang" / "kernels"
OPS_ROOT = KERNELS_ROOT / "ops"
JIT_CSRC_ROOT = KERNELS_ROOT / "jit" / "csrc"
AOT_ROOT = KERNELS_ROOT / "aot"


def _directory_names(root: Path) -> set[str]:
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith((".", "__"))
        and any(path.rglob("*.py"))
    }


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.List, ast.Tuple)):
        return {name for element in target.elts for name in _target_names(element)}
    return set()


def _bound_names(statements: list[ast.stmt]) -> set[str]:
    """Collect names a module can bind without importing it."""
    names: set[str] = set()
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(statement.name)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                names.update(_target_names(target))
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            names.update(_target_names(statement.target))
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            names.update(_target_names(statement.target))
            names.update(_bound_names(statement.body))
            names.update(_bound_names(statement.orelse))
        elif isinstance(statement, ast.If):
            names.update(_bound_names(statement.body))
            names.update(_bound_names(statement.orelse))
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            names.update(_bound_names(statement.body))
        elif isinstance(statement, ast.Try):
            names.update(_bound_names(statement.body))
            names.update(_bound_names(statement.orelse))
            names.update(_bound_names(statement.finalbody))
            for handler in statement.handlers:
                names.update(_bound_names(handler.body))
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                names.update(_bound_names(case.body))
    return names


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        for target in targets:
            for name in _target_names(target):
                constants[name] = value.value
    return constants


def _source_patterns(expression: ast.expr, constants: dict[str, str]) -> list[str]:
    if isinstance(expression, (ast.List, ast.Tuple)):
        return [
            pattern
            for element in expression.elts
            for pattern in _source_patterns(element, constants)
        ]
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return [expression.value]
    if isinstance(expression, ast.Name) and expression.id in constants:
        return [constants[expression.id]]
    if isinstance(expression, ast.JoinedStr):
        parts = []
        for value in expression.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append("*")
            else:
                raise AssertionError(f"Unsupported f-string segment: {ast.dump(value)}")
        return ["".join(parts)]
    raise AssertionError(
        f"Unsupported JIT source declaration: {ast.unparse(expression)}"
    )


def test_declared_operator_groups_match_packages():
    assert set(kernels.ops.__all__) == _directory_names(OPS_ROOT)


def test_registered_kernel_test_groups_are_known():
    declared_groups = set(kernels.ops.__all__)
    registered_root = REPO_ROOT / "test" / "registered" / "kernels"
    for kind in ("ops", "benchmark"):
        unknown = _directory_names(registered_root / kind) - declared_groups
        assert (
            not unknown
        ), f"Unknown {kind} kernel group directories: {sorted(unknown)}"


def test_internal_registry_target_attributes_are_declared():
    missing = []
    for spec in kernels.registry.all_specs():
        module_name, _, attribute_path = spec.target.partition(":")
        if not module_name.startswith("sglang.kernels."):
            continue
        module_spec = importlib.util.find_spec(module_name)
        if (
            module_spec is None
            or module_spec.origin is None
            or not module_spec.origin.endswith(".py")
        ):
            continue
        tree = ast.parse(Path(module_spec.origin).read_text())
        root_attribute = attribute_path.split(".", 1)[0]
        if root_attribute not in _bound_names(tree.body):
            missing.append(spec.target)
    assert not missing, f"KernelSpec targets missing attributes: {missing}"


# `load_jit` takes in-tree names and absolute paths on the same keyword, so this
# check can only reach the declarations spelled out in the source. A module that
# assembles its file list at runtime from a package outside `jit/csrc` has no
# in-tree name to verify and belongs here; there is none at the moment.
_RUNTIME_JIT_SOURCE_MODULES: set[str] = set()


def test_jit_source_declarations_exist():
    missing = []
    unsupported = []
    for python_file in OPS_ROOT.rglob("*.py"):
        if python_file.relative_to(OPS_ROOT).as_posix() in _RUNTIME_JIT_SOURCE_MODULES:
            continue
        tree = ast.parse(python_file.read_text())
        constants = _module_string_constants(tree)
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            function_name = (
                call.func.id
                if isinstance(call.func, ast.Name)
                else call.func.attr if isinstance(call.func, ast.Attribute) else None
            )
            if function_name != "load_jit":
                continue
            for keyword in call.keywords:
                if keyword.arg not in {"cpp_files", "cuda_files"}:
                    continue
                try:
                    patterns = _source_patterns(keyword.value, constants)
                except AssertionError as exc:
                    unsupported.append(f"{python_file.relative_to(REPO_ROOT)}: {exc}")
                    continue
                for pattern in patterns:
                    matches = list(JIT_CSRC_ROOT.glob(pattern))
                    if not matches:
                        missing.append(
                            f"{python_file.relative_to(REPO_ROOT)} -> {pattern}"
                        )
    assert not unsupported, "Unsupported JIT source declarations:\n" + "\n".join(
        unsupported
    )
    assert not missing, "Missing JIT sources:\n" + "\n".join(missing)


def test_aot_compilation_units_are_accounted_for():
    manifests = [
        AOT_ROOT / "CMakeLists.txt",
        AOT_ROOT / "setup_metal.py",
        AOT_ROOT / "setup_musa.py",
        AOT_ROOT / "setup_rocm.py",
        AOT_ROOT / "csrc" / "cpu" / "CMakeLists.txt",
        *sorted((AOT_ROOT / "cmake").rglob("*.cmake")),
    ]
    manifest_text = "\n".join(path.read_text() for path in manifests)
    source_text = {
        path: path.read_text(errors="ignore")
        for path in (AOT_ROOT / "csrc").rglob("*")
        if path.is_file()
    }
    compilation_suffixes = {".cc", ".cpp", ".cu", ".hip", ".metal", ".mu"}
    missing = []
    for source in source_text:
        if source.suffix not in compilation_suffixes:
            continue
        if AOT_ROOT / "csrc" / "cpu" in source.parents:
            # The CPU build intentionally uses file(GLOB_RECURSE ... *.cpp).
            continue
        relative_path = source.relative_to(AOT_ROOT).as_posix()
        if relative_path in manifest_text:
            continue
        if any(
            source.name in text
            for other_source, text in source_text.items()
            if other_source != source
        ):
            # Some CUDA translation units are included by another source.
            continue
        missing.append(relative_path)
    assert not missing, f"AOT compilation units missing from build manifests: {missing}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
