#!/usr/bin/env python3
"""Generate the first-party Bazel graph used by registered tests.

The source split, registered test BUILD files, validation, and changed-file
selection all use the resolver in this file. Keeping that mapping in one place
prevents labels such as ``sglang.srt.environ`` from drifting between ``:pkg``
and file-split targets.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_WORKSPACE = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
REPO_ROOT = Path(_WORKSPACE).resolve() if _WORKSPACE else Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SGLANG_ROOT = PYTHON_ROOT / "sglang"
REGISTERED_ROOT = REPO_ROOT / "test" / "registered"
REGISTERED_TEST_DIRS = {
    REGISTERED_ROOT / "unit" / "platforms",
}
FILE_SPLIT_DIRS = {
    SGLANG_ROOT / "srt",
    SGLANG_ROOT / "srt" / "debug_utils",
    SGLANG_ROOT / "test" / "ci",
}
SELECTED_PACKAGE_DIRS = {
    SGLANG_ROOT / "srt",
    SGLANG_ROOT / "srt" / "debug_utils",
    SGLANG_ROOT / "srt" / "platforms",
    SGLANG_ROOT / "srt" / "plugins",
    SGLANG_ROOT / "test" / "ci",
}
SELECTED_SRCS = {
    SGLANG_ROOT / "srt": {"environ.py"},
    SGLANG_ROOT / "srt" / "debug_utils": {"cuda_coredump.py"},
    SGLANG_ROOT / "test" / "ci": {"__init__.py", "ci_register.py"},
}

IGNORED_DIRS = {"__pycache__", ".pytest_cache"}
REGISTER_BACKENDS = {
    "register_amd_ci": "amd",
    "register_cpu_ci": "cpu",
    "register_cuda_ci": "cuda",
    "register_npu_ci": "npu",
}
BACKEND_MACROS = {
    "amd": "sgl_amd_test",
    "cpu": "sgl_cpu_test",
    "cuda": "sgl_cuda_test",
    "npu": "sgl_npu_test",
}
PARAM_ORDER = ("est_time", "suite", "nightly", "disabled")


@dataclass(frozen=True)
class Registration:
    backend: str
    est_time: int
    suite: str
    nightly: bool
    disabled: str | None


def iter_source_package_dirs() -> list[Path]:
    return sorted(path for path in SELECTED_PACKAGE_DIRS if list(path.glob("*.py")))


def _module_for_file(path: Path) -> str:
    try:
        rel = path.relative_to(PYTHON_ROOT).with_suffix("")
    except ValueError:
        rel = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(rel.parts)


def target_for_module(module: str) -> str | None:
    if not module.startswith("sglang"):
        return None

    parts = module.split(".")
    while parts:
        candidate = PYTHON_ROOT.joinpath(*parts)
        candidate_file = candidate.with_suffix(".py")
        if candidate_file.exists():
            selected = SELECTED_SRCS.get(candidate_file.parent)
            if selected is not None and candidate_file.name not in selected:
                return None
            package = candidate_file.parent.relative_to(REPO_ROOT)
            if candidate_file.parent in FILE_SPLIT_DIRS:
                return "//" + package.as_posix() + ":" + candidate_file.stem
            if candidate_file.parent in SELECTED_PACKAGE_DIRS:
                return "//" + package.as_posix() + ":pkg"
            return None
        if candidate.is_dir() and list(candidate.glob("*.py")):
            if candidate in SELECTED_PACKAGE_DIRS:
                return "//python/" + "/".join(parts) + ":pkg"
            return None
        parts.pop()
    return None


def source_target_for_file(path: Path) -> str | None:
    try:
        rel = path.relative_to(PYTHON_ROOT)
    except ValueError:
        return None
    if not rel.parts or rel.parts[0] != "sglang" or path.suffix != ".py":
        return None
    module = ".".join(rel.with_suffix("").parts)
    return target_for_module(module)


def _first_party_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError as exc:
        raise RuntimeError(f"failed to parse {path}: {exc}") from exc

    imports: set[str] = set()
    current_module = _module_for_file(path)
    current_pkg = current_module.rsplit(".", 1)[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("sglang"):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                base_parts = current_pkg.split(".")
                if node.level > len(base_parts):
                    continue
                prefix = ".".join(base_parts[: len(base_parts) - node.level + 1])
                module = f"{prefix}.{module}" if module else prefix
            if module.startswith("sglang"):
                imports.add(module)

    return imports


def _package_deps(path: Path, srcs: list[Path], name: str = "pkg") -> list[str]:
    package_label = "//" + path.relative_to(REPO_ROOT).as_posix() + ":" + name
    if package_label == "//python/sglang/srt/debug_utils:cuda_coredump":
        return []
    deps = set()
    for py_file in srcs:
        for imported in _first_party_imports(py_file):
            dep = target_for_module(imported)
            if dep and dep != package_label:
                deps.add(dep)
    return sorted(deps)


def _import_root_for(path: Path) -> str:
    rel = path.relative_to(PYTHON_ROOT)
    depth = len(rel.parts)
    return "/".join([".."] * depth) if depth else "."


def _render_rule(path: Path, name: str, srcs: list[Path]) -> list[str]:
    deps = _package_deps(path, srcs, name)
    lines = [
        "py_library(",
        f'    name = "{name}",',
        "    srcs = [",
    ]
    lines.extend(f'        "{src.name}",' for src in sorted(srcs))
    lines.extend(
        [
            "    ],",
            f'    imports = ["{_import_root_for(path)}"],',
        ]
    )
    if deps:
        lines.append("    deps = [")
        lines.extend(f'        "{dep}",' for dep in deps)
        lines.append("    ],")
    lines.extend(
        [
            '    visibility = ["//visibility:public"],',
            ")",
            "",
        ]
    )
    return lines


def _render_source_build(path: Path) -> str:
    lines = [
        "# Generated by tools/bazel/gen_source_graph.py. Do not edit.",
        "",
        'load("@rules_python//python:defs.bzl", "py_library")',
        "",
    ]
    selected = SELECTED_SRCS.get(path)
    srcs = sorted(
        src for src in path.glob("*.py") if selected is None or src.name in selected
    )
    if path in FILE_SPLIT_DIRS:
        init = [p for p in srcs if p.name == "__init__.py"]
        if init:
            lines.extend(_render_rule(path, "pkg", init))
        else:
            lines.extend(_render_rule(path, "pkg", []))
        for src in srcs:
            if src.name == "__init__.py":
                continue
            lines.extend(_render_rule(path, src.stem, [src]))
    else:
        lines.extend(_render_rule(path, "pkg", srcs))
    return "\n".join(lines)


def _literal(node: ast.AST, path: Path) -> object:
    if not isinstance(node, ast.Constant):
        raise ValueError(f"{path}: register_*_ci arguments must be constants")
    return node.value


def _parse_registration(call: ast.Call, path: Path) -> Registration | None:
    if not isinstance(call.func, ast.Name):
        return None
    backend = REGISTER_BACKENDS.get(call.func.id)
    if backend is None:
        return None
    if len(call.args) > len(PARAM_ORDER):
        raise ValueError(f"{path}: too many positional arguments in {call.func.id}()")

    values: dict[str, object | None] = {name: None for name in PARAM_ORDER}
    seen = set()
    for name, arg in zip(PARAM_ORDER, call.args):
        values[name] = _literal(arg, path)
        seen.add(name)
    for keyword in call.keywords:
        if keyword.arg not in PARAM_ORDER:
            raise ValueError(f"{path}: unsupported argument in {call.func.id}()")
        if keyword.arg in seen:
            raise ValueError(f"{path}: duplicated argument in {call.func.id}()")
        values[keyword.arg] = _literal(keyword.value, path)
        seen.add(keyword.arg)

    est_time = values["est_time"]
    suite = values["suite"]
    nightly = values["nightly"] if values["nightly"] is not None else False
    disabled = values["disabled"]
    if not isinstance(est_time, (int, float)):
        raise ValueError(f"{path}: est_time must be a number")
    if not isinstance(suite, str):
        raise ValueError(f"{path}: suite must be a string")
    if not isinstance(nightly, bool):
        raise ValueError(f"{path}: nightly must be a boolean")
    if disabled is not None and not isinstance(disabled, str):
        raise ValueError(f"{path}: disabled must be a string or None")
    return Registration(
        backend=backend,
        est_time=math.ceil(est_time),
        suite=suite,
        nightly=nightly,
        disabled=disabled,
    )


def _registrations(path: Path) -> list[Registration]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    regs = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            reg = _parse_registration(stmt.value, path)
            if reg is not None:
                regs.append(reg)
    return regs


def _sanitize(value: str) -> str:
    chars = []
    last_underscore = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            last_underscore = False
        elif not last_underscore:
            chars.append("_")
            last_underscore = True
    return "".join(chars).strip("_")


def _enabled(regs: list[Registration]) -> list[Registration]:
    return [reg for reg in regs if reg.disabled is None]


def _test_target_name(filename: str, reg: Registration, needs_suffix: bool) -> str:
    name = Path(filename).stem
    if not needs_suffix:
        return name
    suffix = f"{reg.backend}_{_sanitize(reg.suite)}"
    if reg.nightly:
        suffix += "_nightly"
    return f"{name}_{suffix}"


def _test_deps(path: Path) -> list[str]:
    return sorted({
        dep
        for imported in _first_party_imports(path)
        for dep in [target_for_module(imported)]
        if dep
    })


def _render_test_rule(path: Path, reg: Registration, needs_suffix: bool) -> list[str]:
    filename = path.name
    lines = [
        f"{BACKEND_MACROS[reg.backend]}(",
        f'    name = "{_test_target_name(filename, reg, needs_suffix)}",',
        f'    srcs = ["{filename}"],',
        f"    est_time = {reg.est_time},",
    ]
    if reg.nightly:
        lines.append("    nightly = True,")
    lines.append(f'    suite = "{reg.suite}",')
    deps = _test_deps(path)
    if deps:
        lines.append("    deps = [")
        lines.extend(f'        "{dep}",' for dep in deps)
        lines.append("    ],")
    lines.extend([")", ""])
    return lines


def _render_registered_test_build(path: Path) -> str:
    tests: list[tuple[Path, list[Registration]]] = []
    macros = set()
    for test in sorted(path.glob("test_*.py")):
        enabled = _enabled(_registrations(test))
        if not enabled:
            continue
        tests.append((test, enabled))
        macros.update(BACKEND_MACROS[reg.backend] for reg in enabled)

    if not tests:
        return ""

    macro_loads = ", ".join(f'"{macro}"' for macro in sorted(macros))
    lines = [
        f'load("//tools/bazel:sgl_defs.bzl", {macro_loads})',
        "",
    ]
    for test, regs in tests:
        for reg in regs:
            lines.extend(_render_test_rule(test, reg, len(regs) > 1))
    return "\n".join(lines)


def generated_outputs() -> dict[Path, str]:
    outputs = {
        path / "BUILD.bazel": _render_source_build(path)
        for path in iter_source_package_dirs()
    }
    outputs.update(
        {
            path / "BUILD.bazel": _render_registered_test_build(path)
            for path in sorted(REGISTERED_TEST_DIRS)
        }
    )
    return outputs


def _write() -> None:
    for path, content in generated_outputs().items():
        path.write_text(content)


def _check() -> int:
    exit_code = 0
    for path, expected in generated_outputs().items():
        actual = path.read_text() if path.exists() else ""
        if actual != expected:
            exit_code = 1
            sys.stderr.writelines(
                difflib.unified_diff(
                    actual.splitlines(keepends=True),
                    expected.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=f"{path} (expected)",
                )
            )
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return _check()
    _write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
