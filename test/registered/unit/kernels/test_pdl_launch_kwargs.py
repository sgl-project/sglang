"""Triton launches must pass their PDL constexprs on every GPU.

When the non-PDL branch omits a constexpr such as ``USE_GDC``, the kernel
falls back to its declared default, but ``torch.compile`` builds the launcher
from the kernel signature and its arity drifts by one on pre-Hopper GPUs:
``Incorrect number of arguments passed to kernel: ... expected [..., 'USE_GDC']``.
``launch_pdl`` is launch metadata, not a kernel parameter, so it stays
conditional.
"""

import ast
from pathlib import Path

import sglang.kernels
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_KERNELS_ROOT = Path(sglang.kernels.__file__).parent
_LAUNCH_METADATA = {"launch_pdl"}


def _is_pdl_check(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "is_arch_support_pdl"
    )


def _string_keys(node: ast.expr) -> set[str] | None:
    if not isinstance(node, ast.Dict):
        return None
    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _missing_constexprs() -> list[str]:
    missing = []
    for path in sorted(_KERNELS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.IfExp) and _is_pdl_check(node.test)):
                continue
            enabled = _string_keys(node.body)
            disabled = _string_keys(node.orelse)
            if enabled is None or disabled is None:
                continue
            omitted = enabled - _LAUNCH_METADATA - disabled
            if omitted:
                rel = path.relative_to(_KERNELS_ROOT)
                missing.append(f"{rel}:{node.lineno} omits {sorted(omitted)}")
    return missing


def test_pdl_constexprs_are_passed_without_pdl():
    assert _missing_constexprs() == []


def test_scan_finds_pdl_launch_sites():
    # Guards against the scan silently matching nothing after a refactor.
    sites = 0
    for path in _KERNELS_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        sites += sum(
            isinstance(node, ast.IfExp) and _is_pdl_check(node.test)
            for node in ast.walk(tree)
        )
    assert sites >= 10


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
