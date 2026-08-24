"""Guards that keep the ``diffusion`` package's import surface from eroding.

The reorganization only stays useful if two invariants hold:

1. runtime code imports from ``sglang.kernels.ops.diffusion`` and not from a
   submodule, so the internal layout can move without touching call sites;
2. the facade's ``_EXPORTS`` table and the registry's ``_SPECS`` table both
   point at symbols that actually exist.

Neither is checkable by the type system, and both fail silently -- a stale
``_EXPORTS`` entry only raises when some model happens to call that kernel, on
a GPU, at serving time.  These are pure-CPU tests: they read the tables and
resolve them with ``importlib``/``ast`` without importing torch backends.
"""

import ast
import importlib
import pathlib
import subprocess
import sys

import pytest

from sglang.kernels.ops.diffusion import _EXPORTS, _SPECS
from sglang.kernels.registry import registry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=69, suite="base-a-test-cpu")

PACKAGE = "sglang.kernels.ops.diffusion"
_PACKAGE_DIR = pathlib.Path(importlib.import_module(PACKAGE).__file__ or "").parent
_REPO_ROOT = _PACKAGE_DIR.parents[4]  # <repo>/python/sglang/kernels/ops/diffusion

# Backend-specific test files may name a leaf module on purpose; everything
# else -- all runtime code -- must go through the facade.
_DEEP_IMPORT_ALLOWLIST = {
    "python/sglang/multimodal_gen/test/unit/test_latent_upsampler_group_norm_silu.py",
    "test/registered/kernels/ops/diffusion/test_model_fast_paths.py",
    "test/registered/kernels/ops/diffusion/test_sites.py",
    # This test exercises the pure-Torch fallback implementation directly.
    "test/registered/unit/utils/test_diffusion_torch_fallback.py",
}


def _module_defines(module_path: str) -> set[str]:
    """Top-level names bound by a submodule, without importing it.

    Importing would pull in Triton / CuTe-DSL / FlyDSL, none of which are
    installed on the CPU CI lane -- so this reads the source instead.
    """
    path = _PACKAGE_DIR / (module_path.replace(".", "/") + ".py")
    if not path.exists():
        path = _PACKAGE_DIR / module_path.replace(".", "/") / "__init__.py"
    assert path.exists(), f"{PACKAGE}.{module_path} does not exist"

    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.If, ast.Try)):
            # Platform-conditional rebinds (``x = select_impl(...)``) and
            # guarded defs still bind a public name.
            for inner in ast.walk(node):
                if isinstance(inner, (ast.FunctionDef, ast.ClassDef)):
                    names.add(inner.name)
                elif isinstance(inner, ast.Assign):
                    names.update(t.id for t in inner.targets if isinstance(t, ast.Name))
    return names


def test_every_export_resolves_to_a_real_symbol():
    missing = [
        f"{symbol} -> {module}"
        for symbol, module in sorted(_EXPORTS.items())
        if symbol not in _module_defines(module)
    ]
    assert not missing, f"stale _EXPORTS entries: {missing}"


def test_every_symbol_imported_from_the_facade_is_exported():
    """The reverse of the check above, and the one that actually bites.

    A missing ``_EXPORTS`` entry raises ``ImportError`` at module import, so a
    module-level ``from ...diffusion import x`` fails loudly.  A *function-local*
    one -- the pattern used for optional backends -- fails only when that test
    or code path runs, on the platform that has the backend.  Enumerating the
    call sites catches it here instead.
    """
    unexported = set()
    for root in ("python/sglang", "test", "benchmark"):
        root_dir = _REPO_ROOT / root
        if not root_dir.exists():
            continue
        for path in root_dir.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel.startswith("python/sglang/kernels/ops/diffusion/"):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == PACKAGE:
                    unexported.update(
                        a.name
                        for a in node.names
                        if a.name not in _EXPORTS and not a.name.startswith("_")
                    )
    assert not unexported, f"imported but not in _EXPORTS: {sorted(unexported)}"


def test_every_registered_spec_target_resolves():
    missing = []
    for _op, _backend, target, _caps, _description in _SPECS:
        module, _, attr = target.partition(":")
        if attr not in _module_defines(module):
            missing.append(target)
    assert not missing, f"stale _SPECS targets: {missing}"


def test_registry_holds_the_diffusion_ops():
    # Registration happens at package import, is metadata-only, and is what
    # ``select_kernel`` / the tracing tools read.
    registered = {op for op in registry.ops() if op.startswith("diffusion.")}
    assert {op for op, *_ in _SPECS} <= registered


def test_facade_rejects_unknown_attributes():
    module = sys.modules[PACKAGE]
    with pytest.raises(AttributeError):
        module.definitely_not_a_kernel
    assert set(module.__all__) == set(_EXPORTS)
    assert set(_EXPORTS) <= set(dir(module))


def test_importing_the_package_does_not_import_any_leaf_module():
    """The reason ``__getattr__`` is lazy rather than a block of re-exports.

    The backends have disjoint, heavy, mutually-exclusive dependencies --
    Triton (CUDA/ROCm), CUTLASS/CuTe-DSL, and FlyDSL (gfx950).  If
    ``_EXPORTS`` ever degrades into eager ``from .norm.x import y`` lines, all
    of them become import-time requirements on every platform, which is how a
    CPU-only or Apple install starts failing at ``import sglang``.

    Asserted on this package's own leaf modules rather than on ``triton`` in
    ``sys.modules``: sibling operator groups import Triton for their own
    reasons, so a global check would not isolate this package's behavior.
    Run in a fresh interpreter because this process has already resolved
    exports through the facade.
    """
    code = (
        "import importlib, sys\n"
        f"importlib.import_module('{PACKAGE}')\n"
        f"prefix = '{PACKAGE}.'\n"
        "leaves = [m for m in sys.modules if m.startswith(prefix)"
        " and not m.endswith('__init__')]\n"
        "print(','.join(sorted(m for m in leaves if '.' in m[len(prefix):]"
        " or sys.modules[m].__file__ and not sys.modules[m].__file__"
        ".endswith('__init__.py'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    leaked = [m for m in result.stdout.strip().split(",") if m]
    assert not leaked, f"importing {PACKAGE} eagerly imported: {leaked}"


@pytest.mark.parametrize("root", ["python/sglang", "test", "benchmark"])
def test_runtime_code_imports_only_through_the_facade(root):
    root_dir = _REPO_ROOT / root
    if not root_dir.exists():  # source checkouts only
        pytest.skip(f"{root} not present in this install")

    offenders = []
    for path in root_dir.rglob("*.py"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel.startswith("python/sglang/kernels/ops/diffusion/"):
            continue  # intra-package imports are the point of the subpackages
        if rel in _DEEP_IMPORT_ALLOWLIST:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith(f"{PACKAGE}.")
            ):
                offenders.append(f"{rel}:{node.lineno} imports {node.module}")
            elif isinstance(node, ast.Import):
                offenders.extend(
                    f"{rel}:{node.lineno} imports {a.name}"
                    for a in node.names
                    if a.name.startswith(f"{PACKAGE}.")
                )
    assert not offenders, (
        "import from sglang.kernels.ops.diffusion instead of a submodule:\n  "
        + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
