#!/usr/bin/env python3
"""
Pre-commit hook: validate CI registry calls under test/registered/.

1. Every test file must contain a CI registry call (register_cuda_ci,
   register_amd_ci, etc.).
2. A CUDA test must register its suite via the modern
   `stage=`/`runner_config=` form. The legacy single-string `suite=` is reserved
   for the stress family (and for AMD/CPU/NPU suites); any other CUDA `suite=`
   resolves to a name no workflow invokes, so the test silently never runs.
   Two shapes are rejected:
     a. `{stage}-test-{runner_config}` -- the modern name stuffed back into the
        legacy form. Reported with the exact stage/runner split to use.
     b. an older `{stage}-{runner_config}` PR-test name (e.g. the pre-migration
        `base-b-kernel-unit-1-gpu-large`) -- no longer matches any workflow
        suite at all.
   The modern form resolves to the identical suite (CIRegistry.effective_suite
   is f"{stage}-test-{runner_config}") and is /rerun-test-able.

Reuses ut_parse_one_file() from ci_register.py (AST-based parsing)
to match the same logic used by run_suite.py's collect_tests().
"""

import ast
import glob
import importlib.util
import os
import re
import sys

# Suite names of the form `{stage}-test-{runner_config}` are exactly what the
# modern stage=/runner_config= form produces, so a legacy suite= carrying this
# shape is always expressible (and should be expressed) the modern way.
_MODERN_SHAPE = re.compile(r"^(.+)-test-(.+)$")

# The only CUDA suite family still allowed on the legacy single-string `suite=`
# form. Anything else needs stage=/runner_config=, or its effective_suite matches
# no suite any workflow invokes and the test silently never runs.
_LEGACY_CUDA_PREFIXES = ("stress",)


def _defines_testcase(tree: ast.AST) -> bool:
    """True if the file defines unittest classes, statically or via type()."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any("TestCase" in ast.unparse(b) for b in node.bases):
                return True
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "type"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Tuple)
                and any("TestCase" in ast.unparse(e) for e in node.args[1].elts)
            ):
                return True
    return False


def _main_runs_tests(tree: ast.Module) -> bool:
    for stmt in tree.body:
        if not (
            isinstance(stmt, ast.If)
            and ast.unparse(stmt.test).replace("'", '"') == '__name__ == "__main__"'
        ):
            continue
        body = ast.unparse(ast.Module(body=stmt.body, type_ignores=[]))
        if "unittest.main" in body or "pytest.main" in body:
            return True
    return False


def main() -> int:
    # Import ci_register directly to avoid pulling in all of sglang
    spec = importlib.util.spec_from_file_location(
        "ci_register",
        os.path.join("python", "sglang", "test", "ci", "ci_register.py"),
    )
    ci_register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci_register)
    cuda = ci_register.HWBackend.CUDA

    # Same filter as run_suite.py: skip conftest.py, __init__.py, and utils.py
    files = sorted(
        f
        for f in glob.glob("test/registered/**/*.py", recursive=True)
        if os.path.basename(f) not in ("conftest.py", "__init__.py", "utils.py")
    )
    if not files:
        return 0

    missing = []
    legacy_shape = []  # (file, suite, stage, runner_config) -- has a -test- split
    non_dispatchable = []  # (file, suite) -- legacy CUDA suite no workflow invokes
    dead_tests = []  # (file) -- TestCase classes that `python3 file.py` never runs
    for f in files:
        try:
            registries, _has_main_entry = ci_register.ut_parse_one_file(f)
        except Exception:
            # Skip files that can't be parsed (syntax errors, etc.)
            continue
        if len(registries) == 0:
            missing.append(f)
            continue
        # TestCase classes are dead unless __main__ runs them (CI does
        # `python3 file.py`); the ERROR text below explains the fix.
        with open(f, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=f)
        if _defines_testcase(tree) and not _main_runs_tests(tree):
            dead_tests.append(f)
        for r in registries:
            # Pure legacy form on a CUDA registry: suite set, stage/runner unset.
            if not (
                r.backend == cuda
                and r.suite is not None
                and r.stage is None
                and r.runner_config is None
            ):
                continue
            if r.suite.split("-", 1)[0] in _LEGACY_CUDA_PREFIXES:
                continue
            m = _MODERN_SHAPE.match(r.suite)
            if m:
                legacy_shape.append((f, r.suite, m.group(1), m.group(2)))
            else:
                non_dispatchable.append((f, r.suite))

    exit_code = 0
    if missing:
        print("ERROR: Files in test/registered/ missing CI registry call:")
        print("  Move manual-only tests to test/manual/.\n")
        for f in missing:
            print(f"  {f}")
        print()
        exit_code = 1
    if legacy_shape:
        print(
            "ERROR: CUDA test(s) register a `{stage}-test-{runner_config}`-shaped "
            'suite via the legacy `suite="..."` form, which is not dispatchable '
            "via /rerun-test. Switch to the modern `stage=`/`runner_config=` form "
            "(same stage, same runner):\n"
        )
        for f, suite, stage, runner_config in legacy_shape:
            print(
                f"  {f}\n"
                f'    suite="{suite}"'
                f'  ->  stage="{stage}", runner_config="{runner_config}"'
            )
        print()
        exit_code = 1
    if non_dispatchable:
        print(
            'ERROR: CUDA test(s) register a legacy `suite="..."` that is neither a '
            "nightly/stress/weekly suite nor the modern `stage=`/`runner_config=` "
            "form. This name matches no suite the PR-test workflows invoke, so the "
            "test silently never runs. Switch to the modern form:\n"
        )
        for f, suite in non_dispatchable:
            print(
                f"  {f}\n"
                f'    suite="{suite}"'
                f'  ->  stage="...", runner_config="..."'
            )
        print()
        exit_code = 1
    if dead_tests:
        print(
            "ERROR: Test file(s) define TestCase classes that CI never runs: "
            "the registered file is executed as `python3 file.py`, but its "
            '`if __name__ == "__main__"` block does not call unittest.main() '
            "or pytest.main(), so the classes are silently skipped while the "
            "file reports success. Make __main__ run the tests (put any CLI "
            "entry point behind an explicit flag):\n"
        )
        for f in dead_tests:
            print(f"  {f}")
        print()
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
