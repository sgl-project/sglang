#!/usr/bin/env python3
"""
Pre-commit hook: validate CI registry calls under test/registered/.

Reuses ut_parse_one_file() from ci_register.py (AST-based parsing) to match
run_suite.py's collect_tests().
"""

import ast
import glob
import importlib.util
import os
import re
import sys

# Exactly what stage=/runner_config= produces, so a legacy suite= of this shape
# is always expressible the modern way.
_MODERN_SHAPE = re.compile(r"^(.+)-test-(.+)$")

# The only CUDA family still allowed on legacy `suite=`; anything else resolves
# to a suite no workflow invokes and the test silently never runs.
_LEGACY_CUDA_PREFIXES = ("stress",)

_KERNEL_LAYOUT = "test/registered/kernels/{ops,benchmark}/<group>/"


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


def _contains_call(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
        for node in ast.walk(tree)
    )


def taxonomy_errors(path: str, tree: ast.AST) -> list[str]:
    parts = path.split("/")
    if parts[:2] != ["test", "registered"] or len(parts) < 3:
        return []
    relative_parts = parts[2:]
    root = relative_parts[0]

    if root in ("kernel", "kernels"):
        canonical = (
            root == "kernels"
            and len(relative_parts) >= 4
            and relative_parts[1] in ("ops", "benchmark")
        )
        return (
            [] if canonical else [f"{path}: kernel tests live under {_KERNEL_LAYOUT}"]
        )

    if root != "unit":
        return []
    if _contains_call(tree, "popen_launch_server"):
        return [f"{path}: unit tests may not launch a server"]
    return []


def declared_amd_pr_suites() -> set[str]:
    """Read the dispatch catalog without importing the GPU test harness."""
    with open("test/run_suite.py", encoding="utf-8") as source:
        tree = ast.parse(source.read())
    for stmt in tree.body:
        if not (
            isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "PER_COMMIT_SUITES"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Dict)
        ):
            continue
        for key, suites in zip(stmt.value.keys, stmt.value.values):
            if isinstance(key, ast.Attribute) and key.attr == "AMD":
                return set(ast.literal_eval(suites))
    raise ValueError("PER_COMMIT_SUITES must declare AMD PR suites")


def main() -> int:
    # Import ci_register directly to avoid pulling in all of sglang
    spec = importlib.util.spec_from_file_location(
        "ci_register",
        os.path.join("python", "sglang", "test", "ci", "ci_register.py"),
    )
    ci_register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci_register)
    cuda = ci_register.HWBackend.CUDA
    amd = ci_register.HWBackend.AMD
    amd_pr_suites = declared_amd_pr_suites()

    # Same exclusion as run_suite.py: pytest+package structure files.
    files = sorted(
        f
        for f in glob.glob("test/registered/**/*.py", recursive=True)
        if os.path.basename(f) not in ("conftest.py", "__init__.py")
    )
    if not files:
        return 0

    missing = []
    legacy_shape = []  # (file, suite, stage, runner_config) -- has a -test- split
    non_dispatchable = []  # (file, suite) -- legacy CUDA suite no workflow invokes
    dead_tests = []  # (file) -- TestCase classes that `python3 file.py` never runs
    taxonomy_violations = []
    invalid_amd_pr = []
    for f in files:
        try:
            registries, _has_main_entry = ci_register.ut_parse_one_file(f)
        except Exception:
            continue
        if len(registries) == 0:
            missing.append(f)
            continue
        # TestCase classes are dead unless __main__ runs them; CI runs the
        # registered file as `python3 file.py`.
        with open(f, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=f)
        taxonomy_violations.extend(taxonomy_errors(f, tree))
        if _defines_testcase(tree) and not _main_runs_tests(tree):
            dead_tests.append(f)
        for r in registries:
            if (
                r.backend == amd
                and not r.nightly
                and r.effective_suite not in amd_pr_suites
            ):
                invalid_amd_pr.append((f, r.effective_suite))
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
    if invalid_amd_pr:
        print("ERROR: AMD PR registrations name undeclared suites:")
        for f, suite in invalid_amd_pr:
            print(f"  {f}: {suite}")
        print("  Choose an AMD suite from PER_COMMIT_SUITES in test/run_suite.py.\n")
        exit_code = 1
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
            print(f'  {f}\n    suite="{suite}"  ->  stage="...", runner_config="..."')
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
    if taxonomy_violations:
        print("ERROR: Registered-test taxonomy violations:")
        for error in taxonomy_violations:
            print(f"  {error}")
        print()
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
