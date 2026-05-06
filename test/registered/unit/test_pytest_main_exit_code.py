"""Guards the CI runner contract that test files using pytest.main propagate
the exit code so failing tests cause the wrapping python process to exit
non-zero. ci_utils.run_files reads process.returncode; a bare
pytest.main([__file__]) silently lets the script exit 0 even when assertions
fail, which masks regressions.
"""

import ast
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCAN_ROOTS = [_REPO_ROOT / "python", _REPO_ROOT / "test"]


class TestPytestMainExitCode(CustomTestCase):
    def test_wrapper_propagates_pytest_failure(self):
        """A file using `sys.exit(pytest.main([__file__]))` must exit non-zero
        when the contained pytest assertions fail."""
        script = textwrap.dedent("""
            import sys
            import pytest

            def test_intentional_failure():
                assert False, "intentional failure for exit-code propagation contract"

            if __name__ == "__main__":
                sys.exit(pytest.main([__file__]))
            """).strip()

        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "fake_failing_test.py"
            path.write_text(script)
            result = subprocess.run(
                [sys.executable, str(path), "-f"],
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(
            result.returncode,
            0,
            msg=(
                "sys.exit(pytest.main(...)) must propagate failing tests as "
                f"non-zero exit code; got 0.\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
        )

    def test_no_bare_pytest_main_in_repo(self):
        """Every `if __name__ == '__main__':` block that calls pytest.main
        must wrap it in sys.exit(...). A bare expression statement leaves the
        script's exit code at 0, so the CI runner reports the file as PASSED
        even when pytest reports failures."""
        offenders = []
        for root in _SCAN_ROOTS:
            if not root.exists():
                continue
            for path in root.rglob("*.py"):
                violation = _find_bare_pytest_main(path)
                if violation is not None:
                    offenders.append(violation)

        self.assertFalse(
            offenders,
            msg=(
                "Found bare `pytest.main(...)` in __main__ blocks (must be "
                "wrapped in sys.exit(...) so failing tests propagate the exit "
                "code to the CI runner):\n  " + "\n  ".join(offenders)
            ),
        )


def _find_bare_pytest_main(path: pathlib.Path):
    """Return `<rel_path>:<lineno>` if `path` has a bare pytest.main(...) call
    inside `if __name__ == "__main__":`, else None."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not _is_main_guard(node.test):
            continue
        for stmt in node.body:
            if _is_bare_pytest_main_call(stmt):
                rel = path.relative_to(_REPO_ROOT)
                return f"{rel}:{stmt.lineno}"
    return None


def _is_main_guard(test: ast.expr) -> bool:
    """Match `__name__ == "__main__"` (either side)."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    sides = [test.left, *test.comparators]
    has_name = any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
    has_main = any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides)
    return has_name and has_main


def _is_bare_pytest_main_call(stmt: ast.stmt) -> bool:
    """Match `pytest.main(...)` as a bare expression statement (i.e. its
    return value is discarded). `sys.exit(pytest.main(...))` and
    `code = pytest.main(...)` are fine."""
    if not isinstance(stmt, ast.Expr):
        return False
    call = stmt.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "main"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


if __name__ == "__main__":
    unittest.main()
