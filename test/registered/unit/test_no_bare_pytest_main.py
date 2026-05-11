import ast
import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=19, suite="stage-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCAN_ROOTS = [_REPO_ROOT / "python", _REPO_ROOT / "test"]


class TestNoBarePytestMain(CustomTestCase):
    def test_no_bare_pytest_main_in_repo(self):
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
    """Match `pytest.main(...)` whose return value is discarded.
    `sys.exit(pytest.main(...))` and `code = pytest.main(...)` are fine."""
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
