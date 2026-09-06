"""No module-scope name may freeze a platform fact.

The address exists so `override_platform(...)` reaches every reader at once.
A module-level `_is_sm120 = get_platform().is_sm120` defeats that completely:
the value is read when the module is first imported and never again, so whether
an override is visible depends on import order -- and the line *looks* like it
went through the address, which is worse than the bare probe it replaced.

Four of these were written during this refactor's own conversion (three in
`fp8_utils`, one in `deepseek_v4_backend`), by substituting the accessor into
lines that were already frozen. Substituting the call is not the conversion; the
conversion is the reader asking at the point of decision.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import ast
import pathlib
import unittest

import sglang
from sglang.test.test_utils import CustomTestCase

_ROOT = pathlib.Path(next(iter(sglang.__path__))) / "srt"


def _frozen_platform_reads():
    """(file, line, name) for each module-scope `x = get_platform().y`."""
    found = []
    for path in sorted(_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        if "get_platform" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        # Module scope only: inside a function the call runs per invocation,
        # which is the shape the address is for.
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Call)
                and getattr(value.value.func, "id", None) == "get_platform"
            ):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    rel = path.relative_to(_ROOT).as_posix()
                    found.append(f"{rel}:{node.lineno} {target.id}")
    return found


class TestPlatformAddressNotFrozen(CustomTestCase):
    def test_the_scan_reaches_the_address(self):
        """The premise: `get_platform()` is used somewhere under srt/."""
        users = [
            path
            for path in _ROOT.rglob("*.py")
            if "get_platform()" in path.read_text(encoding="utf-8-sig")
        ]
        self.assertGreater(len(users), 20, "the scan found almost no readers")

    def test_no_module_scope_name_freezes_a_platform_fact(self):
        frozen = _frozen_platform_reads()
        self.assertEqual(
            [],
            frozen,
            "these read a platform fact once at import and keep the answer, so "
            "`override_platform(...)` cannot reach them and the result depends "
            f"on import order. Ask at the point of decision instead: {frozen}",
        )


if __name__ == "__main__":
    unittest.main()
