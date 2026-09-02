"""A function does not take the record it never reads.

A `server_args` parameter that the body never names keeps a reference to the
whole record alive across a call boundary, and it reads as an invitation: the
next person to need one value takes it off the parameter that is already there,
instead of deciding where that value should come from. Removing one usually
uncovers the next -- the caller that only had a record to pass it along.

Class methods are exempt: a base class, an override, or one implementation of a
strategy carries the parameter for its contract, and the body of any single one
of them is not evidence. This walks module-level functions only.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_PACKAGE_ROOT = pathlib.Path(next(iter(sglang.__path__)))

# The resolution pipeline builds the record, so a parameter there is the subject
# rather than a passenger. `multimodal_gen` has a different, same-named class
# outside this contract, as the mutation ratchet also records.
_EXCLUDED = ("srt/arg_groups", "srt/server_args.py", "multimodal_gen")

_BASELINE = 0


def _dead_parameters():
    found = []
    scanned = 0
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        if rel.startswith(_EXCLUDED):
            continue
        source = path.read_text(encoding="utf-8-sig")
        if "server_args" not in source:
            continue
        scanned += 1
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            taken = [a.arg for a in node.args.args] + [
                a.arg for a in node.args.kwonlyargs
            ]
            if "server_args" not in taken:
                continue
            named = any(
                isinstance(inner, ast.Name) and inner.id == "server_args"
                for inner in ast.walk(node)
                if inner is not node
            )
            if not named:
                found.append(f"{rel}:{node.lineno} {node.name}")
    return found, scanned


class TestNoDeadServerArgsParameter(CustomTestCase):
    def test_no_module_level_function_takes_a_record_it_ignores(self):
        found, scanned = _dead_parameters()
        self.assertGreater(
            scanned,
            50,
            f"only {scanned} files mention server_args; the scan is broken, not "
            "the tree",
        )
        self.assertEqual(
            _BASELINE,
            len(found),
            "these functions take `server_args` and never name it; drop the "
            "parameter and the argument at every call site, then check whether "
            f"the caller still needs its own: {found}",
        )


if __name__ == "__main__":
    unittest.main()
