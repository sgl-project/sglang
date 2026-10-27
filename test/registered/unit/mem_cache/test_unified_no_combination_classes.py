"""The unified memory pool keeps no class per combination of KV kinds.

A plan's sub-pools are routed by one `RoutedKVCache` and allocated by one
`ChainAllocator`, whatever set of kinds the model declares. A class named for a
combination (a window with full attention, recurrent state with a window, ...)
would bring back one code path per combination. The classes listed below still
name one and are on their way out.
"""

import ast
import pathlib
import re
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_COMBINATION = re.compile(r"^Unified(Draft)?(SWA|MambaSWA|SWAMamba|Hybrid)\w*")

# Classes that name a combination and still exist. None is left.
_ALLOWED: set = set()


def _combination_classes():
    srt = pathlib.Path(sglang.__file__).parent / "srt"
    found = set()
    for path in srt.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and _COMBINATION.match(node.name):
                found.add(node.name)
    return found


class TestNoCombinationClasses(CustomTestCase):
    def test_no_new_combination_class(self):
        self.assertEqual(_combination_classes() - _ALLOWED, set())

    def test_allowed_classes_still_exist(self):
        # An entry whose class is gone is removed from the list with it.
        self.assertEqual(_ALLOWED - _combination_classes(), set())


if __name__ == "__main__":
    unittest.main()
