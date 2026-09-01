"""Ratchet guard: module-level runtime state in the flag-owning layers may
only shrink.

Runtime flags belong on ``get_flags()`` groups, which have lifecycle reset and
a scoped test-override primitive; a module-level ``global`` has neither and
leaks across test teardowns. ``_PINNED_GLOBALS`` names the survivors --
migrating one must shrink it, adding one fails the ratchet.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path

import sglang.srt
from sglang.test.test_utils import CustomTestCase

_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))

_PINNED_GLOBALS = {
    "layers/moe/utils.py": frozenset(),
    "layers/dp_attention.py": frozenset(
        {
            # DP-attention topology (parallel vertical scope).
            "_ATTN_DP_RANK",
            "_ATTN_DP_SIZE",
        }
    ),
}


class TestModuleStateRatchet(CustomTestCase):
    def test_global_statements_match_the_pins(self):
        for rel, pinned in _PINNED_GLOBALS.items():
            tree = ast.parse((_SRT_ROOT / rel).read_text())
            declared = {
                name
                for node in ast.walk(tree)
                if isinstance(node, ast.Global)
                for name in node.names
            }
            grown = declared - pinned
            self.assertFalse(
                grown,
                f"{rel} declares new module-level runtime state {sorted(grown)}; "
                "put runtime flags on a get_flags() group instead "
                "(see runtime_context.MoeFlags / DpFlags).",
            )
            shrunk = pinned - declared
            self.assertFalse(
                shrunk,
                f"{rel} no longer declares {sorted(shrunk)}; "
                "shrink the pin in this file to lock in the progress.",
            )


if __name__ == "__main__":
    unittest.main()
