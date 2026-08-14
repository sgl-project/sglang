"""Ratchet guard: module-level runtime state in the flag-owning layers may
only shrink.

Runtime flags belong on ``get_flags()`` groups, which have lifecycle reset and
a scoped test-override primitive; a module-level ``global`` has neither and
leaks across test teardowns. The pin below names the survivors -- migrating one
must shrink it.
"""

import ast
from pathlib import Path

_SRT_ROOT = Path(__file__).resolve().parents[2] / "python" / "sglang" / "srt"

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


def check_module_state_ratchet():
    for rel, pinned in _PINNED_GLOBALS.items():
        tree = ast.parse((_SRT_ROOT / rel).read_text())
        declared = {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Global)
            for name in node.names
        }
        grown = declared - pinned
        if grown:
            raise AssertionError(
                f"{rel} declares new module-level runtime state {sorted(grown)}; "
                "put runtime flags on a get_flags() group instead "
                "(see runtime_context.MoeFlags / DpFlags).",
            )
        shrunk = pinned - declared
        if shrunk:
            raise AssertionError(
                f"{rel} no longer declares {sorted(shrunk)}; "
                "shrink the pin in this file to lock in the progress.",
            )


if __name__ == "__main__":
    check_module_state_ratchet()
