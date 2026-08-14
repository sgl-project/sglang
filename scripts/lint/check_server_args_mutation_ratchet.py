"""Guard: no server_args mutation outside the resolution pipeline, pinned at 0.

``ServerArgs.__setattr__`` already raises on a bare assignment after
resolution; this static scan is what reaches the sites tests never execute.
"""

import re
from pathlib import Path

_SGLANG_ROOT = Path(__file__).resolve().parents[2] / "python" / "sglang"

# Assignments to a server_args attribute (``server_args.x = ...``,
# ``self.server_args.x = ...``, and the ``sa`` alias used by a few helpers).
# ``==`` comparisons are excluded by the negative lookahead.
_MUTATION_PATTERNS = [
    # (?![=}]) skips ``==`` comparisons and f-string ``{x=}`` debug specs.
    re.compile(r"\bserver_args\.[a-z0-9_]+\s*=(?![=}])"),
    re.compile(r"\bsa\.[a-z0-9_]+\s*=(?![=}])"),
    re.compile(r"get_(?:global_)?server_args\(\)\.[a-z0-9_]+\s*=(?![=}])"),
    # setattr is the same write with the attribute name behind a variable.
    re.compile(
        r"setattr\(\s*(?:[\w.]+\.)?(?:server_args|sa|get_(?:global_)?server_args\(\))\s*,"
    ),
]

# The resolution pipeline itself (mutation is its job) and multimodal_gen,
# whose ServerArgs is a different class outside this contract.
_EXCLUDED = (
    "srt/server_args.py",
    "srt/arg_groups",
    "multimodal_gen",
)

_BASELINE = 0


def check_server_args_mutation_ratchet():
    count = 0
    for path in sorted(_SGLANG_ROOT.rglob("*.py")):
        rel = path.relative_to(_SGLANG_ROOT).as_posix()
        if rel.startswith(_EXCLUDED):
            continue
        source = path.read_text()
        count += sum(len(pattern.findall(source)) for pattern in _MUTATION_PATTERNS)
    if count > _BASELINE:
        raise AssertionError(
            f"server_args mutations outside the resolution pipeline grew: "
            f"{count} > baseline {_BASELINE}. Configuration is resolved in "
            "ServerArgs.__post_init__; declare through the pipeline "
            "(passes / declare_late_resolution), change resolved config "
            "with get_context().override(source, ...), or hand the value "
            "to its runner as a constructor argument — do not assign fields."
        )


if __name__ == "__main__":
    check_server_args_mutation_ratchet()
