"""Ratchet guard: server_args mutations outside the resolution pipeline may
only decrease.

After ``ServerArgs.__post_init__`` returns, the instance carries the resolved
configuration and the resolution pipeline (``server_args.py`` and
``arg_groups/``) is the only place that computes it: resolved config changes go
to the context bags via ``get_context().override(source, **fields)``, and a
value one runner or worker owns travels as a constructor argument. The baseline
is therefore an exact pin at zero -- new mutations must not appear, and removals
must lower it.

``ServerArgs.__setattr__`` already raises on a bare assignment after
resolution; this textual scan is what reaches the sites tests never execute.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")

import re
import unittest
from pathlib import Path

import sglang
from sglang.test.test_utils import CustomTestCase

_SGLANG_ROOT = Path(next(iter(sglang.__path__)))

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


class TestServerArgsMutationRatchet(CustomTestCase):
    def test_out_of_pipeline_mutations_match_the_baseline(self):
        count = 0
        for path in sorted(_SGLANG_ROOT.rglob("*.py")):
            rel = path.relative_to(_SGLANG_ROOT).as_posix()
            if rel.startswith(_EXCLUDED):
                continue
            source = path.read_text()
            count += sum(len(p.findall(source)) for p in _MUTATION_PATTERNS)
        if count > _BASELINE:
            self.fail(
                f"server_args mutations outside the resolution pipeline grew: "
                f"{count} > baseline {_BASELINE}. Configuration is resolved in "
                "ServerArgs.__post_init__; declare through the pipeline "
                "(passes / declare_late_resolution), change resolved config "
                "with get_context().override(source, ...), or hand the value "
                "to its runner as a constructor argument — do not assign fields."
            )
        if count < _BASELINE:
            self.fail(
                f"server_args mutations outside the resolution pipeline "
                f"shrank: {count} < baseline {_BASELINE}. Lower the baseline "
                "in this file to lock in the progress."
            )


if __name__ == "__main__":
    unittest.main()
