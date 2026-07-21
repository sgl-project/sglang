"""Ratchet guard: server_args mutations outside the resolution pipeline may
only decrease.

After ``ServerArgs.__post_init__`` returns, the instance carries the resolved
configuration; the resolution pipeline (``server_args.py`` and
``arg_groups/``) is the only place that computes it. Every assignment to a
``server_args`` field elsewhere weakens that contract, so the count below is
an exact pin: new mutations must not appear, and removals must lower the
baseline to lock in the progress.

Every audited runtime adjustment goes through ``ServerArgs.override(source,
**fields)`` — the single mutation entry point, which records provenance and
keeps whitelisted fields consistent with the declaration stash. The baseline
is therefore zero. The registered test harness additionally runs with
``SGLANG_STRICT_CONFIG_MUTATION=1``, under which a bare assignment after
resolution raises at runtime; this ratchet catches sites the tests never
execute.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

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
                "(passes / declare_load_time_override) or go through "
                "ServerArgs.override(source, ...) instead of assigning fields."
            )
        if count < _BASELINE:
            self.fail(
                f"server_args mutations outside the resolution pipeline "
                f"shrank: {count} < baseline {_BASELINE}. Lower the baseline "
                "in this file to lock in the progress."
            )


if __name__ == "__main__":
    unittest.main()
