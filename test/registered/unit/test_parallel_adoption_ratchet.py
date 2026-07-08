"""Ratchet guard: legacy parallel-getter calls in swept directories may only
shrink.

``models/`` and ``layers/`` read parallel topology through
``get_parallel().<dim>`` (the read-through wrapper in ``runtime_context``),
which gives one import, one naming scheme, and the scoped ``override()``
test primitive. Direct calls to the ``parallel_state`` size/rank getters in
these directories are regressions against that sweep.

Exemptions, pinned by path: ``layers/dp_attention.py`` is delegation
substrate (the wrapper's attn-DP dims delegate TO it), and ``layers/dcp/``
is the DCP subsystem's own plumbing, booked for a follow-up sweep. Sweeping
an exempt path must remove it from the pin.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import re
import unittest
from pathlib import Path

import sglang.srt
from sglang.test.test_utils import CustomTestCase

_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))

_BANNED_CALLS = re.compile(
    r"\bget_(?:"
    r"tensor_model_parallel_(?:world_size|rank)"
    r"|pipeline_model_parallel_(?:world_size|rank)"
    r"|moe_expert_parallel_(?:world_size|rank)"
    r"|moe_tensor_parallel_(?:world_size|rank)"
    r"|moe_data_parallel_(?:world_size|rank)"
    r"|attn_tensor_model_parallel_(?:world_size|rank)"
    r"|attn_context_model_parallel_(?:world_size|rank)"
    r"|dcp_(?:world_size|rank)"
    r")\(\)"
)

_SWEPT_DIRS = ("models", "layers")

_EXEMPT = (
    "layers/dp_attention.py",  # delegation substrate for the attn-DP dims
    "layers/dcp/",  # DCP subsystem plumbing; follow-up sweep
)


class TestParallelAdoptionRatchet(CustomTestCase):
    def test_no_legacy_parallel_getters_in_swept_dirs(self):
        offenders = []
        for top in _SWEPT_DIRS:
            for path in sorted((_SRT_ROOT / top).rglob("*.py")):
                rel = path.relative_to(_SRT_ROOT).as_posix()
                if rel.startswith(_EXEMPT):
                    continue
                for i, line in enumerate(path.read_text().split("\n"), 1):
                    if _BANNED_CALLS.search(line):
                        offenders.append(f"{rel}:{i}")
        self.assertFalse(
            offenders,
            "legacy parallel-getter calls in swept directories (use "
            f"get_parallel().<dim> instead): {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
