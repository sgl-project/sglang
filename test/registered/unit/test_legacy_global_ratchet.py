"""Ratchet guard: legacy global-accessor call-sites may only decrease.

The process-wide ``ServerArgs`` is owned by the runtime context; the legacy
``get_global_server_args`` / ``set_global_server_args_for_*`` names survive as
thin shims for the existing call-sites. New code should use the
``sglang.srt.runtime_context`` accessors (``get_server_args()`` /
``get_context().set_server_args()``), so the shim call-site counts below must
never grow. When your change removes call-sites, lower the matching baseline
to the new count.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import re
import unittest
from pathlib import Path

import sglang.srt
from sglang.test.test_utils import CustomTestCase

_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))

# Baselines counted over python/sglang/srt/**/*.py, including each function's
# own def line. Ratchet: decrease-only.
_RATCHETS = [
    ("get_global_server_args", r"\bget_global_server_args\s*\(", 279),
    (
        "set_global_server_args_for_*",
        r"\bset_global_server_args_for_(?:scheduler|tokenizer)\s*\(",
        5,
    ),
]


class TestLegacyGlobalRatchet(CustomTestCase):
    def test_legacy_accessor_call_sites_match_the_baselines(self):
        # Exact pin, failing in BOTH directions: a grown count means new code
        # bypassed the runtime_context accessors; a shrunk count means a
        # removal forgot to lower the baseline, which would let later changes
        # silently re-add call-sites up to the stale ceiling.
        sources = [
            path.read_text(encoding="utf-8", errors="replace")
            for path in sorted(_SRT_ROOT.rglob("*.py"))
        ]
        for name, pattern, baseline in _RATCHETS:
            count = sum(len(re.findall(pattern, source)) for source in sources)
            if count > baseline:
                self.fail(
                    f"{name} call-sites grew: {count} > baseline {baseline}. "
                    "New code must use the sglang.srt.runtime_context accessors "
                    "(get_server_args() / get_context().set_server_args())."
                )
            if count < baseline:
                self.fail(
                    f"{name} call-sites shrank: {count} < baseline {baseline}. "
                    "Lower the baseline in this file to lock in the progress."
                )


if __name__ == "__main__":
    unittest.main()
