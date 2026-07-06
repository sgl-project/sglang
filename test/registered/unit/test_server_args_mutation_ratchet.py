"""Ratchet guard: server_args mutations outside the resolution pipeline may
only decrease.

After ``ServerArgs.__post_init__`` returns, the instance carries the resolved
configuration; the resolution pipeline (``server_args.py`` and
``arg_groups/``) is the only place that computes it. Every assignment to a
``server_args`` field elsewhere weakens that contract, so the count below is
an exact pin: new mutations must not appear, and removals must lower the
baseline to lock in the progress.

The remaining call-sites fall into three audited families:

- **Load-/runtime-resolved values** that cannot be decided at
  ``__post_init__`` (weight-resolved kv-cache dtype for unpublished mock
  runners, memory-budget mamba cache sizing, draft-worker copies deriving
  ``context_length`` from the loaded model, grammar-backend import fallback).
  Whitelisted resolvable fields must go through
  ``declare_load_time_override`` / ``record_runtime_overrides`` instead of a
  bare assignment.
- **Control-plane reconfiguration** at runtime (hicache attach/detach,
  weight updates rewriting ``model_path`` / ``load_format`` /
  ``weight_version``, adaptive speculative-decoding retuning).
- **Deployment wiring** computed per process or per rank (ray placement
  groups, metrics IPC endpoints, the multi-tokenizer disaggregation-mode
  shuffle).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import re
import unittest
from pathlib import Path

import sglang.srt
from sglang.test.test_utils import CustomTestCase

_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))

# Assignments to a server_args attribute (``server_args.x = ...``,
# ``self.server_args.x = ...``, and the ``sa`` alias used by a few helpers).
# ``==`` comparisons are excluded by the negative lookahead.
_MUTATION_PATTERNS = [
    re.compile(r"\bserver_args\.[a-z0-9_]+\s*=(?!=)"),
    re.compile(r"\bsa\.[a-z0-9_]+\s*=(?!=)"),
]

# The resolution pipeline itself: mutation is its job.
_PIPELINE = ("server_args.py", "arg_groups")

_BASELINE = 45


class TestServerArgsMutationRatchet(CustomTestCase):
    def test_out_of_pipeline_mutations_match_the_baseline(self):
        count = 0
        for path in sorted(_SRT_ROOT.rglob("*.py")):
            rel = path.relative_to(_SRT_ROOT)
            if rel.parts[0] in _PIPELINE:
                continue
            source = path.read_text()
            count += sum(len(p.findall(source)) for p in _MUTATION_PATTERNS)
        if count > _BASELINE:
            self.fail(
                f"server_args mutations outside the resolution pipeline grew: "
                f"{count} > baseline {_BASELINE}. Configuration is resolved in "
                "ServerArgs.__post_init__; declare through the pipeline "
                "(passes / declare_load_time_override / "
                "record_runtime_overrides) instead of assigning fields."
            )
        if count < _BASELINE:
            self.fail(
                f"server_args mutations outside the resolution pipeline "
                f"shrank: {count} < baseline {_BASELINE}. Lower the baseline "
                "in this file to lock in the progress."
            )


if __name__ == "__main__":
    unittest.main()
