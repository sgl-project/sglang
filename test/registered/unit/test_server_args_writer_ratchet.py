"""Ratchet guard: ``ServerArgs.override`` call-sites may only decrease.

``ServerArgs.override(source, **fields)`` mutates a ``ServerArgs`` *instance*
only — the resolved-config bags on the runtime context never see the write, so
any consumer reading the namespace accessors (``get_exec()`` / ``get_memory()``
/ …) desyncs from the writer. The migration end-state removes this primitive
entirely: post-publish, process-global config changes go through
``get_context().override(source, **fields)`` (which writes the bags), and
per-runner resolved values live on the runner object rather than on a
``ServerArgs`` copy.

Until every call-site is rerouted together with its readers, this exact pin
keeps the writer surface from growing unwatched: new writers must use
``get_context().override``, and each rerouted batch lowers the baseline to
lock in the progress. (The count is textual and includes docstring mentions
and the test-kit's private-config use — the pin tracks growth, not the exact
production-writer census.)
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import re
import unittest
from pathlib import Path

import sglang
from sglang.test.test_utils import CustomTestCase

_SGLANG_ROOT = Path(next(iter(sglang.__path__)))

# ``server_args.override(`` also matches ``self.server_args.override(``,
# ``<obj>.server_args.override(``, and the ``draft_server_args`` /
# ``dp_server_args`` copies; ``args`` / ``sa`` are the aliases a few call-sites
# bind first.
_WRITER_PATTERNS = [
    re.compile(r"server_args\.override\("),
    re.compile(r"\bargs\.override\("),
    re.compile(r"\bsa\.override\("),
]

# The resolution pipeline itself (its declare face forwards through
# ``override`` by design) and multimodal_gen, whose ServerArgs is a different
# class outside this contract.
_EXCLUDED = (
    "srt/server_args.py",
    "srt/arg_groups",
    "multimodal_gen",
)

_BASELINE = 18


class TestServerArgsWriterRatchet(CustomTestCase):
    def test_server_args_override_call_sites_match_the_baseline(self):
        count = 0
        for path in sorted(_SGLANG_ROOT.rglob("*.py")):
            rel = path.relative_to(_SGLANG_ROOT).as_posix()
            if rel.startswith(_EXCLUDED):
                continue
            source = path.read_text()
            count += sum(len(p.findall(source)) for p in _WRITER_PATTERNS)
        if count > _BASELINE:
            self.fail(
                f"ServerArgs.override call-sites grew: {count} > baseline "
                f"{_BASELINE}. Instance writes never reach the resolved-config "
                "bags, so namespace readers desync from the writer. Post-publish "
                "process-global changes go through get_context().override(...); "
                "per-runner resolved values belong on the runner object."
            )
        if count < _BASELINE:
            self.fail(
                f"ServerArgs.override call-sites shrank: {count} < baseline "
                f"{_BASELINE}. Lower the baseline in this file to lock in the "
                "progress."
            )


if __name__ == "__main__":
    unittest.main()
