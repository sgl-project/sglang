"""Regression guard for the `SGLANG_ENABLE_MINIMAL_INIT` fast path.

`test/run_suite.py` sets `SGLANG_ENABLE_MINIMAL_INIT=1` so the orchestrator's
`import sglang` skips the heavy public-API init (torch / transformers /
frontend language). The whole point is that the minimal import chain
(`sglang/__init__.py` -> `sglang.srt.environ`, `sglang.version`, and the
`sglang.test.ci.*` modules run_suite imports) never pulls in torch — that is
what turns a multi-second orchestrator startup into ~10 ms on a contended CI
host.

Nothing else enforces that invariant: a stray top-level `import torch` (or an
import of a module that transitively loads it) added above the guard in
`__init__.py`, or into `environ` / `ci_register` / `ci_utils`, would silently
revert the speedup with no other test failing. These probes lock it in.

Each probe runs in a FRESH interpreter subprocess: `sys.modules` is
process-global and this test process has already imported torch (via
CustomTestCase), so "did importing sglang load torch?" is only answerable in a
clean process.
"""

import json
import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# Probe: import sglang in a clean interpreter and report what happened.
_PROBE = (
    "import sys, json, sglang; "
    "print(json.dumps({"
    "'version': bool(getattr(sglang, '__version__', '')), "
    "'torch_loaded': 'torch' in sys.modules, "
    "'has_server_args': hasattr(sglang, 'ServerArgs'), "
    "}))"
)


def _run_probe(minimal: bool) -> dict:
    env = dict(os.environ)
    if minimal:
        env["SGLANG_ENABLE_MINIMAL_INIT"] = "1"
    else:
        env.pop("SGLANG_ENABLE_MINIMAL_INIT", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"probe (minimal={minimal}) exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestMinimalInit(CustomTestCase):
    def test_minimal_init_skips_torch(self):
        r = _run_probe(minimal=True)
        # __version__ must survive the guard — run_suite reads it.
        self.assertTrue(r["version"], "__version__ missing under minimal init")
        # The load-bearing assertion: the minimal import chain must not pull
        # in torch. If this fails, some import above the guard (or in environ /
        # ci_register / ci_utils) started dragging torch back in.
        self.assertFalse(
            r["torch_loaded"],
            "minimal init imported torch — the run_suite fast path is defeated",
        )
        # Proves the flag actually took the minimal branch rather than falling
        # through to full init.
        self.assertFalse(
            r["has_server_args"],
            "ServerArgs present under minimal init — the guard did not engage",
        )

    def test_full_init_unchanged(self):
        # Default (flag unset) init still exposes the public API. Guards against
        # an over-eager edit that leaves real imports gated behind the flag.
        r = _run_probe(minimal=False)
        self.assertTrue(r["version"], "__version__ missing under full init")
        self.assertTrue(r["has_server_args"], "ServerArgs missing under full init")


if __name__ == "__main__":
    unittest.main()
