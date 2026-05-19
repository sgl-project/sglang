"""End-to-end canary tests on Qwen3-0.6B for the periodic sweep path.

Two paired scenarios:

- ``TestCanaryRealDataSweepClean``: ``--kv-cache-canary=raise`` with
  ``--kv-cache-canary-real-data=bit`` and sweep every 5 steps, no fault
  injection. Server must stay healthy under a parallel request burst.
- ``TestCanaryRealDataSweepPerturbed``: same config plus
  ``SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB=0.01``. The sweep path must
  fire — either the launch warmup catches it, or live traffic does.
"""

from __future__ import annotations

import os
import time
import unittest
from typing import ClassVar, Dict, List, Optional

import requests

from sglang.test.canary_e2e_base import CanaryE2EBase
from sglang.test.ci.ci_register import register_cuda_ci

_MODEL = "Qwen/Qwen3-0.6B"

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-small")


class _CanaryRealDataSweepBase(CanaryE2EBase):
    """Common config: canary=raise + real-data=bit + sweep every 5 steps.

    The base class already injects ``--kv-cache-canary raise``, so we
    only layer the real-data + sweep flags via ``extra_server_args``.
    Subclasses optionally set ``server_env`` to inject extra env vars
    (e.g. the real-KV perturb knobs) into the server subprocess.
    """

    model = _MODEL
    extra_server_args = [
        "--kv-cache-canary-real-data",
        "bit",
        "--kv-cache-canary-real-data-sweep-every-n-steps",
        "5",
    ]
    server_env: ClassVar[Dict[str, str]] = {}

    _saved_env: ClassVar[Optional[Dict[str, Optional[str]]]] = None

    @classmethod
    def setUpClass(cls) -> None:
        # Inject server_env into os.environ before the base class snapshots
        # it into the server subprocess; restore in tearDownClass.
        cls._saved_env = {k: os.environ.get(k) for k in cls.server_env}
        for k, v in cls.server_env.items():
            os.environ[k] = v
        try:
            super().setUpClass()
        except Exception:
            cls._restore_env()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            super().tearDownClass()
        finally:
            cls._restore_env()

    @classmethod
    def _restore_env(cls) -> None:
        if cls._saved_env is None:
            return
        for k, old in cls._saved_env.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        cls._saved_env = None


class TestCanaryRealDataSweepClean(_CanaryRealDataSweepBase):
    """Sweep ON + no perturb: clean burst, no violations expected."""

    def test_clean_burst_no_violation(self) -> None:
        results = self.send_parallel_requests(n=50)
        bad = [r for r in results if r.get("status_code") != 200]
        self.assertFalse(bad, f"non-200 responses on clean sweep run: {bad[:3]}")

        # Allow the side-stream event pump a beat to refresh counters.
        time.sleep(2.0)

        self.assert_health_ok()


class TestCanaryRealDataSweepPerturbed(_CanaryRealDataSweepBase):
    """Sweep ON + real-KV byte perturb: sweep path must raise.

    Perturb flips one real-KV byte on an alive-but-not-this-step-verified
    slot, which only the periodic sweep can catch. Outcome: either the
    server fails to come up (warmup sweep tripped) OR the burst returns
    errors / /health flips to non-200.
    """

    server_env = {
        "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB": "0.01",
        "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED": "42",
    }
    allow_launch_failure = True

    def test_perturbed_burst_triggers_sweep_raise(self) -> None:
        if not self.launch_failed:
            results: List[Dict[str, object]] = self.send_parallel_requests(
                n=200, max_new_tokens=32, timeout=30.0
            )
            triggered = any(
                "error" in r or int(r.get("status_code", 0)) >= 500 for r in results
            )

            time.sleep(1.5)
            try:
                health_ok = (
                    requests.get(self.base_url + "/health", timeout=5).status_code
                    == 200
                )
            except requests.exceptions.RequestException:
                health_ok = False

            self.assertTrue(
                triggered or not health_ok,
                f"Expected sweep to fire under perturb+raise, but server still "
                f"healthy and no failed requests; first 3 responses: {results[:3]}",
            )
        # Hard-assert the SWEEP path caught it. The perturb hook targets
        # alive slots NOT in this step's verify list, so the per-step path
        # cannot observe the byte flip -- only the periodic sweep can.
        self.assert_violation_kind_logged(["sweep_k", "sweep_v"])


if __name__ == "__main__":
    unittest.main(verbosity=3)
