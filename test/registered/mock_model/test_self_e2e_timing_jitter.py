"""Timing-jitter fuzzer e2e self-tests.

Group A (no-bug-no-violation) and group B (perturb-survives-jitter) run against the standard MHA
fixture (Qwen3-0.6B with num_hidden_layers=1) — the same shape as ``test_self_e2e_mha_full.py``.
Group C (known-race speed-up) is parked as ``pytest.skip`` until SteppableEngine is implemented;
the file keeps the case so reviewers see the planned shape.
"""

from __future__ import annotations

import unittest
from typing import ClassVar, List

from sglang.test.canary_e2e_base import CanaryE2EBase
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-large")


_QWEN3_MODEL = "Qwen/Qwen3-0.6B"
_NUM_LAYERS_OVERRIDE = '{"num_hidden_layers": 1}'


class _JitterE2EBase(CanaryE2EBase):
    model: ClassVar[str] = _QWEN3_MODEL
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-jitter-enabled",
    ]


class TestJitterModerateNoCanaryViolation(_JitterE2EBase, unittest.TestCase):
    """Group A: jitter on, no perturb, moderate cycles -> no canary violation."""

    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-jitter-enabled",
        "--kv-canary-jitter-per-slot-fire-prob",
        "1.0",
        "--kv-canary-jitter-max-cycles",
        "100000",
    ]

    def test_jitter_moderate_no_violation(self) -> None:
        # Step 1: drive enough traffic so every jitter slot fires across many steps.
        results = self.send_parallel_requests(n=32, max_new_tokens=32, timeout=60.0)

        # Step 2: every request must complete successfully and the canary must stay silent.
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)
        self.assert_health_ok()


class TestJitterExtremeMaxCyclesNoViolation(_JitterE2EBase, unittest.TestCase):
    """Group A: jitter at 10M-cycle ceiling (~5ms per slot) — only slow, must not raise."""

    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-jitter-enabled",
        "--kv-canary-jitter-per-slot-fire-prob",
        "1.0",
        "--kv-canary-jitter-max-cycles",
        "10000000",
    ]

    def test_jitter_extreme_max_cycles_no_violation(self) -> None:
        # Step 1: short burst; per-step latency is ~20ms higher under this jitter setting.
        results = self.send_parallel_requests(n=8, max_new_tokens=8, timeout=120.0)

        # Step 2: requests still finish and canary stays silent.
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)
        self.assert_health_ok()


class TestJitterDoesNotSwallowPerturbSignal(_JitterE2EBase, unittest.TestCase):
    """Group B: req_to_token perturb still triggers a canary violation while jitter is on."""

    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-jitter-enabled",
        "--kv-canary-jitter-per-slot-fire-prob",
        "1.0",
        "--kv-canary-jitter-max-cycles",
        "100000",
    ]
    perturb_prob: ClassVar[float] = 0.5
    allow_launch_failure: ClassVar[bool] = True

    def test_perturb_signal_survives_jitter(self) -> None:
        # Step 1: warmup may already trigger a violation under raise mode.
        if self.launch_failed:
            self.assert_violation_kind_logged(
                ["per_forward_", "sweep_"], flush_wait_seconds=2.0
            )
            return

        # Step 2: drive traffic so the perturb knob fires.
        self.send_parallel_requests(n=64, max_new_tokens=32, timeout=30.0)

        # Step 3: jitter must not mask the perturb-induced violation.
        self.assert_violation_kind_logged(
            ["per_forward_", "sweep_"], flush_wait_seconds=2.0
        )


class TestJitterAlonePerturbZeroNoFalseViolation(_JitterE2EBase, unittest.TestCase):
    """Group B: jitter on, perturb off -> no false canary violation introduced by jitter."""

    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-jitter-enabled",
        "--kv-canary-jitter-per-slot-fire-prob",
        "1.0",
        "--kv-canary-jitter-max-cycles",
        "1000000",
    ]
    perturb_prob: ClassVar[float] = 0.0

    def test_jitter_alone_no_false_violation(self) -> None:
        # Step 1: heavier traffic than group A so the no-false-positive case is well-exercised.
        results = self.send_parallel_requests(n=48, max_new_tokens=32, timeout=60.0)

        # Step 2: requests must finish; canary stays silent under jitter alone.
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)
        self.assert_health_ok()


@unittest.skip("depends on SteppableEngine (not yet implemented)")
class TestJitterSpeedsUpKnownRaceDetection(unittest.TestCase):
    """Group C: jitter halves the time-to-first-violation against a revertable known-race PR.

    Depends on SteppableEngine to deterministically drive the revert flow; left as a placeholder
    skip until that lands.
    """

    def test_jitter_speeds_up_race_detection_eagle_positions(self) -> None:
        self.skipTest(
            "revert + baseline vs jitter wall-time comparison; needs SteppableEngine"
        )


if __name__ == "__main__":
    unittest.main()
