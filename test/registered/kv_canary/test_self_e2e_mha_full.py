"""Canary self-e2e on a minimal MHA + FULL pool fixture.

Qwen3-0.6B with ``num_hidden_layers=1`` baked into ``--json-model-override-args``
gives the cheapest possible "real attention + real KV pool + canary attached"
e2e configuration so PR CI catches integration regressions in 1-2 minutes.

8 cases, all registered to ``extra-a`` / ``1-gpu-large``.
"""

from __future__ import annotations

import unittest
from typing import ClassVar, List

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.utils import CanaryE2EBase

register_cuda_ci(est_time=210, stage="extra-a", runner_config="1-gpu-large")


_QWEN3_MODEL = "Qwen/Qwen3-0.6B"
_NUM_LAYERS_OVERRIDE = '{"num_hidden_layers": 1}'


class _MhaFullBase(CanaryE2EBase):
    model: ClassVar[str] = _QWEN3_MODEL
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
    ]


class TestNoPerturbNoViolation(_MhaFullBase, unittest.TestCase):
    def test_no_perturb_no_violation(self) -> None:
        # Step 1: drive enough forward steps to clear the warmup window.
        results = self.send_parallel_requests(n=16, max_new_tokens=32)

        # Step 2: every request must complete and the /health endpoint must stay up.
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)
        self.assert_health_ok()


class TestPerturbReqToTokenDetectsViolation(_MhaFullBase, unittest.TestCase):
    # Use --kv-canary log (not raise) so the perturb-triggered violation is
    # captured in server stderr without SIGQUIT killing the process mid-flush.
    kv_canary_mode: ClassVar[str] = "log"
    perturb_prob: ClassVar[float] = 0.05

    def test_perturb_req_to_token_detects_violation(self) -> None:
        # Drive traffic; perturb fires per-step at perturb_prob, canary logs.
        self.send_parallel_requests(n=64, max_new_tokens=32, timeout=30.0)
        self.assert_violation_kind_logged(
            ["per_forward_", "sweep_"], flush_wait_seconds=2.0
        )


class TestRealDataOff(_MhaFullBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-real-data",
        "off",
    ]

    def test_real_data_off(self) -> None:
        results = self.send_parallel_requests(n=8, max_new_tokens=16)
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)


class TestRealDataPartial(_MhaFullBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-real-data",
        "partial",
    ]

    def test_real_data_partial(self) -> None:
        results = self.send_parallel_requests(n=8, max_new_tokens=16)
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)


class TestRealDataAll(_MhaFullBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-real-data",
        "all",
    ]

    def test_real_data_all(self) -> None:
        results = self.send_parallel_requests(n=8, max_new_tokens=16)
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)


class TestRealDataAllPerturbKvByteDetectsViolation(_MhaFullBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
    ]
    allow_launch_failure: ClassVar[bool] = True

    @classmethod
    def setUpClass(cls) -> None:
        import os

        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB"] = "0.5"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB", None)
        super().tearDownClass()

    def test_real_data_all_perturb_kv_byte_detects_violation(self) -> None:
        if self.launch_failed:
            self.assert_violation_kind_logged(
                ["per_forward_", "sweep_"], flush_wait_seconds=2.0
            )
            return
        self.send_parallel_requests(n=64, max_new_tokens=32, timeout=30.0)
        self.assert_violation_kind_logged(
            ["per_forward_", "sweep_"], flush_wait_seconds=2.0
        )


class TestLogModeKeepsServerAlive(_MhaFullBase, unittest.TestCase):
    perturb_prob: ClassVar[float] = 0.05
    kv_canary_mode: ClassVar[str] = "log"

    def test_log_mode_keeps_server_alive(self) -> None:
        # Step 1: log mode never fails warmup — violations log, do not raise.
        self.assertFalse(
            self.launch_failed,
            f"Server failed to launch under --kv-canary log: {self.launch_exception!r}",
        )

        # Step 2: clients keep getting 200s even while violations fire on the server side.
        results = self.send_parallel_requests(n=64, max_new_tokens=32, timeout=30.0)
        success_count = sum(1 for r in results if r.get("status_code") == 200)
        self.assertGreater(
            success_count,
            0,
            f"Expected >=1 successful response under log mode, got 0. results={results}",
        )

        # Step 3: server still alive + violation surfaced in server stderr/stdout.
        self.assertIsNone(
            self.process.poll(),
            "Server process died during the test under --kv-canary log",
        )
        self.assert_violation_kind_logged(
            ["per_forward_", "sweep_"], flush_wait_seconds=2.0
        )
        self.assert_health_ok()


class TestSweepOrphanRadixDetectsViolation(_MhaFullBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
    ]
    allow_launch_failure: ClassVar[bool] = True

    @classmethod
    def setUpClass(cls) -> None:
        import os

        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB"] = "0.5"
        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_REQUIRE_ORPHAN"] = "1"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB", None)
        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_REQUIRE_ORPHAN", None)
        super().tearDownClass()

    def test_sweep_orphan_radix_detects_violation(self) -> None:
        # Step 1: send the same prefix repeatedly so radix cache retains orphans.
        prompts = ["The capital of France is" for _ in range(8)]
        if self.launch_failed:
            self.assert_violation_kind_logged(["sweep_"], flush_wait_seconds=2.0)
            return
        self.send_parallel_requests(
            n=32, prompts=prompts, max_new_tokens=8, timeout=30.0
        )

        # Step 2: orphan-targeted perturb is only caught by the sweep path.
        self.assert_violation_kind_logged(["sweep_"], flush_wait_seconds=2.0)


if __name__ == "__main__":
    unittest.main()
