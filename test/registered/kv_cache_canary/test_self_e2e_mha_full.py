"""Canary self-e2e on a minimal MHA + FULL pool fixture.

Qwen3-0.6B with ``num_hidden_layers=1`` baked into ``--json-model-override-args``
gives the cheapest possible "real attention + real KV pool + canary attached"
e2e configuration so PR CI catches integration regressions in 1-2 minutes.

testing.md SOT §3.2 — 7 cases, all under ``extra-a`` / ``1-gpu-large``.
"""

from __future__ import annotations

import unittest
from typing import ClassVar, List

from sglang.test.canary_e2e_base import CanaryE2EBase
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")


_QWEN3_MODEL = "Qwen/Qwen3-0.6B"
_NUM_LAYERS_OVERRIDE = '{"num_hidden_layers": 1}'


class _MhaFullBase(CanaryE2EBase):
    model: ClassVar[str] = _QWEN3_MODEL
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
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
    perturb_prob: ClassVar[float] = 0.01
    allow_launch_failure: ClassVar[bool] = True

    def test_perturb_req_to_token_detects_violation(self) -> None:
        # Step 1: warmup itself may already trigger a violation under raise mode.
        if self.launch_failed:
            self.assert_violation_kind_logged(
                ["per_forward_", "sweep_"], flush_wait_seconds=2.0
            )
            return

        # Step 2: drive traffic to give the perturb knob a chance to fire.
        self.send_parallel_requests(n=64, max_new_tokens=32, timeout=30.0)

        # Step 3: confirm a violation surfaced in captured server output.
        self.assert_violation_kind_logged(
            ["per_forward_", "sweep_"], flush_wait_seconds=2.0
        )


class TestRealDataOff(_MhaFullBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--kv-cache-canary-real-data",
        "off",
    ]

    def test_real_data_off(self) -> None:
        results = self.send_parallel_requests(n=8, max_new_tokens=16)
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)


class TestRealDataBit(_MhaFullBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--kv-cache-canary-real-data",
        "portion",
    ]

    def test_real_data_bit(self) -> None:
        results = self.send_parallel_requests(n=8, max_new_tokens=16)
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)


class TestRealDataAll(_MhaFullBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--kv-cache-canary-real-data",
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
        "--kv-cache-canary-real-data",
        "all",
        "--kv-cache-canary-real-data-sweep-every-n-steps",
        "1",
    ]
    allow_launch_failure: ClassVar[bool] = True

    @classmethod
    def setUpClass(cls) -> None:
        import os

        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB"] = "0.05"
        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED"] = "1"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB", None)
        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED", None)
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


class TestSweepOrphanRadixDetectsViolation(_MhaFullBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--kv-cache-canary-real-data",
        "all",
        "--kv-cache-canary-real-data-sweep-every-n-steps",
        "1",
    ]
    allow_launch_failure: ClassVar[bool] = True

    @classmethod
    def setUpClass(cls) -> None:
        import os

        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB"] = "0.05"
        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED"] = "2"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB", None)
        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED", None)
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
