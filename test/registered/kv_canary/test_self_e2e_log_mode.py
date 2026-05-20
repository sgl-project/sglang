"""Canary e2e for ``--kv-canary log`` mode.

``log`` mode (vs ``raise``) keeps the server alive when canary violations are
detected; it merely logs them. This test verifies that contract end-to-end:
inject a violation via perturb knob, drive traffic, then assert the server
keeps serving and the violation message appears in captured server output.
"""

from __future__ import annotations

import time
import unittest
from typing import ClassVar, List

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.utils import CanaryE2EBase

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-large")


_QWEN3_MODEL = "Qwen/Qwen3-0.6B"
_NUM_LAYERS_OVERRIDE = '{"num_hidden_layers": 1}'


class TestLogModeKeepsServerAlive(CanaryE2EBase, unittest.TestCase):
    model: ClassVar[str] = _QWEN3_MODEL
    kv_canary_mode: ClassVar[str] = "log"
    perturb_prob: ClassVar[float] = 0.2
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--kv-canary-real-data",
        "partial",
    ]

    def test_log_mode_keeps_server_alive(self) -> None:
        # Step 1: server must have launched (log mode never fails warmup).
        self.assertFalse(
            self.launch_failed,
            f"Server failed to launch under --kv-canary log: {self.launch_exception!r}",
        )
        self.assertIsNotNone(self.process)
        self.assertIsNone(
            self.process.poll(),
            "Server process exited unexpectedly after launch",
        )

        # Step 2: drive traffic to give the perturb knob a chance to fire.
        results = self.send_parallel_requests(n=64, max_new_tokens=32, timeout=30.0)

        # Step 3: at least one prompt must succeed (log mode does not raise to client).
        success_count = sum(1 for r in results if r.get("status_code") == 200)
        self.assertGreater(
            success_count,
            0,
            f"Expected at least one successful response under log mode, got 0. "
            f"Results: {results}",
        )

        # Step 4: server must still be alive at the end.
        self.assertIsNone(
            self.process.poll(),
            "Server process died during the test under --kv-canary log",
        )

        # Step 5: at least one violation warning should appear in server logs.
        time.sleep(2.0)
        haystack = (self._stderr_buf.getvalue() if self._stderr_buf else "") + (
            self._stdout_buf.getvalue() if self._stdout_buf else ""
        )
        self.assertIn(
            "KV cache canary violation detected",
            haystack,
            f"Expected canary violation log under log mode + perturb, but none found. "
            f"Tail of captured output:\n{haystack[-2000:]}",
        )

        # Step 6: /health still responds (definitive liveness check).
        self.assert_health_ok()


if __name__ == "__main__":
    unittest.main()
