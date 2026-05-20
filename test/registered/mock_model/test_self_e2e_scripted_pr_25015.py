"""Regression for PR #25015 EAGLE positions misalign: revert the fix and expect canary fire."""

from __future__ import annotations

import io
import json
import os
import unittest
from typing import List

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")


_MOCK_MODEL = "Qwen/Qwen3-0.6B"
_NUM_LAYERS_OVERRIDE = json.dumps({"num_hidden_layers": 1})


def _spec_eagle_server_args() -> List[str]:
    return [
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--sampling-backend",
        "token_oracle",
        "--kv-canary",
        "raise",
        "--speculative-algorithm",
        "EAGLE",
        # Cap canary's per-forward + sweep capacities below the 1M
        # cuda-grid-safe ceiling enforced by install_canary():
        #   per-forward: cuda_graph_max_bs * req_to_token_cols
        #   sweep:       max_total_num_tokens
        # EAGLE defaults blow past both with 256 * 40968 = 10.4M and KV-pool
        # auto-sized to ~32M slots. Pin them small.
        "--cuda-graph-max-bs",
        "8",
        "--context-length",
        "2048",
        "--max-total-tokens",
        "16384",
        # Workaround: sglang piecewise CUDA graph hits FusedAddRMSNorm illegal
        # memory access during warmup_compile under EAGLE + Qwen3-0.6B(1 layer).
        # Reproduces with --kv-canary off (verified in repro test), so this is an
        # upstream sglang piecewise bug, not a canary issue. sglang itself prints
        # the suggested workaround in its error message. This is the ONLY canary
        # e2e test allowed to pass --disable-piecewise-cuda-graph; the rest must
        # exercise the in-graph canary kernel path per user-instruction b 段.
        "--disable-piecewise-cuda-graph",
    ]


def _spec_eagle_env() -> dict[str, str]:
    env = os.environ.copy()
    env["SGLANG_KV_CANARY_INPUT_CHECK"] = "0"
    env["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1"
    return env


class TestEaglePositionsMisalignRegression(CustomTestCase):
    """Revert PR #25015 fix and expect canary to fire POSITION_MISMATCH."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._stdout_buf = io.StringIO()
        cls._stderr_buf = io.StringIO()
        cls.process = None
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = _spec_eagle_env()
        env["SGLANG_DEBUG_REVERT_PR"] = "25015"
        try:
            cls.process = popen_launch_server(
                _MOCK_MODEL,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=_spec_eagle_server_args(),
                env=env,
                return_stdout_stderr=(cls._stdout_buf, cls._stderr_buf),
            )
        except Exception:
            pass

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_position_mismatch_in_server_stderr(self) -> None:
        haystack = (self._stderr_buf.getvalue() if self._stderr_buf else "") + (
            self._stdout_buf.getvalue() if self._stdout_buf else ""
        )
        self.assertIn("POSITION_MISMATCH", haystack)


class TestEaglePositionsMatchWithFix(CustomTestCase):
    """With the PR #25015 fix in place, no canary fires."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._stdout_buf = io.StringIO()
        cls._stderr_buf = io.StringIO()
        cls.process = None
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.process = popen_launch_server(
            _MOCK_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_spec_eagle_server_args(),
            env=_spec_eagle_env(),
            return_stdout_stderr=(cls._stdout_buf, cls._stderr_buf),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_no_canary_fire(self) -> None:
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": list(range(1, 65)),
                "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
            },
            timeout=60.0,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        health = requests.get(self.base_url + "/health", timeout=10.0)
        self.assertEqual(health.status_code, 200, health.text)


if __name__ == "__main__":
    unittest.main()
