"""Regression for PR #25015 EAGLE positions misalign: revert the fix and expect canary fire."""

from __future__ import annotations

import io
import json
import logging
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

logger = logging.getLogger(__name__)

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
        # Caps kept small to keep the e2e test cheap (run time + device memory budget for the
        # canary buffers). Not load-bearing on canary's overflow behavior.
        "--cuda-graph-max-bs",
        "8",
        "--max-running-requests",
        "32",
        "--context-length",
        "2048",
        "--max-total-tokens",
        "16384",
        # sglang piecewise CUDA graph crashes on 1-layer Qwen3 with FusedAddRMSNorm
        # IMA during warmup_compile (reproduces with --kv-canary off). Disable
        # piecewise only; the main cuda graph is still on and still exercises the
        # in-graph canary kernel path.
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
        cls._launch_exc = None
        try:
            cls.process = popen_launch_server(
                _MOCK_MODEL,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=_spec_eagle_server_args(),
                env=env,
                return_stdout_stderr=(cls._stdout_buf, cls._stderr_buf),
            )
        except Exception as exc:
            cls._launch_exc = exc
            logger.warning(
                "server launch raised during revert path: %r", exc, exc_info=True
            )

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
