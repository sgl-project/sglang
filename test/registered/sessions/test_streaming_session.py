"""Per-commit streaming-session tests.

Only TestStreamingSessionEagleV2RetractLargePage and
TestStreamingSessionAbortLeakRepro stay per-commit. Everything else
(base + Retract* + Eagle*/EagleV2 + EagleRetractLargePage) lives in
the sibling test_streaming_session_extra.py.
"""

import asyncio
import os
import sys
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# test/ has no __init__.py; add sibling dir so the nightly module is
# importable when this file is run as a script via `python3 <path>`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_streaming_session_extra import (  # noqa: E402
    ABORT_REPRO_CHUNKED_PREFILL_SIZE,
    ABORT_REPRO_CONTEXT_LEN,
    ABORT_REPRO_PAGE_SIZE,
)
from test_streaming_session_extra import (  # noqa: E402
    TestStreamingSession as _StreamingSessionBase,
)
from test_streaming_session_extra import (  # noqa: E402
    _abort_repro_run_all,
)

register_cuda_ci(est_time=691, stage="stage-b", runner_config="1-gpu-large")


class TestStreamingSessionEagleV2RetractLargePage(_StreamingSessionBase):
    """EAGLE3 spec v2 + retract + page=256."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE3
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_SPEC_V2.override(
            True
        ), envs.SGLANG_TEST_RETRACT.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(
            2
        ), envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(
            True
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "4096",
                    "--dtype=float16",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model",
                    DEFAULT_DRAFT_MODEL_EAGLE3,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-fraction-static",
                    "0.7",
                    "--page-size",
                    "256",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionAbortLeakRepro(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    str(ABORT_REPRO_CHUNKED_PREFILL_SIZE),
                    "--context-length",
                    str(ABORT_REPRO_CONTEXT_LEN),
                    "--page-size",
                    str(ABORT_REPRO_PAGE_SIZE),
                    "--max-running-requests",
                    "32",
                    "--log-level",
                    "info",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_abort_heavy_chunked_prefill_does_not_leak(self) -> None:
        requests.post(self.base_url + "/flush_cache")

        asyncio.run(_abort_repro_run_all(self.base_url, self.tokenizer))

        for i in range(3):
            ids = self.tokenizer.encode(f"Post-session cleanup request {i}.")
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
                timeout=30,
            )
            self.assertEqual(response.status_code, 200, response.text)

        time.sleep(5)
        self.assertIsNone(
            self.process.poll(),
            "Server crashed during abort-heavy streaming session repro.",
        )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after abort-heavy streaming session cleanup.",
        )


if __name__ == "__main__":
    unittest.main()
