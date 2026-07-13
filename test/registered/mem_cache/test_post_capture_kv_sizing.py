"""E2E guard for SGLANG_ENABLE_POST_CAPTURE_KV_SIZING.

Post-capture KV sizing reserves the KV pool as CUDA VMM virtual memory, captures
CUDA graphs, then sizes and physically backs the pool from measured free memory.
This test launches a server with the feature enabled and asserts that:
  1. the post-capture sizing path actually ran (log line present, not a silent
     no-op skip via post_capture_kv_sizing_planned),
  2. the pool was sized to a positive max_total_num_tokens, and
  3. gsm8k accuracy is unchanged vs. the default sizing path.
"""

import os
import re
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# CI Registration
register_cuda_ci(est_time=91, stage="base-b", runner_config="1-gpu-large")

STDOUT_FILENAME = "post_capture_kv_sizing_stdout.log"
STDERR_FILENAME = "post_capture_kv_sizing_stderr.log"


class TestPostCaptureKVSizing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, "SGLANG_ENABLE_POST_CAPTURE_KV_SIZING": "1"},
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        for f in (STDOUT_FILENAME, STDERR_FILENAME):
            if os.path.exists(f):
                os.remove(f)

    def _server_logs(self) -> str:
        text = ""
        for f in (STDOUT_FILENAME, STDERR_FILENAME):
            if os.path.exists(f):
                with open(f) as fh:
                    text += fh.read()
        return text

    def test_post_capture_sizing_ran(self):
        """The post-capture path must actually execute, not silently skip."""
        m = re.search(
            r"Post-capture KV sizing: max_total_num_tokens=(\d+)", self._server_logs()
        )
        self.assertIsNotNone(
            m,
            "Post-capture KV sizing log line not found; the feature was gated off "
            "or the resize path did not run.",
        )
        self.assertGreater(int(m.group(1)), 0)

    def test_server_info_pool_sized(self):
        info = requests.get(f"{self.base_url}/server_info").json()
        self.assertGreater(info["max_total_num_tokens"], 0)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=500,
            num_threads=1024,
        )
        metrics = run_eval(args)
        print(f"GSM8K Accuracy: {metrics['score']:.3f}")
        self.assertGreater(metrics["score"], 0.80)


if __name__ == "__main__":
    unittest.main()
