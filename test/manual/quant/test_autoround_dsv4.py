"""
Usage:
python3 -m unittest test_autoround_dsv4.TestAutoRoundDeepSeekV4.test_mmlu

Requires 8 GPUs (TP=8) and the quantized checkpoint:
  Intel/DeepSeek-V4-Pro-W4A16-AutoRound
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = "Intel/DeepSeek-V4-Pro-W4A16-AutoRound"


class TestAutoRoundDeepSeekV4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--quantization",
                "auto-round",
                "--tp",
                "8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=MODEL,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.6)


if __name__ == "__main__":
    unittest.main()
