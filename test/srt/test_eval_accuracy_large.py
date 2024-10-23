"""
Usage:
python -m unittest test_eval_accuracy_large.TestEvalAccuracyLarge.test_mmlu
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestEvalAccuracyLarge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--log-level-http", "warning"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=5000,
            num_threads=1024,
        )

        metrics = run_eval(args)
        assert metrics["score"] >= 0.71, f"{metrics}"

    def test_human_eval(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="humaneval",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        assert metrics["score"] >= 0.64, f"{metrics}"

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        assert metrics["score"] >= 0.84, f"{metrics}"


if __name__ == "__main__":
    unittest.main()
