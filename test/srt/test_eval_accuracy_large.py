"""
Usage:
python -m unittest test_eval_accuracy_large.TestEvalAccuracyLarge.test_mmlu
"""

import os
import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class TestEvalAccuracyLarge(CustomTestCase):
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
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=5000,
            num_threads=1024,
        )

        metrics = run_eval(args)

        if is_in_ci():
            write_github_step_summary(f"### test_mmlu\n" f'{metrics["score"]=:.4f}\n')

        self.assertGreater(metrics["score"], 0.70)

    def test_human_eval(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="humaneval",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)

        if is_in_ci():
            write_github_step_summary(
                f"### test_human_eval\n" f'{metrics["score"]=:.4f}\n'
            )

        self.assertGreater(metrics["score"], 0.64)

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)

        if is_in_ci():
            write_github_step_summary(
                f"### test_mgsm_en\n" f'{metrics["score"]=:.4f}\n'
            )

        self.assertGreater(metrics["score"], 0.835)


if __name__ == "__main__":
    unittest.main()
