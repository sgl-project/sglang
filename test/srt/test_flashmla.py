"""
Usage:
python3 test/srt/test_flashmla.py
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch,
)


class TestFlashMLAAttnBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(
                [
                    "--enable-torch-compile",
                    "--cuda-graph-max-bs",
                    "2",
                    "--attention-backend",
                    "flashmla",
                ]
            )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)


class TestFlashMLAAttnLatency(unittest.TestCase):
    def test_latency(self):
        output_throughput = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST_MLA,
            [
                "--attention-backend",
                "flashmla",
                "--enable-torch-compile",
                "--cuda-graph-max-bs",
                "16",
                "--trust-remote-code",
            ],
        )

        if is_in_ci():
            self.assertGreater(output_throughput, 100)


if __name__ == "__main__":
    unittest.main()
