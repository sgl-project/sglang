import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

"""
Integration test for python/sglang/srt/layers/attention/flashattention_backend.py
"""
# Change to your own model if testing model is not public.
MODEL_USED_FOR_TEST = DEFAULT_MODEL_NAME_FOR_TEST
# Setting data path to None uses default data path in few_shot_gsm8k eval test.
DATA_PATH = None


class TestFlashAttention3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_USED_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(
                [
                    "--enable-torch-compile",
                    "--cuda-graph-max-bs",
                    "2",
                    "--attention-backend",
                    "fa3",
                ]
            )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
            data_path=DATA_PATH,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.62)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestFlashAttention3DisableCudaGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_USED_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(
                [
                    "--enable-torch-compile",
                    "--disable-cuda-graph",
                    "--cuda-graph-max-bs",
                    "4",
                    "--attention-backend",
                    "fa3",
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
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
            data_path=DATA_PATH,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.62)


if __name__ == "__main__":
    unittest.main()
