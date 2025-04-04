import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
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
MODEL_USED_FOR_TEST_MLA = DEFAULT_MLA_MODEL_NAME_FOR_TEST
# Setting data path to None uses default data path in few_shot_gsm8k eval test.
DATA_PATH = None


class BaseFlashAttentionTest(unittest.TestCase):
    """Base class for FlashAttention tests to reduce code duplication."""

    model = MODEL_USED_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.62

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            args.extend(
                [
                    "--enable-torch-compile",
                    "--attention-backend",
                    "fa3",
                ]
            )
        return args

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
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

        # Use the appropriate metric key based on the test class
        metric_key = "accuracy"
        self.assertGreater(metrics[metric_key], self.accuracy_threshold)


class TestFlashAttention3(BaseFlashAttentionTest):
    """Test FlashAttention3 with MLA model and CUDA graph enabled."""

    model = MODEL_USED_FOR_TEST_MLA

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
        if torch.cuda.is_available() and torch.version.cuda:
            args.extend(
                [
                    "--cuda-graph-max-bs",
                    "2",
                ]
            )
        return args


class TestFlashAttention3DisableCudaGraph(BaseFlashAttentionTest):
    """Test FlashAttention3 with CUDA graph disabled."""

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
        if torch.cuda.is_available() and torch.version.cuda:
            args.extend(
                [
                    "--disable-cuda-graph",
                ]
            )
        return args


class TestFlashAttention3DisableMLA(BaseFlashAttentionTest):
    """Test FlashAttention3 with MLA disabled."""

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
        if torch.cuda.is_available() and torch.version.cuda:
            args.extend(
                [
                    "--cuda-graph-max-bs",
                    "2",
                    "--disable-mla",
                ]
            )
        return args


if __name__ == "__main__":
    unittest.main()
