import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import get_device_sm, kill_process_tree
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
MODEL_USED_FOR_TEST_MLA = "lmsys/sglang-ci-dsv3-test"
# Setting data path to None uses default data path in few_shot_gsm8k eval test.
DATA_PATH = None


@unittest.skipIf(get_device_sm() < 90, "Test requires CUDA SM 90 or higher")
class BaseFlashAttentionTest(unittest.TestCase):
    """Base class for FlashAttention tests to reduce code duplication."""

    model = MODEL_USED_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.62

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        args = [
            "--trust-remote-code",
            "--enable-torch-compile",
            "--attention-backend",
            "fa3",
        ]
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

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
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
        args.extend(
            [
                "--disable-cuda-graph",
            ]
        )
        return args


class TestFlashAttention3MLA(BaseFlashAttentionTest):
    """Test FlashAttention3 with MLA."""

    model = MODEL_USED_FOR_TEST_MLA

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
        args.extend(
            [
                "--cuda-graph-max-bs",
                "2",
            ]
        )
        return args


class TestFlashAttention3SpeculativeDecode(BaseFlashAttentionTest):
    """Test FlashAttention3 with speculative decode enabled."""

    model = "meta-llama/Llama-3.1-8B-Instruct"

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
        args.extend(
            [
                "--cuda-graph-max-bs",
                "2",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft",
                "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "3",
                "--dtype",
                "float16",
            ]
        )
        return args

    def test_gsm8k(self):
        """
        Override the test_gsm8k to further test for average speculative accept length.
        """
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=DATA_PATH,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["avg_spec_accept_length"]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 1.5)


class TestFlashAttention3MLASpeculativeDecode(BaseFlashAttentionTest):
    """Test FlashAttention3 with speculative decode enabled."""

    model = MODEL_USED_FOR_TEST_MLA

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
        args.extend(
            [
                "--cuda-graph-max-bs",
                "2",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft",
                "lmsys/sglang-ci-dsv3-test-NextN",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "3",
            ]
        )
        return args

    def test_gsm8k(self):
        """
        Override the test_gsm8k to further test for average speculative accept length.
        """
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=DATA_PATH,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["avg_spec_accept_length"]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 1.5)


if __name__ == "__main__":
    unittest.main()
