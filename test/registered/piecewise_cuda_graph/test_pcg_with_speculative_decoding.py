"""Test piecewise CUDA graph coexisting with speculative decoding.

PCG handles prefill/extend path while speculative decoding (MTP/EAGLE3/STANDALONE/NGRAM)
uses decode CUDA graphs. This test verifies they don't interfere with each other.
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=531, stage="stage-b", runner_config="2-gpu-large")


class TestPCGWithMTP(unittest.TestCase):
    """Test PCG + MTP (NEXTN) on Qwen3.5-35B-A3B with FP8."""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3.5-35B-A3B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "2",
            "--trust-remote-code",
            "--quantization",
            "fp8",
            "--mamba-scheduler-strategy",
            "extra_buffer",
            "--enable-piecewise-cuda-graph",
            "--speculative-algorithm",
            "NEXTN",
            "--reasoning-parser",
            "qwen3",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=8192,
            num_examples=200,
            num_threads=200,
            thinking_mode="qwen3",
        )
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], 0.75)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 1.5)


class TestPCGWithEAGLE3(unittest.TestCase):
    """Test PCG + EAGLE3 on Qwen3-30B-A3B-Instruct-2507."""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "2",
            "--trust-remote-code",
            "--enforce-piecewise-cuda-graph",
            "--mem-fraction-static",
            "0.6",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex",
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "4",
            "--speculative-num-draft-tokens",
            "8",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=other_args,
            env={"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=512,
            num_examples=200,
            num_threads=200,
        )
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], 0.75)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 1.5)


class TestPCGWithSTANDALONE(unittest.TestCase):
    """Test PCG + STANDALONE on Llama-3.1-8B-Instruct + Llama-3.2-1B-Instruct."""

    @classmethod
    def setUpClass(cls):
        cls.model = "meta-llama/Llama-3.1-8B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--enforce-piecewise-cuda-graph",
            "--mem-fraction-static",
            "0.5",
            "--speculative-algorithm",
            "STANDALONE",
            "--speculative-draft-model-path",
            "meta-llama/Llama-3.2-1B-Instruct",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 2,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=512,
            num_examples=200,
            num_threads=200,
        )
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], 0.50)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 1.5)


class TestPCGWithNGRAM(unittest.TestCase):
    """Test PCG + NGRAM on Qwen2.5-Coder-7B-Instruct."""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--enforce-piecewise-cuda-graph",
            "--speculative-algorithm",
            "NGRAM",
            "--speculative-num-draft-tokens",
            "16",
            "--cuda-graph-max-bs",
            "8",
            "--mem-fraction-static",
            "0.8",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 2,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=512,
            num_examples=200,
            num_threads=200,
        )
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], 0.70)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 1.5)


if __name__ == "__main__":
    unittest.main()
