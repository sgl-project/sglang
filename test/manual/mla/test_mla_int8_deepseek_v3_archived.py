"""Archived test classes split out of test/registered/mla/test_mla_int8_deepseek_v3.py.

Originally registered with `register_cuda_ci(...)`. Moved here as part of
the per-commit pruning effort to keep the code reachable manually.
Run with `python3 test/manual/mla/test_mla_int8_deepseek_v3_archived.py`.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)


# DeepSeek-V3 INT8 quantization tests (channel and block INT8)
class TestMLADeepseekV3ChannelInt8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-channel-int8-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(
                [
                    "--cuda-graph-max-bs",
                    "16",
                    "--enable-torch-compile",
                    "--torch-compile-max-bs",
                    "2",
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
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreaterEqual(metrics["score"], 0.61)


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestMLADeepseekV3BlockInt8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-block-int8-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(
                [
                    "--cuda-graph-max-bs",
                    "16",
                    "--enable-torch-compile",
                    "--torch-compile-max-bs",
                    "2",
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
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreater(metrics["score"], 0.62)


if __name__ == "__main__":
    unittest.main()
