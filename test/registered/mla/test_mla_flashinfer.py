import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# FlashInfer MLA backend tests with MTP speculative decoding
register_cuda_ci(est_time=264, suite="stage-b-test-1-gpu-large")


class TestFlashinferMLA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(
                [
                    "--enable-torch-compile",
                    "--cuda-graph-max-bs",
                    "4",
                    "--attention-backend",
                    "flashinfer",
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

        self.assertGreater(metrics["score"], 0.615)


class TestFlashinferMLAMTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(
                [
                    "--cuda-graph-max-bs",
                    "4",
                    "--enable-torch-compile",
                    "--torch-compile-max-bs",
                    "1",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--attention-backend",
                    "flashinfer",
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
        requests.get(self.base_url + "/flush_cache")

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

        self.assertGreater(metrics["score"], 0.60)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 2.5)


if __name__ == "__main__":
    unittest.main()
