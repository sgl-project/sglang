import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestMLA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.8)


class TestDeepseekV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "sgl-project/sglang-ci-dsv3-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(["--enable-torch-compile", "--cuda-graph-max-bs", "2"])
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

        self.assertGreater(metrics["accuracy"], 0.62)


class TestDeepseekV3MTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "sgl-project/sglang-ci-dsv3-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(
                [
                    "--cuda-graph-max-bs",
                    "2",
                    "--disable-radix",
                    "--enable-torch-compile",
                    "--torch-compile-max-bs",
                    "1",
                    "--speculative-algorithm",
                    "NEXTN",
                    "--speculative-draft",
                    "sgl-project/sglang-ci-dsv3-test-NextN",
                    "--speculative-num-steps",
                    "2",
                    "--speculative-eagle-topk",
                    "4",
                    "--speculative-num-draft-tokens",
                    "4",
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

        self.assertGreater(metrics["accuracy"], 0.62)


if __name__ == "__main__":
    unittest.main()
