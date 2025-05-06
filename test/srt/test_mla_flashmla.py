import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch,
)


class TestFlashMLAAttnBackend(unittest.TestCase):
    def test_latency(self):
        output_throughput = run_bench_one_batch(
            DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            [
                "--attention-backend",
                "flashmla",
                "--enable-torch-compile",
                "--cuda-graph-max-bs",
                "16",
                "--trust-remote-code",
                "--page-size",
                "64",
            ],
        )

        if is_in_ci():
            self.assertGreater(output_throughput, 153)

    def test_mmlu(self):
        model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "flashmla", "--trust-remote-code", "--page-size", "64"],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], 0.2)
        finally:
            kill_process_tree(process.pid)


class TestFlashMLAMTP(CustomTestCase):
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
                    "--disable-radix",
                    "--enable-torch-compile",
                    "--torch-compile-max-bs",
                    "1",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft",
                    "lmsys/sglang-ci-dsv3-test-NextN",
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--attention-backend",
                    "flashmla",
                    "--page-size",
                    "64",
                    "--disable-cuda-graph",
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

        self.assertGreater(metrics["accuracy"], 0.50)

        server_info = requests.get(self.base_url + "/get_server_info")
        print(f"{server_info=}")
        avg_spec_accept_length = server_info.json()["avg_spec_accept_length"]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 2.5)


if __name__ == "__main__":
    unittest.main()
