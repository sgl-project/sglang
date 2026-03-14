import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="stage-c-test-4-gpu-b200")

NEMOTRON_3_SUPER_NVFP4_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

NEMOTRON_3_SUPER_NVFP4_ARGS = [
    "--tp-size",
    "4",
    "--trust-remote-code",
    "--reasoning-parser",
    "nemotron_3",
    "--tool-call-parser",
    "qwen3_coder",
    "--disable-radix-cache",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 17}',
]

MTP_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]


def _run_gsm8k(test_case):
    args = SimpleNamespace(
        model=test_case.model,
        eval_name="gsm8k",
        num_shots=5,
        num_examples=200,
        max_tokens=16000,
        num_threads=200,
        repeat=1,
        temperature=1.0,
        top_p=0.95,
        base_url=test_case.base_url,
        host="http://127.0.0.1",
        port=int(test_case.base_url.split(":")[-1]),
    )
    metrics = run_eval(args)
    print(f"{metrics=}")
    test_case.assertGreaterEqual(metrics["score"], 0.96)


class TestNvidiaNemotron3SuperNVFP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = NEMOTRON_3_SUPER_NVFP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=NEMOTRON_3_SUPER_NVFP4_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _run_gsm8k(self)


class TestNvidiaNemotron3SuperNVFP4MTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = NEMOTRON_3_SUPER_NVFP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=NEMOTRON_3_SUPER_NVFP4_ARGS
            + MTP_ARGS
            + ["--max-running-requests", "200"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _run_gsm8k(self)


if __name__ == "__main__":
    unittest.main()
