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

register_cuda_ci(est_time=1800, stage="nightly", runner_config="4-gpu-b200")

LAGUNA_XS_NVFP4_MODEL = "poolside/Laguna-XS-2.1-NVFP4"
LAGUNA_S_NVFP4_MODEL = "poolside/Laguna-S-2.1-NVFP4"

LAGUNA_NVFP4_ARGS = ["--tp-size", "1"]


def _run_gsm8k(test_case, baseline):
    args = SimpleNamespace(
        model=test_case.model,
        eval_name="gsm8k",
        num_shots=5,
        num_examples=200,
        max_tokens=4096,
        num_threads=128,
        repeat=1,
        temperature=1.0,
        top_p=0.95,
        base_url=test_case.base_url,
        host="http://127.0.0.1",
        port=int(test_case.base_url.split(":")[-1]),
    )
    metrics = run_eval(args)
    print(f"{metrics=}")
    test_case.assertGreaterEqual(metrics["score"], baseline)


class TestLagunaXS21NVFP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LAGUNA_XS_NVFP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=LAGUNA_NVFP4_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # Measured 0.935 (temp=1.0, 200 examples); floored with margin for
        # FP4-kernel variance across Blackwell parts.
        _run_gsm8k(self, 0.87)


class TestLagunaS21NVFP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LAGUNA_S_NVFP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=LAGUNA_NVFP4_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # Measured 0.95 (temp=1.0, 200 examples); floored with margin.
        _run_gsm8k(self, 0.89)


if __name__ == "__main__":
    unittest.main()
