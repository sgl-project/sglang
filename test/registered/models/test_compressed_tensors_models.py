# Model tests for compressed tensors (FP8)

import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=42, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=42, suite="stage-b-test-1-gpu-small-amd")


class TestCompressedTensorsLlama3FP8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "RedHatAI/Meta-Llama-3.1-8B-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[],
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
        print(f"{metrics=}")
        if is_hip():
            # Lower threshold for AMD because FP8 dtype differs (fp8_fnuz)
            self.assertGreaterEqual(metrics["score"], 0.40)
        else:
            self.assertGreaterEqual(metrics["score"], 0.45)


if __name__ == "__main__":
    unittest.main()
