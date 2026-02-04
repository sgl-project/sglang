import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestAscendDistTimeout(CustomTestCase):
    """Testcase: Verify that when --dist-timeout is set to 3600, no timeout is triggered during service startup and the model accuracy remains uncompromised.

    [Test Category] Parameter
    [Test Target] --dist-timeout
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
        cls.accuracy = 0.95
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)

        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--mem-fraction-static",
            0.87,
            "--disable-radix-cache",
            "--chunked-prefill-size",
            32768,
            "--tp-size",
            16,
            "--dist-timeout",
            3600,
            "--disable-cuda-graph",
        ]

    def test_gsm8k(self):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=3600,
            other_args=[
                *self.common_args,
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=1319,
                max_new_tokens=512,
                parallel=128,
                host=f"http://{self.url.hostname}",
                port=int(self.url.port),
            )

            metrics = run_eval_few_shot_gsm8k(args)
            self.assertGreaterEqual(
                metrics["accuracy"],
                self.accuracy,
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
