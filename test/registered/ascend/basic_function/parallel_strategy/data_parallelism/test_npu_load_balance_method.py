import random
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_npu_ci(est_time=700, suite="nightly-16-npu-a3", nightly=True)


class TestDPAttentionRoundBinLoadBalance(CustomTestCase):
    """Testcase：Verify that the model accuracy did not decrease when --load-balance-method is set to round_robin, auto,
    total_requests or total_tokens in PD mixed scenario

    [Test Category] Parameter
    [Test Target] --load-balance-method
    """

    mode = "round_robin"

    @classmethod
    def setUpClass(cls):
        cls.model_path = DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        other_args = [
            "--trust-remote-code",
            "--tp",
            "16",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--enable-torch-compile",
            "--torch-compile-max-bs",
            "2",
            "--load-balance-method",
            cls.mode,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--quantization",
            "modelslim",
            "--mem-fraction-static",
            "0.75",
        ]

        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=3 * DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
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
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )

        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            0.95,
        )

    def test_server_info(self):
        response = requests.get(f"{self.base_url}/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertIn(self.mode, response.text)


class _TestDPAttentionAutoLoadBalance(TestDPAttentionRoundBinLoadBalance):
    mode = "auto"


class _TestDPAttentionTotalRequestsLoadBalance(TestDPAttentionRoundBinLoadBalance):
    mode = "total_requests"


class _TestDPAttentionTotalTokensLoadBalance(TestDPAttentionRoundBinLoadBalance):
    mode = "total_tokens"


if __name__ == "__main__":
    # To reduce the CI execution time.
    if is_in_ci():
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        RUN_FLAG = [
            TestDPAttentionRoundBinLoadBalance,
            _TestDPAttentionAutoLoadBalance,
            _TestDPAttentionTotalRequestsLoadBalance,
            _TestDPAttentionTotalTokensLoadBalance,
        ]
        suite.addTests(loader.loadTestsFromTestCase(random.choice(RUN_FLAG)))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        unittest.main()
