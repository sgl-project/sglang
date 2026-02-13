import random
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_npu_ci(est_time=500, suite="nightly-16-npu-a3", nightly=True)


class TestDPAttentionRoundBinLoadBalance(CustomTestCase):
    """
    Testcase：Verify that the inference is successful when --load-balance-method is set to round_robin, auto,
    follow_bootstrap_room, total_requests, total_tokens

    [Test Category] Parameter
    [Test Target] --load-balance-method
    """

    mode = "round_robin"

    @classmethod
    def setUpClass(cls):
        cls.model_path = DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "16",
            "--enable-dp-attention",
            "--dp",
            "1",
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

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model_path,
            eval_name="mgsm_en",
            num_examples=10,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class _TestDPAttentionAutoLoadBalance(TestDPAttentionRoundBinLoadBalance):
    mode = "auto"


class _TestDPAttentionFollowBootstrapRoomLoadBalance(
    TestDPAttentionRoundBinLoadBalance
):
    mode = "follow_bootstrap_room"


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
            _TestDPAttentionFollowBootstrapRoomLoadBalance,
            _TestDPAttentionTotalRequestsLoadBalance,
            _TestDPAttentionTotalTokensLoadBalance,
        ]
        suite.addTests(loader.loadTestsFromTestCase(random.choice(RUN_FLAG)))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        unittest.main()
