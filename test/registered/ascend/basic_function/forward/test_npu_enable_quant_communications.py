import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_5_35B_W8A8_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-4-npu-a3", nightly=True)


class TestEnableQuantCommunications(CustomTestCase):
    """Testcase: Verify set --enable-quant-communications the inference accuracy of the model on the
    GSM8K dataset is no less than 0.74.

    [Test Category] Parameters
    [Test Target] --enable-quant-communications
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_5_35B_W8A8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-quant-communications",
                "--log-level",
                "info",
            ],
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
        print(f"Eval accuracy of GSM8K: {metrics=}")
        self.assertGreater(metrics["score"], 0.74)


if __name__ == "__main__":
    unittest.main()
