import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_5_35B_A3B_WEIGHTS_PATH,
    QWEN3_5_35B_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-4-npu-a3", nightly=True)


class TestDtypeAuto(CustomTestCase):
    """Testcase: Verify set --deepep-dispatcher-output-dtype the inference accuracy of the model on the
    GSM8K dataset is no less than 0.74.

    [Test Category] Parameters
    [Test Target] --deepep-dispatcher-output-dtype
    """

    dtype = "auto"

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
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--deepep-dispatcher-output-dtype",
                cls.dtype,
                "--attention-backend",
                "ascend",
                "--max-total-tokens",
                "66000",
                "--cuda-graph-max-bs",
                "128",
                "--log-level",
                "info",
                "--disable-radix-cache",
            ],
            env={
                "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",  # Quantize activations to INT8 before dispatch
                "HCCL_BUFFSIZE": "1536",
            },
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


class TestDtypeBf16(TestDtypeAuto):
    dtype = "bf16"

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_5_35B_A3B_WEIGHTS_PATH
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
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--deepep-dispatcher-output-dtype",
                cls.dtype,
                "--attention-backend",
                "ascend",
                "--max-total-tokens",
                "66000",
                "--cuda-graph-max-bs",
                "128",
                "--log-level",
                "info",
                "--disable-radix-cache",
            ],
            env={
                "HCCL_BUFFSIZE": "1536",
            },
        )


class TestDtypeInt8(TestDtypeAuto):
    dtype = "int8"


if __name__ == "__main__":
    unittest.main()
