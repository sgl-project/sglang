import unittest

import requests
from sglang.test.ascend.test_ascend_utils import QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestEnableTokenizerBatchEncode(CustomTestCase):
    """Testcase：Verify the MambaCache

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.model_path

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--max-mamba-cache-size",
            256,
            "--mamba-ssm-dtype",
            "float32",
            "--mamba-full-memory-ratio",
            "0.9",
            "--mamba-scheduler-strategy",
            "auto",
            "--mamba-track-interval",
            "256",
            "--tp-size",
            8,
        ]

        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_enable_tokenizer_batch_encode(self):
        for i in range(50) :
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(
                response.status_code, 200, "The request status code is not 200."
            )
            self.assertIn(
                "Paris", response.text, "The inference result does not include Paris."
            )


if __name__ == "__main__":
    unittest.main()
