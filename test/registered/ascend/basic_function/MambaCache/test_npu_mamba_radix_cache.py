import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestMambaCache(CustomTestCase):
    """Testcase：Verify the test Radix Cache reuse, when use mamba cache.

    [Test Category] Parameter
    [Test Target] --disable-radix-cache
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.model_path

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.5",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--device",
            "npu",
            "--tp-size",
            "8",
            "--mamba-ssm-dtype",
            "float32",
            "--mamba-full-memory-ratio",
            "0.5",
            "--mamba-scheduler-strategy",
            "auto",
            "--mamba-track-interval",
            "256",
            "--base-gpu-id",
            "8",
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

    def test_mamba_cache_kv_cache(self):
        # test kv cache reuse
        input_ids_first = [1] * 200
        input_ids_second = input_ids_first + [2] * 70

        def make_request(input_ids, expected_cached_tokens):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["meta_info"]["cached_tokens"], expected_cached_tokens
            )

        make_request(input_ids_first, 0)

        make_request(input_ids_second, 128)


if __name__ == "__main__":
    unittest.main()
