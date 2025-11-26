import os
import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEplbAlgorithm(CustomTestCase):
    eplb_algorithm = "dynamic"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        if is_npu():
            cls.model = (
                "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-W8A8"
            )
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size",
                    "16",
                    "--dp-size",
                    "1",
                    "--attention-backend",
                    "ascend",
                    "--quantization",
                    "w8a8_int8",
                    "--mem-fraction-static",
                    "0.9",
                    "--enable-dp-attention",
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "normal",
                    "--disable-cuda-graph",
                    "--enable-eplb",
                    "--ep-num-redundant-experts",
                    "16",
                    "--eplb-rebalance-num-iterations",
                    "50",
                    "--expert-distribution-recorder-buffer-size",
                    "50",
                    # TODO pr-chain: enable later
                    "--enable-expert-distribution-metrics",
                    # TODO auto determine these flags
                    "--expert-distribution-recorder-mode",
                    "stat",
                    "--ep-dispatch-algorithm",
                    "static",
                    "--eplb-algorithm",
                    cls.eplb_algorithm,
                ],
                env={
                    "SGL_ENABLE_JIT_DEEPGEMM": "0",
                    "HCCL_BUFFSIZE": "512",
                    **os.environ,
                },
            )
        else:
            cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp",
                    "2",
                    "--dp",
                    "2",
                    "--enable-dp-attention",
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "normal",
                    "--disable-cuda-graph",
                    "--enable-eplb",
                    "--ep-num-redundant-experts",
                    "4",
                    "--eplb-rebalance-num-iterations",
                    "50",
                    "--expert-distribution-recorder-buffer-size",
                    "50",
                    # TODO pr-chain: enable later
                    # "--enable-expert-distribution-metrics",
                    # TODO auto determine these flags
                    "--expert-distribution-recorder-mode",
                    "stat",
                    "--ep-dispatch-algorithm",
                    "static",
                    "--eplb-algorithm",
                    cls.eplb_algorithm,
                ],
                env={
                    "SGL_ENABLE_JIT_DEEPGEMM": "0",
                    "SGLANG_EXPERT_LOCATION_UPDATER_CANARY": "1",
                    **os.environ,
                },
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_eplb_algorithm(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        print(response.json())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.eplb_algorithm, response.json().get("eplb_algorithm"))

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

        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


class TestEplbAlgorithmStatic(TestEplbAlgorithm):
    eplb_algorithm = "static"


if __name__ == "__main__":
    unittest.main()
