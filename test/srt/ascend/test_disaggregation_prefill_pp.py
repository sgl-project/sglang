import json
import os
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_npu,
    kill_process_tree,
    popen_launch_pd_server,
)


class TestDisaggregationPrefillPp(TestDisaggregationBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = (
            "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
            if is_npu()
            else DEFAULT_MODEL_NAME_FOR_TEST
        )
        if is_npu():
            os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"
            env = os.environ.copy()
        else:
            env = None

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = (
            [
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.8,
            ]
            if is_npu()
            else [
                "--disaggregation-mode",
                "prefill",
            ]
        )

        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = (
            [
                "--disaggregation-mode",
                "decode",
                "--base-gpu-id",
                "2",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
                "--disaggregation-prefill-pp",
                "2",
                "--mem-fraction-static",
                0.8,
            ]
            if is_npu()
            else [
                "--disaggregation-mode",
                "decode",
                "--base-gpu-id",
                "2",
                "--disaggregation-prefill-pp",
                "2",
            ]
        )
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_disaggregation_prefill_pp(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(self.decode_url + "/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["disaggregation_prefill_pp"], 2)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
