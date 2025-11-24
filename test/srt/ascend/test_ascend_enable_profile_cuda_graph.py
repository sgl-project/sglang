import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEnableProfileCudaGraph(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-profile-cuda-graph",
            ]
            if is_npu()
            else ["--enable-profile-cuda-graph"]
        )

        cls.process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B"
                if is_npu()
                else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_enable_profile_cuda_graph(self):
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
        print(response.text)
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )

        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        print(response.json())
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertTrue(
            response.json()["enable_profile_cuda_graph"],
            "--enable-profile-cuda-graph is not taking effect.",
        )


if __name__ == "__main__":
    unittest.main()
