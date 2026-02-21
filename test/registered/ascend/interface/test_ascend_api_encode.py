import logging
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import GME_QWEN2_VL_2B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestAscendApi(CustomTestCase):
    """Testcase: Verify the availability and correctness of the /encode API on Ascend backend with GME_QWEN2_VL_2B_INSTRUCT model.

    [Test Category] Interface
    [Test Target] /encode
    """

    @classmethod
    def setUpClass(cls):
        cls.model = GME_QWEN2_VL_2B_INSTRUCT_WEIGHTS_PATH
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            2,
            "--is-embedding",
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

    def test_api_encode_01(self):
        # Test Scenario 1: Call /encode API with plain text parameter
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "2",
                "text": "what is the capital of France",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200,
                    "top_p": 1,
                },
            },
        )
        logger.info("Test 01 response keys: %s", response.json().keys())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["meta_info"]["id"], "2")

    def test_api_encode_02(self):
        # Test Scenario 2: Call /encode API with input_ids parameter
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "3",
                "input_ids": [101, 7592, 2088, 102],
                "sampling_params": {"temperature": 0, "max_new_tokens": 200},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["meta_info"]["id"], "3")

    def test_api_encode_03(self):
        # Test Scenario 3: Call /encode API with text and image parameters (multimodal capability verification)
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "4",
                "text": "show me the words",
                "image_data": "https://miaobi-lite.bj.bcebos.com/miaobi/5mao/b%27b2Ny6K%2BG5Yir5Luj56CBXzE3MzQ2MzcyNjAuMzgxNDk5NQ%3D%3D%27/0.png",
                "sampling_params": {"temperature": 0, "max_new_tokens": 200},
            },
        )
        logger.info("Test 03 response keys: %s", response.json().keys())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["meta_info"]["id"], "4")

    def test_api_encode_04(self):
        # Test Scenario 4: Call /encode API with list of rids (multiple requests) - text input
        request_rids = ["5", "6", "7"]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": request_rids,
                "text": [
                    "what is the capital of UK",
                    "what is the capital of Germany",
                    "what is the capital of Japan",
                ],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200,
                    "top_p": 1,
                },
            },
        )
        response_json = response.json()
        logger.info(
            "Test 04 response type: %s, first item meta_info: %s",
            type(response_json),
            response_json[0].get("meta_info", {}),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response_json), len(request_rids))
        for idx, result in enumerate(response_json):
            self.assertEqual(result["meta_info"]["id"], request_rids[idx])


if __name__ == "__main__":
    unittest.main()
