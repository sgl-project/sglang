import os
import shutil
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestAscendDeleteCkptAfterLoading(CustomTestCase):
    """
    Testcase：Verify the weight directory is deleted after loading and when --delete-ckpt-after-loading is set

    [Test Category] Parameter
    [Test Target] --delete-ckpt-after-loading
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
        cls.back_up_model_path = cls.model + "-back-up"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--delete-ckpt-after-loading",
        ]

        if not os.path.exists(cls.back_up_model_path):
            shutil.copytree(cls.model, cls.back_up_model_path)

        cls.process = popen_launch_server(
            cls.back_up_model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *cls.common_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if os.path.exists(cls.back_up_model_path):
            shutil.rmtree(cls.back_up_model_path)

    def test_delete_ckpt_after_loading(self):
        response = requests.post(
            f"{self.base_url}/generate",
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

        response = requests.get(f"{self.base_url}/get_server_info")
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertTrue(
            response.json()["delete_ckpt_after_loading"],
            "--delete-ckpt-after-loading is not taking effect.",
        )

        # Verify the weight directory is deleted after loading
        self.assertFalse(
            os.path.exists(self.back_up_model_path),
            "--delete-ckpt-after-loading is not taking effect.",
        )


if __name__ == "__main__":
    unittest.main()
