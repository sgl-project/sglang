import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestRevisionDefault(CustomTestCase):
    revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"

    @classmethod
    def setUpClass(cls):
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--trust-remote-code",
            ]
            if is_npu()
            else ["--trust-remote-code"]
        )

        cls.process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/microsoft/Phi-4-multimodal-instruct"
                if is_npu()
                else "microsoft/Phi-4-multimodal-instruct"
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_revision_default(self):
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
        self.assertIsNone(response.json()["revision"])


class TestRevision(CustomTestCase):
    revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"

    @classmethod
    def setUpClass(cls):
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--trust-remote-code",
                "--revision",
                cls.revision,
            ]
            if is_npu()
            else ["--trust-remote-code", "--revision", cls.revision]
        )

        cls.process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/microsoft/Phi-4-multimodal-instruct"
                if is_npu()
                else "microsoft/Phi-4-multimodal-instruct"
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_revision(self):
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
        self.assertEqual(
            response.json()["revision"],
            self.revision,
            "--revision is not taking effect.",
        )


if __name__ == "__main__":
    unittest.main()
