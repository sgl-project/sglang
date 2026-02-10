import unittest
import requests
from sglang.test.ascend.test_ascend_utils import PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestRevision(CustomTestCase):
    """Testcase：Verify set --revision parameter, the inference request is successfully processed.

       [Test Category] Parameter
       [Test Target] --revision
       """
    revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
    model = PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--trust-remote-code",
            ]
        if cls.revision is not None:
            other_args.extend(["--revision", cls.revision])
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _send_request(self):
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
        return response


    def test_revision(self):
        response = self._send_request()
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertEqual(
            response.json()["revision"],
            self.revision,
            "--revision is not taking effect.",
        )


class TestNoRevision(TestRevision):
    """
    unset --revision parameter
    """
    revision = None
    def test_revision(self):
        response = self._send_request()
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIsNone(response.json()["revision"])


if __name__ == "__main__":
    unittest.main()
