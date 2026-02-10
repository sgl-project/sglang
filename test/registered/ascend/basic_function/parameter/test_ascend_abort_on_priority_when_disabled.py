import unittest
import logging

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestAbortOnPriority(CustomTestCase):
    """Testcase: Verify the effectiveness of --abort-on-priority-when-disabled parameter while sending request.

    [Test Category] Parameter
    [Test Target] --abort-on-priority-when-disabled
    """

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--abort-on-priority-when-disabled",
        ]

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_abort_on_priority_when_disabled(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "priority": 2,
                },
            },
        )
        logger.info(response.text)

        self.assertEqual(
            response.status_code, 500, "The request status code is not 500."
        )

        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        logger.info(response.json())

        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertTrue(
            response.json()["abort_on_priority_when_disabled"],
            "abort_on_priority_when_disabled is not taking effect.",
        )


if __name__ == "__main__":
    unittest.main()
