import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=200, stage="base-b", runner_config="1-gpu-large")


class TestQwen3_6Responses(CustomTestCase):
    model = "Qwen/Qwen3.6-27B"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--cuda-graph-max-bs=4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_responses_text_only(self):
        r = requests.post(
            f"{self.base_url}/v1/responses",
            headers={"Content-Type": "application/json"},
            json={"model": self.model, "input": "hello"},
            timeout=120,
        )
        self.assertEqual(r.status_code, 200, msg=r.text)


if __name__ == "__main__":
    unittest.main()
