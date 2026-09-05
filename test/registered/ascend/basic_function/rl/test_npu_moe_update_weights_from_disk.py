import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=120, suite="stage-b-test-1-npu-a2")


class TestNPUMoEUpdateWeightsFromDisk(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.95",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--max-running-requests",
                "8",
                "--tp-size",
                "1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def _generate(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["text"]

    def test_idempotent_moe_weight_reload(self):
        expected = self._generate()
        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": self.model, "flush_cache": True},
            timeout=120,
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["success"], response.text)
        self.assertEqual(self._generate(), expected)


if __name__ == "__main__":
    unittest.main()
