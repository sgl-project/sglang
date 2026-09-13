import unittest

import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")


class TestNearKVAdmission(CustomTestCase):
    """Exercise both sides of the near-capacity admission boundary."""

    max_total_tokens = 512
    page_size = 64

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--max-total-tokens",
                str(cls.max_total_tokens),
                "--page-size",
                str(cls.page_size),
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_near_kv_boundary_does_not_block_followup(self):
        # 447 + one allocator page is still below capacity. Its generation
        # budget is clipped to zero, but it must enter a runnable batch and
        # finish rather than poisoning the next scheduler iteration.
        admitted = requests.post(
            self.base_url + "/generate",
            json={
                "rid": "near-kv-admitted",
                "input_ids": [1] * (self.max_total_tokens - self.page_size - 1),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": self.page_size,
                    "ignore_eos": True,
                },
            },
            timeout=30,
        )
        self.assertEqual(admitted.status_code, 200, admitted.text)
        self.assertEqual(
            admitted.json()["meta_info"]["prompt_tokens"],
            self.max_total_tokens - self.page_size - 1,
        )
        self.assertEqual(admitted.json()["meta_info"]["completion_tokens"], 0)

        # 448 + one allocator page is exactly capacity. PrefillAdder cannot
        # admit this request, so the public length check must reject it before
        # it can enter the waiting queue with max_new_tokens clipped to zero.
        response = requests.post(
            self.base_url + "/generate",
            json={
                "rid": "near-kv-rejected",
                "input_ids": [1] * (self.max_total_tokens - self.page_size),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": self.page_size,
                    "ignore_eos": True,
                },
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Input length", response.json()["error"]["message"])

        follow_up = requests.post(
            self.base_url + "/generate",
            json={
                "rid": "near-kv-admission-followup",
                "input_ids": [1] * 8,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
            },
            timeout=30,
        )
        self.assertEqual(follow_up.status_code, 200, follow_up.text)
        self.assertEqual(follow_up.json()["meta_info"]["completion_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
