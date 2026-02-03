import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH, run_command, get_device_ids
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestGpuId(CustomTestCase):
    """Testcase: Verify that the --base-gpu-id and --gpu-id-step parameters can correctly control the NPU devices occupied during model loading.

    [Test Category] Parameter
    [Test Target] --base-gpu-id; --gpu-id-step
    """
    @classmethod
    def setUpClass(cls):
        cls.device_id = get_device_ids(0)
        cls.step = 2
        other_args = (
            [
                "--base-gpu-id", # Starting device ID
                cls.device_id,
                "--tp-size",  # tp = 2 (2 devices required for occupation)
                "2",
                "--gpu-id-step", # Device ID step size = 2 (occupies the starting device ID and starting device ID + 2)
                cls.step,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gpu_id(self):
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
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        # Filter out the memory usage values of each device
        result = run_command(
           "npu-smi info | grep '/ 65536' | awk -F '|' '{print $4}' | awk '{print $5}' | awk -F '/' '{print $1}'"
        ).split("\n")
        for i in range(len(result)-1):
            # Occupied devices show high memory usage.
            if i in [self.device_id, self.device_id + self.step]:
                self.assertGreater(int(result[i]), 10000)
            # Unoccupied devices show low memory usage.
            else:
                self.assertLess(int(result[i]), 5000)


if __name__ == "__main__":
    unittest.main()
