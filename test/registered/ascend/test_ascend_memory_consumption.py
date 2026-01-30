"""
Usage:
python3 -m unittest test_ascend_memory_consumption.TestMemoryConsumptionAscend.test_memory_consumption
"""

import os
import unittest

import torch

from sglang.srt.utils import kill_process_tree
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

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
    8000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
)
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"


class TestMemoryConsumptionAscend(CustomTestCase):

    def test_memory_consumption(self):

        model = "nytopop/Qwen3-30B-A3B.w8a8"
        base_url = DEFAULT_URL_FOR_TEST

        ### Calculate initial used memory
        free_npu_memory, total_npu_memory = torch.npu.mem_get_info()
        initial_used_memory = total_npu_memory - free_npu_memory

        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-bs",
                "1",
                "--max-total-tokens",
                "1024",
                "--disable-radix-cache",
                "--disable-cuda-graph",
            ],
        )

        ### Calculate initial used memory
        free_npu_memory, total_npu_memory = torch.npu.mem_get_info()
        used_memory_after_server_starting = (
            total_npu_memory - free_npu_memory - initial_used_memory
        ) / (1 << 30)
        self.assertLessEqual(float(used_memory_after_server_starting), 16.00)

        # Clean up everything
        kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
