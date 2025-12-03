"""
Usage:
python3 -m unittest test_ascend_double_memory_consumption.TestMemoryConsumptionAscend.test_memory_consumption
"""

import os
import re
import threading
import time
import unittest
import torch
from typing import List

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
    8000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
)
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"

STDERR_FILENAME = "/tmp/stderr.txt"
STDOUT_FILENAME = "/tmp/stdout.txt"


class TestMemoryConsumptionAscend(CustomTestCase):

    def test_memory_consumption(self):

        model = "/mnt/share/weights/msit_w8a8_static_dynamic_Qwen3-30B_no_anti_outlier/"
        base_url = DEFAULT_URL_FOR_TEST

        ### Calculate initial used memory
        free_npu_memory, total_npu_memory = torch.npu.mem_get_info()
        initial_used_memory = (total_npu_memory - free_npu_memory)

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
                "--quantization",
                "w8a8_int8",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-bs",
                "1",
                "--disable-radix-cache",
                "--max-total-tokens",
                "1024",
                "--disable-cuda-graph"
            ],
        )
        
        
        ### Calculate initial used memory
        free_npu_memory, total_npu_memory = torch.npu.mem_get_info()
        used_memory_after_server_starting = (total_npu_memory - free_npu_memory - initial_used_memory) / (1 << 30)
        print(used_memory_after_server_starting)
        self.assertLessEqual(float(used_memory_after_server_starting), 17.00)

        # Clean up everything
        kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
