"""
Usage:
python3 -m unittest test_ascend_double_memory_consumption.TestMemoryConsumptionAscend.test_memory_consumption
"""

import os
import re
import threading
import time
import unittest
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
    7000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
)
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"

STDERR_FILENAME = "/tmp/stderr.txt"
STDOUT_FILENAME = "/tmp/stdout.txt"


class TestMemoryConsumptionAscend(CustomTestCase):

    def test_memory_consumption(self):
        stdout = open(STDOUT_FILENAME, "w")
        stderr = open(STDERR_FILENAME, "w")

        model = "/mnt/share/weights/Qwen3-32B-w8a8/"
        base_url = DEFAULT_URL_FOR_TEST

        output_lines = []
        t = threading.Thread(target=self.read_output, args=(output_lines,))
        t.start()

        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            return_stdout_stderr=(stdout, stderr),
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
                "64",
                "--disable-radix-cache",
            ],
        )

        # Launch a thread to stream the output
        for line in output_lines:
            if "Load weight end" in line and "mem usage" in line:
                match = re.search(r"mem usage=[+-]?[0-9]+\.[0-9]+ GB", line)
                self.assertLessEqual(float(match.group(0)[10:-3]), 17.00)
            if "KV Cache is allocated" in line and "K size" in line:
                match = re.search(r"K size: [+-]?[0-9]+\.[0-9]+ GB", line)
                self.assertLessEqual(float(match.group(0)[8:-3]), 17.00)
            if "KV Cache is allocated" in line and "V size" in line:
                match = re.search(r"V size: [+-]?[0-9]+\.[0-9]+ GB", line)
                self.assertLessEqual(float(match.group(0)[8:-3]), 17.00)

        # Clean up everything
        kill_process_tree(process.pid)
        stdout.close()
        stderr.close()
        if os.path.exists(STDOUT_FILENAME):
            os.remove(STDOUT_FILENAME)
        if os.path.exists(STDERR_FILENAME):
            os.remove(STDERR_FILENAME)
        t.join()

    def read_output(self, output_lines: List[str], filename: str = STDERR_FILENAME):
        """Print the output in real time with another thread."""

        while not os.path.exists(filename):
            time.sleep(0.01)

        pt = 0
        while pt >= 0:
            if pt > 0 and not os.path.exists(filename):
                break
            try:
                lines = open(filename).readlines()
            except FileNotFoundError:
                print(f"{pt=}, {os.path.exists(filename)=}")
                raise
            for line in lines[pt:]:
                print(line, end="", flush=True)
                output_lines.append(line)
                pt += 1
            time.sleep(0.1)


if __name__ == "__main__":
    unittest.main()
