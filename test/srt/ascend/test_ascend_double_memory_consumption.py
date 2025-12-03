"""
Usage:
python3 -m unittest test_ascend_double_memory_consumption.TestMemoryConsumptionAscend.test_memory_consumption
"""

import os
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import subprocess
import requests
import threading

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

import re

from typing import Any, Awaitable, Callable, List, Optional, Tuple

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
    7000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
)
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"

STDERR_FILENAME = "./tmp/stderr.txt"
STDOUT_FILENAME = "./tmp/stdout.txt"

def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass

class TestMemoryConsumptionAscend(CustomTestCase):
        

    def test_memory_consumption(self):
        stdout = open(STDOUT_FILENAME, "w")
        stderr = open(STDERR_FILENAME, "w")

        model = "/mnt/share/weights/Qwen3-32B-w8a8/"
        base_url = DEFAULT_URL_FOR_TEST
        
        output_lines = []
        t = threading.Thread(target=self.read_output, args=(output_lines, ))
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
            if "mem usage" in line:
                mem_usage = re.search(r"mem usage=[+-]?[0-9]+\.[0-9]+ GB")[9:-3]
                self.assertLessEqual(mem_usage, 17.00)
            if "K size" in line:
                k_size = re.search(r"K size=[+-]?[0-9]+\.[0-9]+ GB")[7:-3]
                self.assertLessEqual(k_size, 17.00)
            if "V size" in line:
                v_size = re.search(r"V size=[+-]?[0-9]+\.[0-9]+ GB")[7:-3]
                self.assertLessEqual(v_size, 17.00)
        
        # Clean up everything
        kill_process_tree(process.pid)
        stdout.close()
        stderr.close()
        
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
