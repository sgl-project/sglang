import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH, run_command
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestAscendBasic(CustomTestCase):
    """Without configuring sleep-on-idle, obtain CPU usage."""

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )
        time.sleep(10)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_ascend_sleep_on_idle(self):
        pid = run_command(
            f"ps -ef | grep -E 'sglang::scheduler' | grep -v grep | grep -w {self.process.pid} | tr -s ' '|cut -d' ' -f2"
        )
        self.cpu = run_command(f"ps -p {pid.strip()} -o %cpu --no-headers | xargs")
        self.cpu_float = float(self.cpu.strip())
        run_command(f"echo {self.cpu_float} > ./cpu.txt")

        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

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


class TestSleepOnIdle(CustomTestCase):
    """Testcase: Test configuration --sleep-on-idle reduces CPU utilization.
            Without configuring sleep-on-idle, obtain CPU usage.

    [Test Category] Parameter
    [Test Target] --sleep-on-idle
    """

    cpu_sleep_on_float = 0

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--sleep-on-idle",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )
        time.sleep(10)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        os.remove("./cpu.txt")

    def test_add_sleep_on_idle(self):
        pid_sleep_on = run_command(
            f"ps -ef | grep -E 'sglang::scheduler' | grep -v grep | grep -w {self.process.pid} | tr -s ' '|cut -d' ' -f2"
        )
        self.cpu_sleep_on = run_command(
            f"ps -p {pid_sleep_on.strip()} -o %cpu --no-headers | xargs"
        )
        self.cpu_sleep_on_float = float(self.cpu_sleep_on.strip())

        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

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

    def test_reducation_cpu(self):
        # Comparing CPU usage with and without sleep-on-idle configuration
        cpu_float = float(run_command(f"cat ./cpu.txt"))
        self.assertGreater(
            cpu_float,
            self.cpu_sleep_on_float,
            f"CPU usage should drop with --sleep-on-idle",
        )


if __name__ == "__main__":
    unittest.main()
