import os
import sys
import time
import unittest
from datetime import datetime

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

LOG_DUMP_FILE = f"server_request_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
CUSTOM_SERVER_WAIT_TIME = 20

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestEnableRequestTimeStatsLogging(CustomTestCase):
    """Testcase: Verify the functionality of --enable-request-time-stats-logging to generate Req Time Stats logs on Ascend backend with Llama-3.2-1B-Instruct model.

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging
    """

    @classmethod
    def setUpClass(cls):
        # 1. Save the original stdout/stderr file descriptors at the operating system level
        cls.original_stdout_fd = os.dup(sys.stdout.fileno())
        cls.original_stderr_fd = os.dup(sys.stderr.fileno())

        # 2. Open the log file (OS-level file descriptor for redirection)
        cls.log_fd = os.open(
            LOG_DUMP_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
        )
        cls.log_file = open(LOG_DUMP_FILE, "a+", encoding="utf-8")

        # 3. Redirect stdout and stderr to the log file descriptor at the operating system level
        os.dup2(cls.log_fd, sys.stdout.fileno())
        os.dup2(cls.log_fd, sys.stderr.fileno())

        # 4. Launch the model server with specified configuration
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-request-time-stats-logging",
        ]

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        print(f"Waiting for server to start ({CUSTOM_SERVER_WAIT_TIME} seconds)...")
        time.sleep(CUSTOM_SERVER_WAIT_TIME)

    @classmethod
    def tearDownClass(cls):
        # 1. Terminate the server process tree
        kill_process_tree(cls.process.pid)

        # 2. Restore the original stdout/stderr at the operating system level (for subsequent log printing and prompts)
        os.dup2(cls.original_stdout_fd, sys.stdout.fileno())
        os.dup2(cls.original_stderr_fd, sys.stderr.fileno())

        # 3. Close all file descriptors and file objects (release file occupation)
        os.close(cls.log_fd)
        os.close(cls.original_stdout_fd)
        os.close(cls.original_stderr_fd)
        cls.log_file.close()

        # 4. Print the full log content to the console
        cls.print_full_log()

        # 5. Delete the log file (clean up redundant files)
        cls.delete_log_file()

    @classmethod
    def print_full_log(cls):
        if not os.path.exists(LOG_DUMP_FILE):
            print("\n[Log Tip] Log file does not exist, no content to print")
            return

        print("\n" + "=" * 80)
        print("Full Server Log Content:")
        print("=" * 80)
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            full_log = f.read()
            # Print full log (if the log is too large, only print the last 5000 characters to avoid console flooding)
            print(
                full_log
                if len(full_log) <= 5000
                else f"[Log Too Long, Only Showing Last 5000 Characters]\n{full_log[-5000:]}"
            )
        print("=" * 80)
        print("Log printing completed")

    @classmethod
    def delete_log_file(cls):
        try:
            if os.path.exists(LOG_DUMP_FILE):
                os.remove(LOG_DUMP_FILE)
                print(f"\nLog file deleted: {os.path.abspath(LOG_DUMP_FILE)}")
            else:
                print("\n[Deletion Tip] Log file does not exist, no need to delete")
        except Exception as e:
            print(f"\n[Deletion Warning] Failed to delete log file: {e}")

    def read_log_file(self):
        if not os.path.exists(LOG_DUMP_FILE):
            return ""

        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def test_enable_request_time_stats_logging(self):
        # 1. Send a request to trigger the server to generate Req Time Stats logs
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

        # 2. Extend the log writing waiting time to ensure Req Time Stats is fully written
        time.sleep(5)

        # 3. Restore IO to facilitate subsequent assertion information output to the console
        os.dup2(self.original_stdout_fd, sys.stdout.fileno())
        os.dup2(self.original_stderr_fd, sys.stderr.fileno())

        # 4. Assert that the request was sent successfully
        self.assertEqual(response.status_code, 200, "Failed to call generate API")

        # 5. Read the full content of the log file
        server_logs = self.read_log_file()

        # 6. Assert that the log contains the Req Time Stats keyword
        target_keyword = "Req Time Stats"
        self.assertIn(
            target_keyword,
            server_logs,
            f"Keyword not found in server logs: {target_keyword}\nLog file path: {os.path.abspath(LOG_DUMP_FILE)}\nLog content preview (last 2000 characters):\n{server_logs[-2000:] if len(server_logs) > 2000 else server_logs}",
        )


if __name__ == "__main__":
    unittest.main()
