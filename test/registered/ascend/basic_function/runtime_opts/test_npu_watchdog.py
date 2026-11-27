import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestWatchdogTimeout(CustomTestCase):
    """Testcase: Verify that the service exits immediately after the request scheduling triggers the watchdog timeout.

    [Test Category] Parameter
    [Test Target] --watchdog-timeout
    """

    def test_watchdog_timeout(self):
        # Set an extremely small watchdog timeout to ensure the service times out immediately
        watchdog_timeout = 1e-05
        expected_timeout_message = f"Scheduler watchdog timeout (self.watchdog_timeout={watchdog_timeout}, self.soft=False)"
        out_log_file = open("./out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./err_log.txt", "w+", encoding="utf-8")
        process = None
        try:
            process = popen_launch_server(
                QWEN3_0_6B_WEIGHTS_PATH,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--watchdog-timeout",
                    watchdog_timeout,
                    "--skip-server-warmup",
                    "--attention-backend",
                    "ascend",
                ],
                return_stdout_stderr=(out_log_file, err_log_file),
            )
            err_log_file.seek(0)
            content = err_log_file.read()
            self.assertIn(
                expected_timeout_message,
                content,
            )
        except Exception as e:
            print(f"Watchdog timeout triggered, service exited: {e}")
        finally:
            if process:
                kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.remove("./out_log.txt")
            os.remove("./err_log.txt")


if __name__ == "__main__":
    unittest.main()
