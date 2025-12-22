"""
Test soft watchdog functionality for various processes.

Usage:
python -m pytest test/srt/test_soft_watchdog.py -v -s
"""

import os
import subprocess
import sys
import time
import unittest


class TestSoftWatchdog(unittest.TestCase):
    """Test that soft watchdog detects stuck processes and logs warnings."""

    def _run_server_with_watchdog(
        self, env_var_name: str, process_name: str, timeout: float = 60
    ):
        """Run server with watchdog and check for timeout message in logs."""
        from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        env = os.environ.copy()
        env[env_var_name] = "5"  # Sleep 5 seconds in the target process

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            "--port",
            "30000",
            "--soft-watchdog-timeout",
            "2",  # Watchdog timeout 2 seconds (less than sleep)
            "--disable-radix-cache",
        ]

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            output_lines = []
            start_time = time.time()
            server_ready = False
            watchdog_triggered = False

            while time.time() - start_time < timeout:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue

                output_lines.append(line)
                print(line, end="")

                if "Running on" in line or "Application startup complete" in line:
                    server_ready = True
                    # Send a request to trigger the watchdog
                    self._send_test_request()

                if f"{process_name} watchdog timeout" in line:
                    watchdog_triggered = True
                    break

            self.assertTrue(
                watchdog_triggered,
                f"Watchdog timeout message for {process_name} not found in logs. "
                f"Server ready: {server_ready}",
            )

        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def _send_test_request(self):
        """Send a test request to the server."""
        import requests

        time.sleep(1)  # Wait for server to be fully ready

        try:
            response = requests.post(
                "http://localhost:30000/generate",
                json={
                    "text": "Hello",
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                },
                timeout=30,
            )
        except Exception as e:
            print(f"Request failed (expected during watchdog test): {e}")

    def test_detokenizer_soft_watchdog(self):
        """Test that DetokenizerManager soft watchdog triggers on slow processing."""
        self._run_server_with_watchdog(
            "SGLANG_TEST_WATCHDOG_SLOW_DETOKENIZER", "DetokenizerManager"
        )

    def test_tokenizer_soft_watchdog(self):
        """Test that TokenizerManager soft watchdog triggers on slow processing."""
        self._run_server_with_watchdog(
            "SGLANG_TEST_WATCHDOG_SLOW_TOKENIZER", "TokenizerManager"
        )


if __name__ == "__main__":
    unittest.main()

