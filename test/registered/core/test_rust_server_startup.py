"""A failed startup warmup takes the server down, in Rust-server mode
(SGLANG_RUST_SERVER=1) as in the Python frontend, instead of leaving it
unhealthy for its supervisor to discover."""

import os
import subprocess
import sys
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
)

register_cuda_ci(est_time=200, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    is_rust_server_built(),
    "embedded rust server extension not built",
)
class TestRustServerStartupWarmupFailure(CustomTestCase):
    def _launch_until_exit(self, rust_server: bool):
        host, port = DEFAULT_URL_FOR_TEST.removeprefix("http://").rsplit(":", 1)
        # A context window shorter than the warmup prompt fails the warmup.
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            "--host",
            host,
            "--port",
            port,
            "--context-length",
            "4",
        ]
        process = subprocess.Popen(
            command,
            env={**os.environ, "SGLANG_RUST_SERVER": "1" if rust_server else "0"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            output, _ = process.communicate(timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)
        except subprocess.TimeoutExpired:
            kill_process_tree(process.pid)
            self.fail(f"server (rust_server={rust_server}) outlived a failed warmup")
        return process.returncode, output

    def test_failed_warmup_exits_as_in_python(self):
        python_code, python_output = self._launch_until_exit(rust_server=False)
        self.assertIn("Initialization failed. warmup error", python_output)
        self.assertNotEqual(python_code, 0)

        rust_code, rust_output = self._launch_until_exit(rust_server=True)
        self.assertIn("Initialization failed. warmup error", rust_output)
        self.assertEqual(rust_code, python_code, rust_output[-4000:])


if __name__ == "__main__":
    unittest.main()
