"""A request above `--return-hidden-states-mode last` gets a 400 in Rust-server mode,
as in the Python frontend, and the server keeps serving."""

import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=100, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    is_rust_server_built(),
    "embedded rust server extension not built",
)
class TestRustServerHiddenStatesMode(CustomTestCase):
    def _generate_above_the_mode(self, rust_server: bool):
        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--return-hidden-states-mode", "last"],
            env={"SGLANG_RUST_SERVER": "1" if rust_server else "0"},
        )
        try:
            body = {
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            }
            refused = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={**body, "return_hidden_states": True},
                timeout=60,
            )
            # The decode CUDA graph is captured for `last`:
            # a `full` request reaching the scheduler raises and takes the server down.
            served = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate", json=body, timeout=60
            )
            return (
                refused.status_code,
                refused.json()["error"]["message"],
                served.status_code,
            )
        finally:
            terminate_and_kill_process_tree(process)

    def test_request_above_the_mode_is_refused_as_in_python(self):
        python = self._generate_above_the_mode(rust_server=False)
        self.assertEqual(python[0], 400)
        self.assertIn("exceeds the server maximum `last`", python[1])
        self.assertEqual(python[2], 200)

        self.assertEqual(self._generate_above_the_mode(rust_server=True), python)


if __name__ == "__main__":
    unittest.main()
