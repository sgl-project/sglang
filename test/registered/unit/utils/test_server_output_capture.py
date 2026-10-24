"""Output capture must not certify an incomplete or failed server log."""

import importlib.util
import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    _subprocess_popen_with_outputs,
    wait_for_server_output,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestServerOutputCapture(CustomTestCase):
    def test_late_fallback_is_drained_after_process_exit(self):
        blocked = threading.Event()
        release = threading.Event()

        class DelayedSink(io.StringIO):
            def write(self, text):
                if "False" in text:
                    blocked.set()
                    if not release.wait(timeout=10):
                        raise TimeoutError("Test did not release the delayed sink")
                return super().write(text)

        log = DelayedSink()
        with patch("torch.cuda.is_initialized", return_value=False):
            process = _subprocess_popen_with_outputs(
                [
                    sys.executable,
                    "-c",
                    "print('[PP0] Decode batch, cuda graph: True'); "
                    "print('[PP1] Decode batch, cuda graph: True'); "
                    "print('[PP1] Decode batch, cuda graph: False')",
                ],
                None,
                (log, log),
            )
        try:
            self.assertTrue(blocked.wait(timeout=5))
            process.wait(timeout=5)
            self.assertNotIn("False", log.getvalue())
            with self.assertRaises(TimeoutError):
                wait_for_server_output(process, timeout=0)
            release.set()
            wait_for_server_output(process)
            self.assertIn("cuda graph: False", log.getvalue())
        finally:
            release.set()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            wait_for_server_output(process)

    def test_pp_assertion_rejects_fallback_delivered_during_shutdown(self):
        # Exercise the actual GPU test's lifecycle without launching a model.
        path = Path(__file__).resolve().parents[2] / "e2e/pp/test_pp_cpu_transport.py"
        spec = importlib.util.spec_from_file_location("pp_cpu_transport_e2e", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        case = module.TestPPCPUTransport()
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "meta_info": {"completion_tokens": 32},
                "output_ids": list(range(32)),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "server.log"
            with log_path.open("w") as log:

                def generate(*args, **kwargs):
                    log.write("[PP0] Decode batch, cuda graph: True\n")
                    log.write("[PP1] Decode batch, cuda graph: True\n")
                    log.flush()
                    return response

                def shutdown(process):
                    # The final stats line arrives after the HTTP response.
                    log.write("[PP1] Decode batch, cuda graph: False\n")
                    log.flush()

                with (
                    patch.object(
                        module, "popen_launch_server", return_value=SimpleNamespace()
                    ),
                    patch.object(module.requests, "post", side_effect=generate),
                    patch.object(
                        module, "terminate_and_kill_process_tree", side_effect=shutdown
                    ),
                ):
                    with self.assertRaisesRegex(AssertionError, "decode fell back"):
                        case._generate_with_log("full", log_path, log)

    def test_sink_error_is_not_treated_as_complete_capture(self):
        class BrokenSink(io.StringIO):
            def write(self, text):
                raise OSError("capture disk full")

        with patch("torch.cuda.is_initialized", return_value=False):
            process = _subprocess_popen_with_outputs(
                [sys.executable, "-c", "print('decode result')"],
                None,
                (BrokenSink(), io.StringIO()),
            )
        process.wait(timeout=5)
        with self.assertRaisesRegex(RuntimeError, "capture failed") as caught:
            wait_for_server_output(process)
        self.assertIsInstance(caught.exception.__cause__, OSError)


if __name__ == "__main__":
    unittest.main()
