"""Test --smg-grpc-mode parameter on NPU.
Verifies that the legacy SMG gRPC server starts correctly.
Logs are printed to screen in real-time (tee) while also captured for assertions.
"""

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

# Expected log markers for a successful SMG gRPC startup
_GRPC_READY_MARKERS = [
    "Standard gRPC health service initialized",
    "gRPC scheduler servicer initialized",
    "gRPC server listening on",
    "Sending warmup request to gRPC server...",
    "gRPC warmup request completed successfully",
    "Health service status set to SERVING",
    "The server is fired up and ready to roll!",
]


def _tee_output(pipe, log_file):
    """Read from pipe, print to screen, and write to log_file."""
    for line in iter(pipe.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()


class TestNpuSmgGrpcMode(unittest.TestCase):
    """Testcase: --smg-grpc-mode starts the legacy SMG gRPC server on NPU.

    [Test Category] Functional
    [Test Target] Legacy SMG gRPC server mode on NPU
    --smg-grpc-mode;
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.grpc_port = 30001  # From log: gRPC server listening on 127.0.0.1:30001

        command = [
            "sglang",
            "serve",
            "--model-path",
            cls.model,
            "--tokenizer-path",
            cls.model,
            "--device",
            "npu",
            "--trust-remote-code",
            "--log-requests-level",
            "2",
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
            "--disable-radix-cache",
            "--enable-metrics",
            "--smg-grpc-mode",
        ]

        cls.out_log = tempfile.NamedTemporaryFile(
            mode="w+", suffix="_smg.log", delete=False
        )
        cls.err_log = tempfile.NamedTemporaryFile(
            mode="w+", suffix="_smg_err.log", delete=False
        )

        cls.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Tee stdout/stderr to screen + file in background threads
        cls._stdout_thread = threading.Thread(
            target=_tee_output, args=(cls.process.stdout, cls.out_log), daemon=True
        )
        cls._stderr_thread = threading.Thread(
            target=_tee_output, args=(cls.process.stderr, cls.err_log), daemon=True
        )
        cls._stdout_thread.start()
        cls._stderr_thread.start()

        # Wait for startup by polling logs (max 600s)
        cls._wait_for_ready(timeout=600)

    @classmethod
    def tearDownClass(cls):
        if cls.process and cls.process.poll() is None:
            kill_process_tree(cls.process.pid)
        for f in (cls.out_log, cls.err_log):
            try:
                f.close()
            except Exception:
                pass
        for path in (cls.out_log.name, cls.err_log.name):
            if os.path.exists(path):
                os.remove(path)

    @classmethod
    def _read_combined_logs(cls):
        cls.out_log.flush()
        cls.err_log.flush()
        cls.out_log.seek(0)
        cls.err_log.seek(0)
        return cls.out_log.read() + cls.err_log.read()

    @classmethod
    def _wait_for_ready(cls, timeout=600, interval=5):
        """Poll logs until all readiness markers are found."""
        for _ in range(timeout // interval):
            logs = cls._read_combined_logs()
            if all(marker in logs for marker in _GRPC_READY_MARKERS):
                print("\n[setUpClass] All gRPC readiness markers found.")
                return
            if cls.process.poll() is not None:
                # Wait a moment for tee threads to flush remaining output
                time.sleep(1)
                raise RuntimeError(
                    f"Server process exited early with code {cls.process.poll()}."
                )
            time.sleep(interval)
        raise TimeoutError(f"gRPC server did not become ready within {timeout}s.")

    def test_smg_grpc_server_starts_and_ready(self):
        """Verify SMG gRPC mode starts successfully and reports ready."""
        logs = self._read_combined_logs()

        # Process alive
        self.assertIsNone(
            self.process.poll(),
            "Server process crashed after startup",
        )

        # All readiness markers present
        for marker in _GRPC_READY_MARKERS:
            self.assertIn(
                marker,
                logs,
                f"Missing expected log marker: {marker}",
            )

    def test_smg_grpc_port_listening(self):
        """Verify the SMG gRPC port is actually open."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            result = sock.connect_ex(("127.0.0.1", self.grpc_port))
            self.assertEqual(
                result,
                0,
                f"SMG gRPC port {self.grpc_port} is not listening",
            )
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
