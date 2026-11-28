"""Unit tests for signal-safe stderr writes in srt/utils/process.py."""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from sglang.srt.utils.process import kill_process_tree, write_stderr_signal_safe
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

_REENTRANT = "reentrant call inside <_io.BufferedWriter name='<stderr>'>"


class _ReentrantStderr:
    def write(self, _):
        raise RuntimeError(_REENTRANT)

    def flush(self):
        raise RuntimeError(_REENTRANT)


class TestWriteStderrSignalSafe(CustomTestCase):
    def test_writes_to_fd2_without_sys_stderr(self):
        with tempfile.TemporaryFile() as captured:
            saved_fd2 = os.dup(2)
            try:
                os.dup2(captured.fileno(), 2)
                with mock.patch.object(sys, "stderr", _ReentrantStderr()):
                    write_stderr_signal_safe("child failed")
            finally:
                os.dup2(saved_fd2, 2)
                os.close(saved_fd2)
            captured.seek(0)
            self.assertEqual(captured.read(), b"child failed\n")


class TestKillProcessTreeWhenLoggingReenters(CustomTestCase):
    def test_kills_even_if_logging_raises_reentrant_error(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with mock.patch(
                "sglang.srt.utils.process.logger.info",
                side_effect=RuntimeError(_REENTRANT),
            ):
                kill_process_tree(child.pid, include_parent=True, wait_timeout=30)
            self.assertIsNotNone(child.wait(timeout=10))
        finally:
            if child.poll() is None:
                child.kill()


if __name__ == "__main__":
    unittest.main()
