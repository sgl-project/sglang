"""Tests that the hard watchdog still fires when its diagnostics hang or fail."""

import os
import signal
import stat
import subprocess
import tempfile
import threading
import time
import unittest
import unittest.mock

from sglang.srt.utils import cudacore_pyspy_dump_utils as pyspy_module
from sglang.srt.utils import watchdog as watchdog_module
from sglang.srt.utils.watchdog import WatchdogRaw
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestWatchdogRawDiagnostics(CustomTestCase):
    def setUp(self):
        self.sigquit_sent = threading.Event()
        self.parent = unittest.mock.Mock()
        self.parent.send_signal.side_effect = lambda sig: (
            self.sigquit_sent.set() if sig == signal.SIGQUIT else None
        )
        self.release = threading.Event()
        self.stopped = threading.Event()
        for name, value in [("_DIAGNOSTICS_TIMEOUT_S", 0.5), ("_SIGQUIT_DELAY_S", 0.0)]:
            patcher = unittest.mock.patch.object(watchdog_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        pyspy_patcher = unittest.mock.patch.object(
            watchdog_module, "pyspy_dump_schedulers"
        )
        pyspy_patcher.start()
        self.addCleanup(pyspy_patcher.stop)
        self.addCleanup(self.release.set)
        # Leaked watchdog threads would otherwise keep firing into later tests.
        self.addCleanup(self.stopped.set)

    def _start(self, dump_info, soft=False):
        with unittest.mock.patch.object(
            watchdog_module.psutil, "Process"
        ) as process_cls:
            process_cls.return_value.parent.return_value = self.parent
            WatchdogRaw(
                debug_name="test",
                get_counter=lambda: 0,
                is_active=lambda: not self.stopped.is_set(),
                watchdog_timeout=0.1,
                soft=soft,
                dump_info=dump_info,
            )

    def test_sigquit_when_diagnostics_block_forever(self):
        """A hard watchdog must still kill the server if a diagnostic never returns."""

        def blocking_dump_info():
            self.release.wait()
            return ""

        start = time.monotonic()
        self._start(dump_info=blocking_dump_info)
        self.assertTrue(self.sigquit_sent.wait(timeout=5))
        self.assertLess(time.monotonic() - start, 3)

    def test_sigquit_when_diagnostics_raise(self):
        """A failing diagnostic must not end the watchdog without a SIGQUIT."""

        def raising_dump_info():
            raise RuntimeError("boom")

        self._start(dump_info=raising_dump_info)
        self.assertTrue(self.sigquit_sent.wait(timeout=5))

    def test_soft_does_not_signal_or_pile_up_diagnostics(self):
        """A soft watchdog never signals and keeps at most one wedged diagnostic."""
        calls = []

        def blocking_dump_info():
            calls.append(1)
            self.release.wait()
            return ""

        self._start(dump_info=blocking_dump_info, soft=True)
        time.sleep(2)
        self.assertFalse(self.sigquit_sent.is_set())
        self.assertEqual(len(calls), 1)


class TestPyspyDumpBounded(CustomTestCase):
    def test_timeout_is_caught_and_nonblocking_runs_last(self):
        """A hung py-spy must not escape as TimeoutExpired; --nonblocking is the fallback."""
        with unittest.mock.patch.object(pyspy_module.subprocess, "run") as run:
            run.side_effect = subprocess.TimeoutExpired(cmd="py-spy", timeout=1)
            pyspy_module.pyspy_dump_schedulers()
        cmds = [c.args[0] for c in run.call_args_list]
        self.assertEqual(len(cmds), 3)
        self.assertIn("--native", cmds[0])
        self.assertIn("--nonblocking", cmds[-1])
        for c in run.call_args_list:
            self.assertIsNotNone(c.kwargs["timeout"])

    def test_missing_pyspy_binary_is_caught(self):
        with unittest.mock.patch.object(pyspy_module.subprocess, "run") as run:
            run.side_effect = FileNotFoundError("py-spy")
            pyspy_module.pyspy_dump_schedulers()
        self.assertEqual(run.call_count, 1)

    def test_hung_pyspy_is_killed(self):
        """A py-spy that never exits must be killed, not left behind."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "py-spy")
            with open(fake, "w") as f:
                f.write(f"#!/bin/sh\necho $$ >> {tmp}/pids\nexec sleep 30\n")
            os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
            env = {"PATH": f"{tmp}:{os.environ.get('PATH', '')}"}
            with (
                unittest.mock.patch.dict(os.environ, env),
                unittest.mock.patch.object(pyspy_module, "_PYSPY_TIMEOUT_S", 2.0),
            ):
                start = time.monotonic()
                pyspy_module.pyspy_dump_schedulers()
                self.assertLess(time.monotonic() - start, 15)
            with open(os.path.join(tmp, "pids")) as f:
                pids = [int(line) for line in f]
        self.assertGreater(len(pids), 0)
        for pid in pids:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
