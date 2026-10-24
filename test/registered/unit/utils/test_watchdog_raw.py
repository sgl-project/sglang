"""Tests for WatchdogRaw timeout handling in watchdog.py"""

import signal
import threading
import time
import unittest
import unittest.mock

from sglang.srt.utils import watchdog as watchdog_module
from sglang.srt.utils.watchdog import WatchdogRaw
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestWatchdogRaw(CustomTestCase):
    def setUp(self):
        self.sigquit_sent = threading.Event()
        self.parent = unittest.mock.Mock()
        self.parent.send_signal.side_effect = lambda sig: (
            self.sigquit_sent.set() if sig == signal.SIGQUIT else None
        )
        self.release = threading.Event()
        for target, value in [
            ("_DIAGNOSTICS_TIMEOUT_S", 0.5),
            ("_SIGQUIT_DELAY_S", 0.0),
        ]:
            p = unittest.mock.patch.object(watchdog_module, target, value)
            p.start()
            self.addCleanup(p.stop)
        self.pyspy = unittest.mock.patch.object(
            watchdog_module, "pyspy_dump_schedulers"
        ).start()
        self.addCleanup(unittest.mock.patch.stopall)
        self.addCleanup(self.release.set)

    def _start(self, dump_info, soft=False):
        with unittest.mock.patch.object(
            watchdog_module.psutil, "Process"
        ) as process_cls:
            process_cls.return_value.parent.return_value = self.parent
            WatchdogRaw(
                debug_name="test",
                get_counter=lambda: 0,
                is_active=lambda: True,
                watchdog_timeout=0.1,
                soft=soft,
                dump_info=dump_info,
            )

    def test_normal_path(self):
        dump_info = unittest.mock.Mock(return_value="info")
        with self.assertLogs(watchdog_module.logger, level="ERROR") as logs:
            self._start(dump_info)
            self.assertTrue(self.sigquit_sent.wait(timeout=5))
        dump_info.assert_called()
        self.pyspy.assert_called()
        output = "\n".join(logs.output)
        self.assertIn("watchdog timeout", output)
        self.assertIn("debug info:\ninfo", output)

    def test_blocking_diagnostics_still_send_sigquit(self):
        start = time.perf_counter()
        with self.assertLogs(watchdog_module.logger, level="ERROR") as logs:
            self._start(lambda: self.release.wait() and "")
            self.assertTrue(self.sigquit_sent.wait(timeout=5))
        self.assertLess(time.perf_counter() - start, 3)
        output = "\n".join(logs.output)
        self.assertIn("watchdog timeout", output)
        self.assertIn("did not finish", output)

    def test_raising_diagnostics_still_send_sigquit(self):
        def dump_info():
            raise RuntimeError("boom")

        with self.assertLogs(watchdog_module.logger, level="ERROR") as logs:
            self._start(dump_info)
            self.assertTrue(self.sigquit_sent.wait(timeout=5))
        output = "\n".join(logs.output)
        self.assertIn("watchdog timeout", output)
        self.assertIn("diagnostics failed: boom", output)

    def test_soft_watchdog_does_not_send_sigquit(self):
        logged = threading.Event()
        dump_info = unittest.mock.Mock(side_effect=lambda: logged.set() or "")
        self._start(dump_info, soft=True)
        self.assertTrue(logged.wait(timeout=5))
        time.sleep(0.3)
        self.parent.send_signal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
