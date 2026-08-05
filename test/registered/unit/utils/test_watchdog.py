import time
import unittest
from multiprocessing import Process
from unittest.mock import MagicMock, patch

from sglang.srt.utils.watchdog import SubprocessWatchdog, Watchdog
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


def dummy_worker(sleep_time: float):
    time.sleep(sleep_time)


class TestWatchdog(CustomTestCase):
    def test_watchdog_noop(self):
        """Test Noop watchdog when timeout is set to None."""
        wd = Watchdog.create("test_noop", None)
        wd.feed()
        with wd.disable():
            pass
        self.assertEqual(wd.__class__.__name__, "_WatchdogNoop")

    @patch("sglang.srt.utils.watchdog.WatchdogRaw")
    def test_watchdog_real(self, mock_raw_cls):
        """Test _WatchdogReal creation and feed/disable logic without leaking threads.

        We mock WatchdogRaw to prevent a real background thread from starting,
        which would eventually trigger pyspy_dump_schedulers() and write error
        logs after the test finishes.
        """
        mock_raw_cls.return_value = MagicMock(debug_name="test_real")

        wd = Watchdog.create("test_real", 1.0, soft=True)
        self.assertEqual(wd.__class__.__name__, "_WatchdogReal")
        self.assertTrue(wd._active)

        # Test basic feed increments counter
        initial_counter = wd._counter
        wd.feed()
        self.assertEqual(wd._counter, initial_counter + 1)

        # Test disable context manager
        with wd.disable():
            self.assertFalse(wd._active)
        self.assertTrue(wd._active)


class TestSubprocessWatchdog(CustomTestCase):
    def setUp(self):
        self._processes = []
        self._watchdog = None

    def tearDown(self):
        if self._watchdog is not None:
            self._watchdog.stop()
        for p in self._processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

    def test_subprocess_watchdog_healthy(self):
        """Test that watchdog stays quiet with healthy processes."""
        processes = []
        for _ in range(2):
            p = Process(target=dummy_worker, args=(0.1,))
            p.start()
            processes.append(p)

        self._processes = processes
        wd = SubprocessWatchdog(processes, ["p1", "p2"], interval=0.05)
        self._watchdog = wd
        wd.start()

        for p in processes:
            p.join()

        crashed = wd._check_processes()
        self.assertFalse(crashed)

    def test_subprocess_watchdog_stop(self):
        """Test watchdog stop logic terminates the inner thread."""
        p = Process(target=dummy_worker, args=(0.5,))
        p.start()
        self._processes = [p]

        wd = SubprocessWatchdog([p], ["p_long"], interval=0.05)
        self._watchdog = wd
        wd.start()
        self.assertIsNotNone(wd._thread)
        self.assertTrue(wd._thread.is_alive())

        wd.stop()
        self._watchdog = None  # already stopped
        self.assertIsNone(wd._thread)


if __name__ == "__main__":
    unittest.main()
