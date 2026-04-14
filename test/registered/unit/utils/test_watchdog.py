import os
import signal
import sys
import unittest
from multiprocessing import Process
import time

from sglang.srt.utils.watchdog import Watchdog, SubprocessWatchdog
from sglang.test.test_utils import CustomTestCase, register_cpu_ci

def dummy_worker(sleep_time: float):
    time.sleep(sleep_time)

class TestWatchdog(CustomTestCase):
    def test_watchdog_noop(self):
        """Test Noop watchdog when timeout is set to None."""
        wd = Watchdog.create("test_noop", None)
        # Should not raise exception
        wd.feed()
        with wd.disable():
            pass
        self.assertEqual(wd.__class__.__name__, "_WatchdogNoop")

    def test_watchdog_real(self):
        """Test Real watchdog feeding logic and creation."""
        wd = Watchdog.create("test_real", 1.0)
        self.assertEqual(wd.__class__.__name__, "_WatchdogReal")
        self.assertTrue(wd._active)
        
        # Test basic feed
        initial_counter = wd._counter
        wd.feed()
        self.assertEqual(wd._counter, initial_counter + 1)
        
        # Test disable context manager
        with wd.disable():
            self.assertFalse(wd._active)
        self.assertTrue(wd._active)

class TestSubprocessWatchdog(CustomTestCase):
    def test_subprocess_watchdog_healthy(self):
        """Test that watchdog stays quiet with healthy processes."""
        processes = []
        for _ in range(2):
            p = Process(target=dummy_worker, args=(0.1,))
            p.start()
            processes.append(p)
            
        wd = SubprocessWatchdog(processes, ["p1", "p2"], interval=0.05)
        wd.start()
        
        # Wait for them to finish naturally
        for p in processes:
            p.join()
            
        # Manually invoke check
        crashed = wd._check_processes()
        self.assertFalse(crashed)
        
        wd.stop()

    def test_subprocess_watchdog_stop(self):
        """Test watchdog stop logic terminates the inner thread."""
        p = Process(target=dummy_worker, args=(0.5,))
        p.start()
        
        wd = SubprocessWatchdog([p], ["p_long"], interval=0.05)
        wd.start()
        self.assertIsNotNone(wd._thread)
        self.assertTrue(wd._thread.is_alive())
        
        wd.stop()
        self.assertIsNone(wd._thread)
        
        p.join()

register_cpu_ci(TestWatchdog)
register_cpu_ci(TestSubprocessWatchdog)

if __name__ == "__main__":
    unittest.main()
