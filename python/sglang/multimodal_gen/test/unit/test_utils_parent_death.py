import signal
import unittest
from unittest.mock import Mock, patch

from sglang.multimodal_gen import utils


class TestKillItselfWhenParentDied(unittest.TestCase):
    def _run_linux_case(self, parent_pids):
        libc = Mock()
        libc.prctl.return_value = 0

        with (
            patch.object(utils.sys, "platform", "linux"),
            patch.object(utils.ctypes, "CDLL", return_value=libc) as cdll,
            patch.object(utils.os, "getppid", side_effect=parent_pids) as getppid,
            patch.object(utils.os, "getpid", return_value=12345),
            patch.object(utils.os, "kill") as kill,
        ):
            utils.kill_itself_when_parent_died()

        return libc, cdll, getppid, kill

    def test_stable_pid_1_parent_does_not_self_kill(self):
        libc, cdll, getppid, kill = self._run_linux_case([1, 1])

        cdll.assert_called_once_with("libc.so.6", use_errno=True)
        libc.prctl.assert_called_once_with(1, signal.SIGKILL)
        self.assertEqual(getppid.call_count, 2)
        kill.assert_not_called()

    def test_reparent_to_init_self_kills(self):
        _, _, _, kill = self._run_linux_case([1000, 1])

        kill.assert_called_once_with(12345, signal.SIGKILL)

    def test_reparent_to_subreaper_self_kills(self):
        _, _, _, kill = self._run_linux_case([1000, 42])

        kill.assert_called_once_with(12345, signal.SIGKILL)

    def test_non_linux_noop(self):
        with (
            patch.object(utils.sys, "platform", "darwin"),
            patch.object(utils.ctypes, "CDLL") as cdll,
            patch.object(utils.os, "kill") as kill,
        ):
            utils.kill_itself_when_parent_died()

        cdll.assert_not_called()
        kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
