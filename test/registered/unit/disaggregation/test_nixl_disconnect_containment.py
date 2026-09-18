import signal
import threading
import unittest
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.nixl import conn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class RemoteDisconnect(Exception):
    pass


class TestNixlDisconnectContainment(unittest.TestCase):
    def manager(self):
        manager = conn.NixlKVManager.__new__(conn.NixlKVManager)
        manager._disconnect_shutdown_lock = threading.Lock()
        manager._disconnect_shutdown_signaled = False
        manager._scheduler_parent_pid = 12345
        return manager

    def test_only_explicit_disconnect_with_opt_in_signals_parent(self):
        for enabled, error, expected in (
            (False, RemoteDisconnect(), False),
            (True, RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"), False),
            (True, RuntimeError("NIXL_ERR_NOT_FOUND"), False),
            (True, RemoteDisconnect(), True),
        ):
            with self.subTest(enabled=enabled, error=type(error)):
                manager = self.manager()
                with (
                    patch.object(
                        conn, "_NIXL_REMOTE_DISCONNECT_ERRORS", (RemoteDisconnect,)
                    ),
                    patch.object(
                        conn.envs.SGLANG_DISAGGREGATION_NIXL_EXIT_ON_REMOTE_DISCONNECT,
                        "get",
                        return_value=enabled,
                    ),
                    patch.object(conn.os, "kill") as kill,
                ):
                    manager._shutdown_on_remote_disconnect(error)
                    if expected:
                        kill.assert_called_once_with(12345, signal.SIGQUIT)
                    else:
                        kill.assert_not_called()

    def test_concurrent_failures_signal_once(self):
        manager = self.manager()
        with (
            patch.object(conn, "_NIXL_REMOTE_DISCONNECT_ERRORS", (RemoteDisconnect,)),
            patch.object(
                conn.envs.SGLANG_DISAGGREGATION_NIXL_EXIT_ON_REMOTE_DISCONNECT,
                "get",
                return_value=True,
            ),
            patch.object(conn.os, "kill") as kill,
        ):
            threads = [
                threading.Thread(
                    target=manager._shutdown_on_remote_disconnect,
                    args=(RemoteDisconnect(),),
                )
                for _ in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            kill.assert_called_once_with(12345, signal.SIGQUIT)

    def test_polling_disconnect_triggers_shutdown_without_claiming_handles_settled(
        self,
    ):
        manager = self.manager()
        manager.agent = Mock()
        manager.agent.check_xfer_state.side_effect = RemoteDisconnect()
        with (
            patch.object(conn, "_NIXL_REMOTE_DISCONNECT_ERRORS", (RemoteDisconnect,)),
            patch.object(
                conn.envs.SGLANG_DISAGGREGATION_NIXL_EXIT_ON_REMOTE_DISCONNECT,
                "get",
                return_value=True,
            ),
            patch.object(conn.os, "kill") as kill,
        ):
            result = manager._await_handles([object()], failure_seen=False)
            self.assertEqual(result, (False, True))
            kill.assert_called_once_with(12345, signal.SIGQUIT)

    def test_unclassified_polling_error_does_not_shutdown(self):
        manager = self.manager()
        manager.agent = Mock()
        manager.agent.check_xfer_state.side_effect = RuntimeError("unrelated error")
        with (
            patch.object(conn, "_NIXL_REMOTE_DISCONNECT_ERRORS", (RemoteDisconnect,)),
            patch.object(
                conn.envs.SGLANG_DISAGGREGATION_NIXL_EXIT_ON_REMOTE_DISCONNECT,
                "get",
                return_value=True,
            ),
            patch.object(conn.os, "kill") as kill,
        ):
            self.assertEqual(
                manager._await_handles([object()], failure_seen=False), (False, True)
            )
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
