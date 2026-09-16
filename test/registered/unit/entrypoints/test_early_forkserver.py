"""Unit tests for `entrypoints/early_forkserver.py` -- the pure helpers only.

The forkserver itself (start method, preload, daemon attach) is process-level
plumbing that CPU CI cannot exercise without launching servers; what is tested
here are the decisions a forked child makes: which environment it runs with,
how the HF offline flags follow that environment, and when a daemon address
file is trusted.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import json
import os
import tempfile
import types
import unittest
from unittest import mock

from sglang.srt.entrypoints import early_forkserver
from sglang.srt.entrypoints.early_forkserver import (
    merge_launcher_env,
    read_daemon_info,
    sync_offline_flags,
)
from sglang.test.test_utils import CustomTestCase


class TestMergeLauncherEnv(CustomTestCase):
    def test_launcher_wins_over_server(self):
        merged = merge_launcher_env(
            current={"A": "server", "B": "server"},
            launcher={"A": "launcher"},
            server_initial={"A": "server", "B": "server"},
        )
        self.assertEqual(merged["A"], "launcher")

    def test_daemon_only_variables_are_dropped(self):
        """A variable the daemon was started with, which the launcher does not
        have, must not leak into the launch (NCCL_CUMEM_ENABLE=0 on the daemon
        once broke a launcher that never set it)."""
        merged = merge_launcher_env(
            current={"NCCL_CUMEM_ENABLE": "0", "PATH": "/bin"},
            launcher={"PATH": "/bin"},
            server_initial={"NCCL_CUMEM_ENABLE": "0", "PATH": "/bin"},
        )
        self.assertNotIn("NCCL_CUMEM_ENABLE", merged)

    def test_import_time_variables_of_the_server_are_kept(self):
        """Worker modules set variables at import inside the server (kernel
        cache dirs); the launcher never imports them, so they survive."""
        merged = merge_launcher_env(
            current={"DG_JIT_CACHE_DIR": "/cache", "PATH": "/bin"},
            launcher={"PATH": "/bin"},
            server_initial={"PATH": "/bin"},
        )
        self.assertEqual(merged["DG_JIT_CACHE_DIR"], "/cache")

    def test_launcher_variables_absent_from_server_are_added(self):
        merged = merge_launcher_env(
            current={"PATH": "/bin"},
            launcher={"PATH": "/bin", "HF_HUB_OFFLINE": "1"},
            server_initial={"PATH": "/bin"},
        )
        self.assertEqual(merged["HF_HUB_OFFLINE"], "1")


class TestSyncOfflineFlags(CustomTestCase):
    def _modules(self):
        hub = types.SimpleNamespace(HF_HUB_OFFLINE=False)
        tf_hub = types.SimpleNamespace(_is_offline_mode=False, HF_HUB_OFFLINE=False)
        return hub, tf_hub

    def test_offline_launcher_turns_flags_on(self):
        hub, tf_hub = self._modules()
        sync_offline_flags(
            environ={"HF_HUB_OFFLINE": "1"},
            modules={
                "huggingface_hub.constants": hub,
                "transformers.utils.hub": tf_hub,
            },
        )
        self.assertTrue(hub.HF_HUB_OFFLINE)
        self.assertTrue(tf_hub._is_offline_mode)
        self.assertTrue(tf_hub.HF_HUB_OFFLINE)

    def test_online_launcher_turns_flags_off(self):
        hub, tf_hub = self._modules()
        hub.HF_HUB_OFFLINE = tf_hub._is_offline_mode = True
        sync_offline_flags(
            environ={},
            modules={
                "huggingface_hub.constants": hub,
                "transformers.utils.hub": tf_hub,
            },
        )
        self.assertFalse(hub.HF_HUB_OFFLINE)
        self.assertFalse(tf_hub._is_offline_mode)

    def test_modules_not_imported_are_skipped(self):
        sync_offline_flags(environ={"HF_HUB_OFFLINE": "1"}, modules={})


class TestReadDaemonInfo(CustomTestCase):
    def _write(self, info):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        self.addCleanup(os.unlink, f.name)
        json.dump(info, f)
        f.close()
        return f.name

    def test_dead_forkserver_is_rejected(self):
        # A pid that cannot exist: os.kill(pid, 0) raises ProcessLookupError.
        path = self._write({"address": "/nonexistent.sock", "pid": 2**22 - 1})
        with self.assertRaises(ProcessLookupError):
            read_daemon_info(path)

    def test_missing_socket_is_rejected(self):
        path = self._write({"address": "/nonexistent.sock", "pid": os.getpid()})
        with self.assertRaises(FileNotFoundError):
            read_daemon_info(path)

    def test_live_daemon_is_accepted(self):
        with tempfile.NamedTemporaryFile() as sock:
            path = self._write(
                {"address": sock.name, "pid": os.getpid(), "daemon_pid": os.getpid()}
            )
            self.assertEqual(
                read_daemon_info(path), {"address": sock.name, "pid": os.getpid()}
            )


class TestSwitches(CustomTestCase):
    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_EARLY_FORKSERVER", None)
            self.assertFalse(early_forkserver.enabled())

    def test_start_early_is_a_noop_when_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_EARLY_FORKSERVER", None)
            with mock.patch("multiprocessing.set_start_method") as set_method:
                early_forkserver.start_early()
            set_method.assert_not_called()


if __name__ == "__main__":
    unittest.main()
