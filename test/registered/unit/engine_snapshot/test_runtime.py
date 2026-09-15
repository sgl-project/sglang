import contextlib
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from snapshot_fixtures import ROOT_PID, SnapshotArtifacts

import sglang.srt.engine_snapshot.runtime as runtime_module
from sglang.srt.engine_snapshot import control
from sglang.srt.engine_snapshot.errors import (
    SnapshotRuntimeFailure,
    SnapshotSecurityError,
    SnapshotUsageError,
)
from sglang.srt.engine_snapshot.manifest import SnapshotFile
from sglang.srt.engine_snapshot.runtime import EngineInventory, SnapshotRuntime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The restore transaction itself (CRIU, cuda-checkpoint, pidfd pinning, the
# release handshake) needs a real engine and is covered by the GPU end-to-end
# run of `sglang snapshot create|restore`; these tests cover the parts that
# decide what the runtime is allowed to touch, and how it reports failures.


def inventory(**overrides):
    """A captured engine tree, the way `create` records it in the manifest."""
    fields = dict(
        root_pid=ROOT_PID,
        pids=[ROOT_PID],
        cuda_pids=[ROOT_PID],
        gpu_uuid="GPU-1",
        shared_paths=set(),
    )
    fields.update(overrides)
    return EngineInventory(**fields)


class TestSnapshotRuntime(SnapshotArtifacts, CustomTestCase):
    DIRECTORIES = ("control", "files", "runtime", "shm", "dev_shm", "work")

    def setUp(self):
        super().setUp()
        self.runtime = SnapshotRuntime()
        shm = patch.object(SnapshotRuntime, "SHM_DIR", self.artifact_path / "shm")
        shm.start()
        self.addCleanup(shm.stop)

    def patched(self, **overrides):
        """Stand in for the host probes so inventory runs without an engine."""
        fields = dict(
            tree_pids=[1],
            cuda_holders=([1], "GPU-1"),
            io_uring_pids=[],
            referenced_paths=set(),
        )
        fields.update(overrides)
        stack = contextlib.ExitStack()
        for name in fields:
            stack.enter_context(
                patch.object(self.runtime, name, return_value=fields[name])
            )
        return stack

    # ------------------------------------------------------------------ #
    # preflight and capture-side checks
    # ------------------------------------------------------------------ #
    def test_preflight_reports_missing_tools_and_pidfd(self):
        with patch.object(runtime_module.shutil, "which", return_value=None):
            with self.assertRaises(SnapshotRuntimeFailure) as error:
                self.runtime.preflight("create", self.artifact_path)
        for tool in ("criu", "cuda-checkpoint", "nvidia-smi"):
            self.assertIn(tool, str(error.exception))

        with (
            patch.object(runtime_module.shutil, "which", return_value="/bin/true"),
            patch.object(runtime_module.os, "pidfd_open", None, create=True),
        ):
            with self.assertRaisesRegex(SnapshotRuntimeFailure, "pidfd"):
                self.runtime.preflight("restore", self.artifact_path)
            self.runtime.preflight("create", self.artifact_path)

    def test_cuda_holders_require_exactly_one_gpu(self):
        rows = "99999991, GPU-1\n99999992, GPU-1\n1234, GPU-2\n"

        # The engine root process holds an nvidia mapping without holding GPU memory,
        # so the mapping probe has to be part of the holder set.
        def mapped(pid):
            return {"/dev/nvidia4"} if pid == 99999993 else set()

        with (
            patch.object(self.runtime, "_capture", return_value=rows),
            patch.object(self.runtime, "_mapped_paths", side_effect=mapped),
        ):
            holders, gpu_uuid = self.runtime.cuda_holders(
                [99999991, 99999992, 99999993]
            )
        self.assertEqual(holders, [99999991, 99999992, 99999993])
        self.assertEqual(gpu_uuid, "GPU-1")

        for rows, expected in (
            ("99999991, GPU-1\n99999992, GPU-2\n", "2 GPUs"),
            ("1234, GPU-1\n", "holds GPU memory"),
        ):
            with self.subTest(rows=rows):
                with (
                    patch.object(self.runtime, "_capture", return_value=rows),
                    patch.object(self.runtime, "_mapped_paths", return_value=set()),
                ):
                    with self.assertRaisesRegex(SnapshotRuntimeFailure, expected):
                        self.runtime.cuda_holders([99999991, 99999992])

    def test_inventory_collects_pids_and_referenced_dev_shm(self):
        shared = self.artifact_path / "shm/load-state"
        unrelated = self.artifact_path / "shm/other-engine"
        shared.write_bytes(b"state")
        unrelated.write_bytes(b"other")
        with self.patched(
            referenced_paths={
                str(shared),
                str(unrelated),
                str(self.artifact_path / "runtime/x"),
            }
        ):
            captured = self.runtime.inventory(self.artifact_path, 1, "/model", "GPU-1")
        self.assertEqual(captured.shared_paths, {shared, unrelated})
        self.assertEqual((captured.pids, captured.cuda_pids), ([1], [1]))

    def test_inventory_rejects_unsafe_engines(self):
        cases = (
            (
                "io_uring",
                {"io_uring_pids": [1]},
                SnapshotUsageError,
                "io_uring",
            ),
            (
                "device mismatch",
                {"cuda_holders": ([1], "GPU-2")},
                SnapshotRuntimeFailure,
                "holds GPU-2",
            ),
            (
                "escaped cache",
                {"referenced_paths": {os.path.expanduser("~/.triton/cache/x")}},
                SnapshotUsageError,
                "caches outside",
            ),
        )
        for name, overrides, error, expected in cases:
            with self.subTest(case=name), self.patched(**overrides):
                with self.assertRaisesRegex(error, expected):
                    self.runtime.inventory(self.artifact_path, 1, "/model", "GPU-1")

    def test_verify_dead_and_pidfile_validation(self):
        with patch.object(self.runtime, "alive", return_value=True):
            with self.assertRaisesRegex(SnapshotRuntimeFailure, "survived the dump"):
                self.runtime.verify_dead(inventory())
        with patch.object(self.runtime, "alive", return_value=False):
            self.runtime.verify_dead(inventory())

        pidfile = self.artifact_path / "root-process.pid"
        for payload in ("", "0", "abc", "99999999999\n", "12junk"):
            with self.subTest(payload=payload):
                pidfile.write_text(payload)
                with self.assertRaisesRegex(SnapshotRuntimeFailure, "invalid"):
                    self.runtime._read_restored_pid(pidfile)
        pidfile.write_text("1234\n")
        self.assertEqual(self.runtime._read_restored_pid(pidfile), 1234)
        pidfile.unlink()
        with self.assertRaisesRegex(SnapshotRuntimeFailure, "missing"):
            self.runtime._read_restored_pid(pidfile)

    # ------------------------------------------------------------------ #
    # artifact files, /dev/shm and process ownership
    # ------------------------------------------------------------------ #
    def test_runtime_files_round_trip_and_integrity(self):
        cache = self.artifact_path / "runtime/cache"
        cache.mkdir(parents=True)
        empty = self.artifact_path / "runtime/tmp/socket-parent"
        empty.mkdir(parents=True)
        (cache / "kernel.bin").write_bytes(b"compiled kernel")
        records = self.runtime.save_files(self.artifact_path)

        (cache / "kernel.bin").write_bytes(b"modified")
        empty.rmdir()
        self.runtime.restore_files(self.artifact_path, records)
        self.assertEqual((cache / "kernel.bin").read_bytes(), b"compiled kernel")
        self.assertTrue(empty.is_dir())

        (self.artifact_path / "files/cache/kernel.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(SnapshotRuntimeFailure, "Corrupt"):
            self.runtime.restore_files(self.artifact_path, records)
        (self.artifact_path / "files/cache/kernel.bin").unlink()
        with self.assertRaisesRegex(SnapshotRuntimeFailure, "missing"):
            self.runtime.restore_files(self.artifact_path, records)

        (self.artifact_path / "runtime/link").symlink_to("/tmp")
        with self.assertRaises(SnapshotSecurityError):
            self.runtime.save_files(self.artifact_path)

    def test_dev_shm_round_trip_and_conflicts(self):
        path = self.artifact_path / "shm/link_remap.608"
        path.write_bytes(b"semaphore bytes")
        records = self.runtime.save_dev_shm(self.artifact_path, {path})
        path.unlink()
        self.runtime.restore_dev_shm(self.artifact_path, records)
        self.assertEqual(path.read_bytes(), b"semaphore bytes")

        # A leftover from a failed attempt is accepted when it is identical.
        (self.artifact_path / "shm/load-state").write_bytes(b"snapshot")
        self.assertEqual(
            self.runtime.restore_dev_shm(
                self.artifact_path, [self.record("load-state")]
            ),
            {},
        )
        # Anything else under the same name is never replaced.
        (self.artifact_path / "shm/conflict").write_bytes(b"another engine")
        with self.assertRaisesRegex(SnapshotSecurityError, "already exists"):
            self.runtime.restore_dev_shm(
                self.artifact_path, [self.record("conflict"), self.record("fresh")]
            )
        self.assertEqual(
            (self.artifact_path / "shm/conflict").read_bytes(), b"another engine"
        )
        self.assertFalse((self.artifact_path / "shm/fresh").exists())

        for name in ("../etc/passwd", "link_remap.../file", "/etc/passwd"):
            with self.subTest(name=name), self.assertRaises(SnapshotSecurityError):
                self.runtime.restore_dev_shm(
                    self.artifact_path, [SnapshotFile(name, "x")]
                )

        target = self.artifact_path / "shm/replaced"
        target.write_bytes(b"owned")
        metadata = target.stat()
        target.rename(self.artifact_path / "owned-old")
        target.write_bytes(b"replacement")
        self.runtime.rollback_dev_shm({target: (metadata.st_dev, metadata.st_ino)})
        self.assertEqual(target.read_bytes(), b"replacement")

    def test_pin_restored_tree_verifies_identity(self):
        startup = ("python", "-m", "sglang.srt.engine_snapshot.startup")
        base = dict(
            root_pid=100,
            pids=[100, 101],
            state=(1, 100, 100, 5),
            command=startup,
            session=[100, 101],
            pidfds=[11, 12],
        )

        def pin_once(**patched):
            with (
                patch.object(
                    self.runtime, "_process_state", return_value=patched["state"]
                ),
                patch.object(
                    self.runtime, "_process_command", return_value=patched["command"]
                ),
                patch.object(
                    self.runtime, "_session_pids", return_value=patched["session"]
                ),
                patch.object(self.runtime, "_pidfd_exited", return_value=False),
                patch.object(
                    runtime_module.os, "pidfd_open", side_effect=patched["pidfds"]
                ),
                patch.object(runtime_module.os, "close"),
            ):
                self.runtime.pin_restored_tree(patched["root_pid"], patched["pids"])

        pin_once(**base)
        self.assertIn(100, self.runtime._restored)

        for name, overrides, expected in (
            ("not the leader", {"state": (1, 99, 100, 5)}, "process-group"),
            ("foreign command", {"command": ("python", "other.py")}, "command"),
            ("extra process", {"session": [100, 101, 102]}, "session"),
        ):
            with self.subTest(case=name):
                self.runtime._restored.clear()
                with self.assertRaisesRegex(SnapshotRuntimeFailure, expected):
                    pin_once(**{**base, **overrides})

    def test_cleanup_only_touches_owned_processes(self):
        self.runtime._restored[100] = ((100, 11), (101, 12))
        with (
            patch.object(runtime_module.signal, "pidfd_send_signal") as send,
            patch.object(self.runtime, "_pidfd_exited", return_value=True),
            patch.object(runtime_module.time, "sleep"),
            patch.object(runtime_module.os, "close") as close,
        ):
            self.runtime.cleanup_restored(100)
        self.assertEqual(
            [call.args[1] for call in send.call_args_list],
            [
                runtime_module.signal.SIGTERM,
                runtime_module.signal.SIGTERM,
                runtime_module.signal.SIGKILL,
                runtime_module.signal.SIGKILL,
            ],
        )
        self.assertEqual(close.call_count, 2)
        with self.assertRaisesRegex(SnapshotRuntimeFailure, "no pinned"):
            self.runtime.complete_restore(100)

        # The marker sweep is the fallback owner check: an untagged process is
        # left alone, a tagged one is killed.
        with patch.object(runtime_module.os, "kill") as kill:
            self.runtime.cleanup_tagged(self.artifact_path)
        kill.assert_not_called()

        tagged = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=dict(os.environ, SGLANG_SNAPSHOT_DIR=str(self.artifact_path)),
        )
        try:
            self.runtime.cleanup_tagged(self.artifact_path)
            self.assertEqual(tagged.wait(timeout=5), -9)
        finally:
            if tagged.poll() is None:
                tagged.kill()
                tagged.wait()

        marker = f"SGLANG_SNAPSHOT_DIR={self.artifact_path}".encode()
        with (
            patch.object(Path, "iterdir", return_value=[Path("/proc/1234")]),
            patch.object(Path, "read_bytes", side_effect=[marker, b"unrelated"]),
            patch.object(runtime_module.os, "pidfd_open", return_value=72),
            patch.object(runtime_module.os, "close") as close,
            patch("signal.pidfd_send_signal") as signal_pidfd,
        ):
            self.runtime.cleanup_tagged(self.artifact_path)
        signal_pidfd.assert_not_called()
        close.assert_called_once_with(72)

    # ------------------------------------------------------------------ #
    # readiness
    # ------------------------------------------------------------------ #
    def test_wait_until_reports_errors(self):
        control.write_error(
            self.artifact_path / "control", SnapshotRuntimeFailure("reload failed")
        )
        with self.assertRaisesRegex(SnapshotRuntimeFailure, "^reload failed$"):
            self.runtime.wait_until(
                self.artifact_path, lambda: True, ROOT_PID, 1, "engine did not start"
            )

        (self.artifact_path / "control" / control.ERROR).unlink()
        with patch.object(self.runtime, "alive", return_value=False):
            with self.assertRaisesRegex(SnapshotRuntimeFailure, "engine process"):
                self.runtime.wait_until(
                    self.artifact_path,
                    lambda: True,
                    ROOT_PID,
                    1,
                    "engine did not start",
                )

        with patch.object(self.runtime, "alive", return_value=True):
            with self.assertRaisesRegex(SnapshotRuntimeFailure, "timed out after"):
                self.runtime.wait_until(
                    self.artifact_path,
                    lambda: False,
                    ROOT_PID,
                    0.1,
                    "engine did not start",
                )

    def test_stdio_resource_ids(self):
        read_end, write_end = os.pipe()
        self.addCleanup(os.close, read_end)
        self.addCleanup(os.close, write_end)
        self.assertEqual(
            runtime_module._stdio_resource(write_end),
            f"pipe:[{os.fstat(write_end).st_ino}]",
        )

        path = self.artifact_path / "redirect.log"
        path.write_text("")
        descriptor = os.open(path, os.O_WRONLY)
        self.addCleanup(os.close, descriptor)
        self.assertEqual(
            runtime_module._stdio_resource(descriptor),
            os.path.realpath(path).removeprefix("/"),
        )

        with open(os.devnull, "w") as null:
            self.assertEqual(runtime_module._stdio_resource(null.fileno()), "dev/null")

        master, slave = os.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        self.assertRegex(
            runtime_module._stdio_resource(slave), r"^tty\[[0-9a-f]+:[0-9a-f]+\]$"
        )


if __name__ == "__main__":
    unittest.main()
