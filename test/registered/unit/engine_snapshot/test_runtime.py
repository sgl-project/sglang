import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sglang.srt.engine_snapshot.runtime as runtime_module
from sglang.srt.engine_snapshot.errors import (
    SnapshotRuntimeFailure,
    SnapshotUsageError,
)
from sglang.srt.engine_snapshot.runtime import EngineInventory, SnapshotRuntime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSnapshotRuntime(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.artifact_path = Path(self.directory.name)
        self.artifact_path.chmod(0o700)
        for directory in (
            "control",
            "files",
            "runtime",
            "shm",
            "dev_shm",
            "work",
        ):
            (self.artifact_path / directory).mkdir()
        self.runtime = SnapshotRuntime()
        shm = patch.object(SnapshotRuntime, "SHM_DIR", self.artifact_path / "shm")
        shm.start()
        self.addCleanup(shm.stop)

    def inventory(self, **overrides):
        fields = dict(
            root_pid=99999991,
            pids=[99999991],
            cuda_pids=[99999991],
            gpu_uuid="GPU-1",
            shared_paths=set(),
        )
        fields.update(overrides)
        return EngineInventory(**fields)

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
        with patch.object(self.runtime, "_capture", return_value=rows):
            holders, gpu_uuid = self.runtime.cuda_holders([99999991, 99999992])
        self.assertEqual(holders, [99999991, 99999992])
        self.assertEqual(gpu_uuid, "GPU-1")

        for rows, expected in (
            ("99999991, GPU-1\n99999992, GPU-2\n", "2 GPUs"),
            ("1234, GPU-1\n", "holds GPU memory"),
        ):
            with self.subTest(rows=rows):
                with patch.object(self.runtime, "_capture", return_value=rows):
                    with self.assertRaisesRegex(SnapshotRuntimeFailure, expected):
                        self.runtime.cuda_holders([99999991, 99999992])

    def test_inventory_rejects_unsafe_engines(self):
        cases = (
            (
                "io_uring",
                {
                    "tree_pids": [1],
                    "cuda_holders": ([1], "GPU-1"),
                    "io_uring_pids": [1],
                },
                SnapshotUsageError,
                "io_uring",
            ),
            (
                "escaped cache",
                {
                    "tree_pids": [1],
                    "cuda_holders": ([1], "GPU-1"),
                    "io_uring_pids": [],
                    "referenced_paths": {os.path.expanduser("~/.triton/cache/x")},
                },
                SnapshotUsageError,
                "caches outside",
            ),
        )
        for name, patched, error, expected in cases:
            with self.subTest(case=name):
                with (
                    patch.object(
                        self.runtime, "tree_pids", return_value=patched["tree_pids"]
                    ),
                    patch.object(
                        self.runtime,
                        "cuda_holders",
                        return_value=patched["cuda_holders"],
                    ),
                    patch.object(
                        self.runtime,
                        "io_uring_pids",
                        return_value=patched["io_uring_pids"],
                    ),
                    patch.object(
                        self.runtime,
                        "referenced_paths",
                        return_value=patched.get("referenced_paths", set()),
                    ),
                ):
                    with self.assertRaisesRegex(error, expected):
                        self.runtime.inventory(self.artifact_path, 1, "/model", "GPU-1")

        with (
            patch.object(self.runtime, "tree_pids", return_value=[1]),
            patch.object(self.runtime, "cuda_holders", return_value=([1], "GPU-2")),
        ):
            with self.assertRaisesRegex(SnapshotRuntimeFailure, "holds GPU-2"):
                self.runtime.inventory(self.artifact_path, 1, "/model", "GPU-1")

        self.assertEqual(
            self.runtime.cache_escapes(
                [
                    os.path.expanduser("~/.humming/launcher.so"),
                    str(self.artifact_path / "runtime/cache/ok.json"),
                    "/model/config.json",
                ],
                self.artifact_path,
                "/model",
            ),
            [os.path.expanduser("~/.humming/launcher.so")],
        )

    def test_inventory_collects_referenced_dev_shm(self):
        shared = self.artifact_path / "shm/load-state"
        unrelated = self.artifact_path / "shm/other-engine"
        shared.write_bytes(b"state")
        unrelated.write_bytes(b"other")
        with (
            patch.object(self.runtime, "tree_pids", return_value=[1]),
            patch.object(self.runtime, "cuda_holders", return_value=([1], "GPU-1")),
            patch.object(self.runtime, "io_uring_pids", return_value=[]),
            patch.object(
                self.runtime,
                "referenced_paths",
                return_value={
                    str(shared),
                    str(unrelated),
                    str(self.artifact_path / "runtime/x"),
                },
            ),
        ):
            inventory = self.runtime.inventory(self.artifact_path, 1, "/model", "GPU-1")
        self.assertEqual(inventory.shared_paths, {shared, unrelated})
        self.assertEqual((inventory.pids, inventory.cuda_pids), ([1], [1]))

    # ------------------------------------------------------------------ #
    # artifact files, /dev/shm and process ownership
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # readiness and the restore transaction
    # ------------------------------------------------------------------ #


if __name__ == "__main__":
    unittest.main()
