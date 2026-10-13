"""Exercise container and ancestor budgets using synthetic procfs/cgroup files."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.mem_cache import host_memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHostMemory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc = self.root / "proc"
        (self.proc / "self").mkdir(parents=True)
        self.mount = self.root / "cgroup mount"
        self.mount.mkdir()

    def configure(self, membership="/task/engine", mount_root="/", v1=False):
        controllers = "memory" if v1 else ""
        (self.proc / "self/cgroup").write_text(f"0:{controllers}:{membership}\n")
        filesystem = "cgroup" if v1 else "cgroup2"
        options = "rw,memory" if v1 else "rw"
        escaped = str(self.mount).replace(" ", r"\040")
        (self.proc / "self/mountinfo").write_text(
            f"1 0 0:1 {mount_root} {escaped} rw - {filesystem} cgroup {options}\n"
        )

    def memory(self, path, usage, maximum="max", high="max", v1=False):
        directory = self.mount / path
        directory.mkdir(parents=True, exist_ok=True)
        files = (
            {"memory.limit_in_bytes": maximum, "memory.usage_in_bytes": usage}
            if v1
            else {"memory.max": maximum, "memory.high": high, "memory.current": usage}
        )
        for name, value in files.items():
            (directory / name).write_text(str(value))

    def test_v2_parent_and_high_limits(self):
        self.configure()
        self.memory("task/engine", 100)
        for maximum, high, usage, expected in [
            (1000, "max", 300, 700),
            (1000, 600, 300, 300),
            ("max", 600, 700, 0),
            ("max", "max", 300, None),
        ]:
            with self.subTest(maximum=maximum, high=high):
                self.memory("task", usage, maximum, high)
                self.assertEqual(
                    host_memory._cgroup_memory_headroom(self.proc), expected
                )
        self.memory("task", 100, 1000)
        self.memory("task/engine", 100, 250)
        self.assertEqual(host_memory._cgroup_memory_headroom(self.proc), 150)

    def test_mount_subtree_and_cgroup_namespace(self):
        self.memory("engine", 100, 900)
        self.memory("", 300, 1000)
        for membership in ["/host/task/engine", "/engine"]:
            with self.subTest(membership=membership):
                self.configure(membership, mount_root="/host/task")
                self.assertEqual(host_memory._cgroup_memory_headroom(self.proc), 700)

    def test_v1_parent_limit_and_unlimited_sentinel(self):
        self.configure(v1=True)
        self.memory("task/engine", 100, 2**63 - 4096, v1=True)
        self.memory("task", 400, 1000, v1=True)
        self.assertEqual(host_memory._cgroup_memory_headroom(self.proc), 600)

    def test_independent_engines_have_separate_allowances(self):
        # Both engines see the same host RAM but have different charged usage.
        for task, usage, expected in [("a", 300, 700), ("b", 600, 400)]:
            with self.subTest(task=task):
                self.configure(f"/{task}/engine")
                self.memory(f"{task}/engine", 0)
                self.memory(task, usage, 1000)
                with (
                    patch.object(
                        host_memory.psutil,
                        "virtual_memory",
                        return_value=Mock(available=2000),
                    ),
                    patch.object(
                        host_memory,
                        "_cgroup_memory_headroom",
                        return_value=host_memory._cgroup_memory_headroom(self.proc),
                    ),
                ):
                    self.assertEqual(
                        host_memory.available_host_memory_bytes(), expected
                    )

    def test_host_availability_is_also_a_bound(self):
        for cgroup, expected in [(None, 100), (200, 100), (50, 50)]:
            with (
                self.subTest(cgroup=cgroup),
                patch.object(
                    host_memory.psutil,
                    "virtual_memory",
                    return_value=Mock(available=100),
                ),
                patch.object(
                    host_memory, "_cgroup_memory_headroom", return_value=cgroup
                ),
            ):
                self.assertEqual(host_memory.available_host_memory_bytes(), expected)

    def test_unmounted_memory_cgroup_fails(self):
        self.configure()
        (self.proc / "self/mountinfo").write_text("")
        with self.assertRaisesRegex(RuntimeError, "Cannot locate"):
            host_memory._cgroup_memory_headroom(self.proc)

    def test_missing_usage_for_known_limit_fails(self):
        self.configure()
        self.memory("task/engine", 100, 1000)
        (self.mount / "task/engine/memory.current").unlink()
        with self.assertRaises(FileNotFoundError):
            host_memory._cgroup_memory_headroom(self.proc)


if __name__ == "__main__":
    unittest.main()
