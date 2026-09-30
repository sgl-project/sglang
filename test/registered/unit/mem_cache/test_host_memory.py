"""Exercise container and ancestor budgets using synthetic procfs/cgroup files."""

import functools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.mem_cache import host_memory
from sglang.test.ci.ci_register import register_cpu_ci

_cgroup_memory_headroom = host_memory._cgroup_memory_headroom

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

    def memory(self, path, usage, maximum="max", high="max", v1=False, stat=None):
        directory = self.mount / path
        directory.mkdir(parents=True, exist_ok=True)
        files = (
            {"memory.limit_in_bytes": maximum, "memory.usage_in_bytes": usage}
            if v1
            else {"memory.max": maximum, "memory.high": high, "memory.current": usage}
        )
        prefix = "total_" if v1 else ""
        stat = {f"{prefix}active_file": 0, f"{prefix}inactive_file": 0} | (stat or {})
        files["memory.stat"] = "".join(
            f"{key} {value}\n" for key, value in stat.items()
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

    def test_reclaimable_page_cache_is_not_used(self):
        """Page cache charged to the cgroup, such as a checkpoint just read, is
        reclaimed under the limit and must not shrink the headroom. Shared
        memory is counted in file/cache but cannot be reclaimed."""
        v2 = {"file": 700, "shmem": 200, "active_file": 200, "inactive_file": 300}
        v1 = {
            "cache": 700,
            "shmem": 200,
            "active_file": 0,
            "inactive_file": 0,
            "total_active_file": 200,
            "total_inactive_file": 300,
        }
        for is_v1, stat, usage, expected in [
            (False, v2, 900, 600),
            (True, v1, 900, 600),
            # v1 usage is approximate and may trail the cache counters.
            (True, v1, 400, 1000),
        ]:
            with self.subTest(v1=is_v1, usage=usage):
                self.configure(v1=is_v1)
                self.memory("task/engine", usage, 1000, v1=is_v1, stat=stat)
                self.assertEqual(
                    host_memory._cgroup_memory_headroom(self.proc), expected
                )

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
        for cgroup, allow_fallback, expected in [
            (None, False, 100),
            (200, False, 100),
            (50, False, 50),
            (50, True, 50),
        ]:
            with (
                self.subTest(cgroup=cgroup, allow_fallback=allow_fallback),
                patch.object(
                    host_memory.psutil,
                    "virtual_memory",
                    return_value=Mock(available=100),
                ),
                patch.object(
                    host_memory, "_cgroup_memory_headroom", return_value=cgroup
                ),
            ):
                self.assertEqual(
                    host_memory.available_host_memory_bytes(
                        allow_cgroup_fallback=allow_fallback
                    ),
                    expected,
                )

    def available(self, host_available=5000, *, allow_cgroup_fallback=False):
        """Public sizing entry point over the synthetic procfs."""
        with (
            patch.object(
                host_memory.psutil,
                "virtual_memory",
                return_value=Mock(available=host_available),
            ),
            patch.object(
                host_memory,
                "_cgroup_memory_headroom",
                functools.partial(_cgroup_memory_headroom, self.proc),
            ),
        ):
            return host_memory.available_host_memory_bytes(
                allow_cgroup_fallback=allow_cgroup_fallback
            )

    def without_cgroupfs(self, v1=False):
        # A container sharing the host cgroup namespace, no cgroupfs mounted.
        self.configure("/system.slice/engine.scope", v1=v1)
        (self.proc / "self/mountinfo").write_text("1 0 0:1 / /proc rw - proc proc rw\n")

    def test_unavailable_cgroup_requires_explicit_sizing(self):
        for v1 in [False, True]:
            for mounted in [False, True]:
                with self.subTest(v1=v1, mounted=mounted):
                    if mounted:
                        self.configure("/system.slice/engine.scope", v1=v1)
                    else:
                        self.without_cgroupfs(v1=v1)
                    with self.assertRaisesRegex(RuntimeError, "set --hicache-size"):
                        self.available()
                    self.assertEqual(self.available(allow_cgroup_fallback=True), 5000)

    def test_missing_counters_for_known_limit_requires_explicit_sizing(self):
        for name in ("memory.current", "memory.stat"):
            with self.subTest(name=name):
                self.configure()
                self.memory("task/engine", 100, 1000)
                (self.mount / "task/engine" / name).unlink()
                with self.assertRaisesRegex(RuntimeError, "set --hicache-size"):
                    self.available()
                self.assertEqual(self.available(allow_cgroup_fallback=True), 5000)


class TestCgroupHugetlb(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc = self.root / "proc"
        (self.proc / "self").mkdir(parents=True)
        self.mount = self.root / "cgroup mount"
        self.mount.mkdir()

    def configure(self, membership="/task/engine", v1=False):
        controllers = "hugetlb" if v1 else ""
        (self.proc / "self/cgroup").write_text(f"0:{controllers}:{membership}\n")
        filesystem = "cgroup" if v1 else "cgroup2"
        options = "rw,hugetlb" if v1 else "rw"
        escaped = str(self.mount).replace(" ", r"\040")
        (self.proc / "self/mountinfo").write_text(
            f"1 0 0:1 / {escaped} rw - {filesystem} cgroup {options}\n"
        )

    def hugetlb(self, path, usage, maximum="max", label="2MB", v1=False):
        directory = self.mount / path
        directory.mkdir(parents=True, exist_ok=True)
        files = (
            {
                f"hugetlb.{label}.limit_in_bytes": maximum,
                f"hugetlb.{label}.usage_in_bytes": usage,
            }
            if v1
            else {
                f"hugetlb.{label}.max": maximum,
                f"hugetlb.{label}.current": usage,
            }
        )
        for name, value in files.items():
            (directory / name).write_text(str(value))

    def headroom(self, page_size=2 * 1024**2):
        return host_memory.cgroup_hugetlb_headroom_bytes(page_size, self.proc)

    def test_v1_limits(self):
        self.configure(v1=True)
        self.hugetlb("task/engine", 100, 1000, v1=True)
        self.assertEqual(self.headroom(), 900)

    def test_v2_limits(self):
        self.configure()
        self.hugetlb("task/engine", 100, 1000)
        self.hugetlb("task", 400, 1000)
        self.assertEqual(self.headroom(), 600)

    def test_controller_not_enabled(self):
        self.configure()
        (self.mount / "task/engine").mkdir(parents=True)
        self.assertIsNone(self.headroom())

    def test_page_size_selects_hugetlb_pool_label(self):
        self.configure()
        self.hugetlb("task/engine", 48, 2048, label="1GB")
        self.assertEqual(self.headroom(1024**3), 2000)


if __name__ == "__main__":
    unittest.main()
