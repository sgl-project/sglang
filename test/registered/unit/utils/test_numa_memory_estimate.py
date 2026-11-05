import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.utils import common as utils_common
from sglang.srt.utils.common import (
    _read_numa_node_meminfos,
    get_available_gpu_memory,
    get_cpu_memory_capacity,
    get_per_numa_node_available_memory,
    get_per_numa_node_total_memory,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

KB = 1024
GB = 1 << 30


def _write_node(root: str, node: int, fields):
    node_dir = os.path.join(root, f"node{node}")
    os.makedirs(node_dir, exist_ok=True)
    with open(os.path.join(node_dir, "meminfo"), "w") as f:
        for name, kb in fields:
            f.write(f"Node {node} {name}:       {kb} kB\n")


class TestPerNumaNodeMemory(CustomTestCase):
    """The sysfs parser behind the asymmetric-NUMA memory bound."""

    def test_balanced_host_reads_each_node(self):
        with tempfile.TemporaryDirectory() as root:
            for node in (0, 1):
                _write_node(
                    root,
                    node,
                    [
                        ("MemFree", 10_000),
                        ("Active(file)", 20_000),
                        ("Inactive(file)", 30_000),
                        ("SReclaimable", 2_000),
                        ("MemTotal", 262_144),
                    ],
                )
            self.assertEqual(
                get_per_numa_node_available_memory(root),
                [62_000 * KB, 62_000 * KB],
            )
            self.assertEqual(
                get_per_numa_node_total_memory(root),
                [262_144 * KB, 262_144 * KB],
            )

    def test_imbalanced_host_smallest_node_wins(self):
        # The #48411 shape: one socket carries far less memory than the other.
        with tempfile.TemporaryDirectory() as root:
            _write_node(
                root,
                0,
                [
                    ("MemFree", 5_000),
                    ("Active(file)", 10_000),
                    ("Inactive(file)", 15_000),
                    ("SReclaimable", 1_000),
                    ("MemTotal", 200_000),
                ],
            )
            _write_node(
                root,
                1,
                [
                    ("MemFree", 50_000),
                    ("Active(file)", 100_000),
                    ("Inactive(file)", 150_000),
                    ("SReclaimable", 5_000),
                    ("MemTotal", 262_144),
                ],
            )
            self.assertEqual(
                get_per_numa_node_available_memory(root),
                [31_000 * KB, 305_000 * KB],
            )
            self.assertEqual(
                get_per_numa_node_total_memory(root),
                [200_000 * KB, 262_144 * KB],
            )

    def test_hidden_sysfs_returns_none(self):
        # Containers that hide /sys must not change the estimate.
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(get_per_numa_node_available_memory(root))
            self.assertIsNone(get_per_numa_node_total_memory(root))

    def test_unparsable_nodes_are_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "node0"))
            with open(os.path.join(root, "node0", "meminfo"), "w") as f:
                f.write("garbage with no kB values\n")
            self.assertEqual(_read_numa_node_meminfos(root), [])

    def test_non_node_directories_are_ignored(self):
        # has_cpu / has_memory live in the same sysfs directory.
        with tempfile.TemporaryDirectory() as root:
            _write_node(root, 0, [("MemFree", 1), ("MemTotal", 2)])
            os.makedirs(os.path.join(root, "has_cpu"))
            self.assertEqual(get_per_numa_node_total_memory(root), [2 * KB])


class TestCpuMemoryEstimates(CustomTestCase):
    """get_available_gpu_memory / get_cpu_memory_capacity on the cpu device."""

    def _patch(self, per_node_available, per_node_total, free_bytes, total_bytes):
        n_node = 2
        return (
            mock.patch.object(
                utils_common, "get_cpu_ids_by_node", return_value=["0", "1"]
            ),
            mock.patch.object(
                utils_common,
                "get_per_numa_node_available_memory",
                return_value=per_node_available,
            ),
            mock.patch.object(
                utils_common,
                "get_per_numa_node_total_memory",
                return_value=per_node_total,
            ),
            mock.patch.object(
                utils_common, "get_free_cpu_memory", return_value=free_bytes
            ),
            mock.patch.object(
                utils_common.psutil,
                "virtual_memory",
                return_value=mock.Mock(total=total_bytes),
            ),
            mock.patch.object(
                utils_common, "get_instance_memory_status", return_value=None
            ),
        )

    def test_balanced_host_keeps_symmetric_estimate(self):
        patches = self._patch(
            per_node_available=[100 * GB, 100 * GB],
            per_node_total=[256 * GB, 256 * GB],
            free_bytes=200 * GB,
            total_bytes=512 * GB,
        )
        for p in patches[:4]:
            p.start()
            self.addCleanup(p.stop)
        self.assertAlmostEqual(
            get_available_gpu_memory("cpu", 0), 100 * GB / GB, places=3
        )

    def test_imbalanced_host_bounded_by_smallest_node(self):
        # Symmetric split alone would claim 100 GiB per rank; socket0 only
        # holds 40 GiB, so the estimate must come down with it.
        patches = self._patch(
            per_node_available=[100 * GB, 40 * GB],
            per_node_total=[256 * GB, 196 * GB],
            free_bytes=200 * GB,
            total_bytes=512 * GB,
        )
        for p in patches[:4]:
            p.start()
            self.addCleanup(p.stop)
        self.assertAlmostEqual(
            get_available_gpu_memory("cpu", 0), 40 * GB / GB, places=3
        )

    def test_hidden_sysfs_falls_back_to_symmetric_estimate(self):
        patches = self._patch(
            per_node_available=None,
            per_node_total=None,
            free_bytes=200 * GB,
            total_bytes=512 * GB,
        )
        for p in patches[:4]:
            p.start()
            self.addCleanup(p.stop)
        self.assertAlmostEqual(
            get_available_gpu_memory("cpu", 0), 100 * GB / GB, places=3
        )

    def test_capacity_bounded_by_smallest_node_total(self):
        patches = self._patch(
            per_node_available=[100 * GB, 100 * GB],
            per_node_total=[256 * GB, 196 * GB],
            free_bytes=200 * GB,
            total_bytes=512 * GB,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_CPU_OMP_THREADS_BIND", None)
            # 512 GiB / 2 = 256 GiB symmetric, bounded to socket1's 196 GiB.
            self.assertEqual(get_cpu_memory_capacity(), 196 * GB // (1 << 20))

    def test_capacity_hidden_sysfs_keeps_symmetric_estimate(self):
        patches = self._patch(
            per_node_available=None,
            per_node_total=None,
            free_bytes=200 * GB,
            total_bytes=512 * GB,
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_CPU_OMP_THREADS_BIND", None)
            self.assertEqual(get_cpu_memory_capacity(), 256 * GB // (1 << 20))


if __name__ == "__main__":
    unittest.main()
