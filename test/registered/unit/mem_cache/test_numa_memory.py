"""NUMA-capacity regressions using synthetic topology, without a GPU or binding."""

import ctypes
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.mem_cache import numa_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNumaMemory(CustomTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "has_memory").write_text("0,2,65\n")
        for node, gib in [(0, 32), (2, 64), (65, 128)]:
            directory = self.root / f"node{node}"
            directory.mkdir()
            (directory / "meminfo").write_text(
                f"Node {node} MemTotal: {gib * 1024**2} kB\n"
            )

    def capacity(self, mode=2, policy_nodes=None, allowed_nodes=None):
        with patch.object(
            numa_memory,
            "_get_numa_policy",
            return_value=(
                mode,
                {2} if policy_nodes is None else policy_nodes,
                {0, 2, 65} if allowed_nodes is None else allowed_nodes,
            ),
        ):
            return numa_memory._numa_memory_capacity(self.root)

    def test_bind_uses_node_capacity_not_machine_capacity(self):
        self.assertEqual(self.capacity(), ([2], 64 * 1024**3))
        self.assertEqual(self.capacity(policy_nodes={0, 2}), ([0, 2], 96 * 1024**3))

    def test_preferred_and_interleave_can_fall_back(self):
        for mode in (0, 1, 3, 4, 5, 6):
            with self.subTest(mode=mode):
                self.assertIsNone(self.capacity(mode=mode))

    def test_unhandled_policy_flags_use_only_cpuset_ceiling(self):
        # get_mempolicy can return flags whose mask rebinding rules differ.
        # In that case, do not infer a stricter limit from the policy mask.
        self.assertIsNone(self.capacity(mode=2 | (1 << 13)))
        self.assertEqual(
            self.capacity(mode=2 | (1 << 13), allowed_nodes={0, 2}),
            ([0, 2], 96 * 1024**3),
        )

    def test_cpuset_limits_fallback_policies(self):
        for mode in (0, 1, 3, 4, 5, 6):
            with self.subTest(mode=mode):
                self.assertEqual(
                    self.capacity(mode=mode, allowed_nodes={0, 2}),
                    ([0, 2], 96 * 1024**3),
                )

    def test_bind_is_intersected_with_cpuset(self):
        self.assertEqual(
            self.capacity(policy_nodes={0, 2}, allowed_nodes={2, 65}),
            ([2], 64 * 1024**3),
        )

    def test_relative_mask_is_mapped_to_allowed_memory_nodes(self):
        self.assertEqual(
            self.capacity(
                mode=2 | numa_memory._MPOL_F_RELATIVE_NODES,
                policy_nodes={2},
                allowed_nodes={2, 65},
            ),
            ([2], 64 * 1024**3),
        )

    def test_static_mask_after_disjoint_cpuset_change_falls_back(self):
        self.assertEqual(
            self.capacity(
                mode=2 | numa_memory._MPOL_F_STATIC_NODES,
                policy_nodes={0},
                allowed_nodes={2},
            ),
            ([2], 64 * 1024**3),
        )

    def test_memory_node_does_not_require_cpus(self):
        # No cpulist files exist in this fixture, including for node 65.
        self.assertEqual(self.capacity(policy_nodes={65}), ([65], 128 * 1024**3))

    def test_policy_is_not_cached_across_calls(self):
        with patch.object(
            numa_memory,
            "_get_numa_policy",
            side_effect=[(2, {0}, {0, 2, 65}), (2, {2}, {0, 2, 65})],
        ):
            self.assertEqual(numa_memory._numa_memory_capacity(self.root)[0], [0])
            self.assertEqual(numa_memory._numa_memory_capacity(self.root)[0], [2])

    def test_missing_counter_does_not_return_a_partial_capacity(self):
        (self.root / "node2/meminfo").unlink()
        self.assertIsNone(self.capacity(policy_nodes={0, 2}))

    def test_unavailable_numa_information_keeps_existing_behavior(self):
        for error in (PermissionError("seccomp"), OSError("no libnuma")):
            with (
                self.subTest(error=error),
                patch.object(numa_memory, "_get_numa_policy", side_effect=error),
            ):
                self.assertIsNone(numa_memory._numa_memory_capacity(self.root))
        (self.root / "has_memory").unlink()
        self.assertIsNone(numa_memory._numa_memory_capacity(self.root))

    def test_malformed_counter_is_not_a_zero_capacity(self):
        for counter in ("unknown kB", "-1 kB", "0 kB", "64 MB"):
            with self.subTest(counter=counter):
                (self.root / "node2/meminfo").write_text(
                    f"Node 2 MemTotal: {counter}\n"
                )
                self.assertIsNone(self.capacity())

    def test_policy_query_reads_both_masks_and_high_node_ids(self):
        lib = Mock()
        lib.numa_num_possible_nodes.return_value = 128
        bits = ctypes.sizeof(ctypes.c_ulong) * 8

        def get_mempolicy(mode, mask, maxnode, address, flags):
            self.assertEqual(maxnode, 128)
            self.assertIsNone(address)
            ctypes.cast(mode, ctypes.POINTER(ctypes.c_int))[0] = 0 if flags else 2
            for node in {2, 65} if flags else {65}:
                mask[node // bits] |= 1 << (node % bits)
            return 0

        lib.get_mempolicy.side_effect = get_mempolicy
        with patch.object(numa_memory, "_libnuma", return_value=lib):
            self.assertEqual(numa_memory._get_numa_policy(), (2, {65}, {2, 65}))
        self.assertEqual([c.args[-1] for c in lib.get_mempolicy.call_args_list], [0, 4])

    def test_oversized_mapping_is_rejected_before_allocation(self):
        with patch.object(
            numa_memory, "_numa_memory_capacity", return_value=([2], 64 * 1024**3)
        ):
            with self.assertRaisesRegex(
                ValueError, r"65.00 GiB.*\[2\].*64.00 GiB total RAM"
            ):
                numa_memory.check_numa_memory_capacity(65 * 1024**3)
            numa_memory.check_numa_memory_capacity(64 * 1024**3)

    def test_unknown_capacity_is_not_treated_as_zero(self):
        with patch.object(numa_memory, "_numa_memory_capacity", return_value=None):
            numa_memory.check_numa_memory_capacity(65 * 1024**3)


if __name__ == "__main__":
    unittest.main()
