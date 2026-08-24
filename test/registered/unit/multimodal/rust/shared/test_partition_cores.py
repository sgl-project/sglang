"""``RustServer._partition_cores`` (managers/rust_server.py): the pool cores must
be a *bounded* slice of this rank's allowed cores, not the whole remainder —
sibling TP ranks share the NUMA node, so an unbounded mask lets MM preprocessing
bursts preempt a sibling's CUDA-launch thread (measured: ~20 ms of ViT wall time
per image request on TP4). Pure computation, so no Rust extension needed."""

import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.rust_server import RustServer  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def partition(node_cores, **kwargs):
    with patch("os.sched_getaffinity", return_value=set(node_cores), create=True):
        return RustServer._partition_cores(**kwargs)


class TestPartitionCores(CustomTestCase):
    def test_pool_is_bounded_not_the_node_remainder(self):
        # A 120-core NUMA node shared with sibling TP ranks: the pools must
        # NOT get cores 2..119.
        launch, pool = partition(range(120), mm_workers=8)
        self.assertEqual(launch, [0, 1])
        self.assertEqual(pool, list(range(2, 14)))  # max(8, 8 + 4) after reserve

    def test_budget_scales_with_mm_workers_with_a_floor(self):
        _, pool_text = partition(range(120), mm_workers=0)
        _, pool_mm = partition(range(120), mm_workers=16)
        self.assertEqual(len(pool_text), 8)  # the floor covers the I/O threads
        self.assertEqual(len(pool_mm), 20)

    def test_small_allowance_degrades_gracefully(self):
        # Fewer allowed cores than the budget: take what exists after the
        # launch reserve, never raise.
        launch, pool = partition(range(6), mm_workers=8)
        self.assertEqual(launch, [0])  # min(2, 6 // 4) == 1
        self.assertEqual(pool, list(range(1, 6)))
        # Below the split threshold: unpinned.
        self.assertEqual(partition(range(3), mm_workers=8), (None, None))

    def test_launch_and_pool_cores_are_disjoint(self):
        launch, pool = partition(range(32), mm_workers=8)
        self.assertFalse(set(launch) & set(pool))


if __name__ == "__main__":
    unittest.main()
