"""HiCache host pools: one free-memory snapshot per build, and the SWA host split.

python -m pytest test/registered/unit/mem_cache/test_hicache_host_budget.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _swa_host_shares
from sglang.srt.mem_cache.pool_host.base import (
    HICACHE_HOST_MEMORY_RESERVE_BYTES,
    _host_memory_budget,
    host_memory_budget_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 10**9


def _pool(size, bytes_per_token, *, rows=None, host_capacity_bytes=None):
    """A device KV pool serving `size` tokens out of `rows` allocated ones."""
    rows = size if rows is None else rows
    half = bytes_per_token * rows // 2
    return SimpleNamespace(
        size=size,
        k_buffer=[torch.empty(rows, 1)],
        get_kv_size_bytes=lambda: (half, half),
        host_capacity_bytes=host_capacity_bytes,
    )


class TestHostBudgetSnapshot(CustomTestCase):
    def _build(self, *, enable_hierarchical_cache, requests=(), free=None):
        """build_kv_cache with the inner build booking `requests` bytes in turn."""
        seen, barriers = [], []

        def build(**kwargs):
            seen.append(_host_memory_budget.get())
            for requested in requests:
                seen.append(host_memory_budget_bytes(requested))
            return "built"

        free = (
            4 * 100 * GB + HICACHE_HOST_MEMORY_RESERVE_BYTES if free is None else free
        )
        with (
            patch.object(kv_cache_builder, "_build_kv_cache", side_effect=build),
            patch.object(
                kv_cache_builder, "available_host_memory_bytes", return_value=free
            ),
            patch.object(kv_cache_builder, "ranks_per_host", return_value=4),
            patch.object(
                kv_cache_builder,
                "_host_rank_barrier",
                side_effect=lambda: barriers.append(len(seen)),
            ),
        ):
            out = kv_cache_builder.build_kv_cache(
                enable_hierarchical_cache=enable_hierarchical_cache
            )
        self.assertEqual(out, "built")
        self.assertIsNone(_host_memory_budget.get())  # the scope ends with the build
        return seen, barriers

    def test_every_pool_is_booked_against_one_snapshot_taken_between_barriers(self):
        with self.assertLogs(kv_cache_builder.logger, "INFO") as logs:
            seen, barriers = self._build(
                enable_hierarchical_cache=True, requests=(30 * GB, 50 * GB)
            )
        # each pool sees what the earlier ones left of the per-rank share
        self.assertEqual(seen, [100 * GB, 100 * GB, 70 * GB])
        # both barriers come before the first pool books anything
        self.assertEqual(barriers, [0, 0])
        self.assertIn("100.00 GB per rank", "".join(logs.output))
        self.assertIn("claimed 80.00 GB", "".join(logs.output))

    def test_no_snapshot_and_no_barrier_without_hierarchical_cache(self):
        seen, barriers = self._build(enable_hierarchical_cache=False)
        self.assertEqual((seen, barriers), ([None], []))

    def test_a_node_without_memory_above_the_reserve_gets_a_zero_budget(self):
        seen, _ = self._build(
            enable_hierarchical_cache=True, free=HICACHE_HOST_MEMORY_RESERVE_BYTES - 1
        )
        self.assertEqual(seen, [0])

    def test_the_barrier_is_a_no_op_without_a_process_group(self):
        self.assertIsNone(kv_cache_builder._host_rank_barrier())


class TestSWAHostShares(CustomTestCase):
    def test_the_stock_split_applies_unless_the_host_ratio_is_above_the_device_ratio(
        self,
    ):
        full, swa = _pool(1000, 100), _pool(100, 50)  # device ratio 0.1
        for ratio in (0.0, 0.05, 0.1):
            with self.subTest(ratio=ratio):
                self.assertIsNone(_swa_host_shares(110, full, swa, ratio))
        explicit = _pool(100, 50, host_capacity_bytes=GB)
        self.assertIsNone(_swa_host_shares(110, full, explicit, 0.2))

    def test_the_host_tiers_are_sized_as_if_the_device_used_the_host_ratio(self):
        # full 100 B/token x 1000; sliding window 50 B/token x 1000 x 0.2
        shares = _swa_host_shares(110, _pool(1000, 100), _pool(100, 50), 0.2)
        self.assertEqual([round(s, 6) for s in shares], [100.0, 10.0])

    def test_bytes_per_token_come_from_the_allocated_rows(self):
        # the full pool serves 500 of its 1000 rows after a resize; the split keeps
        # 100 B/token for it, not 1000 rows' bytes over 500 tokens
        full = _pool(500, 100, rows=1000)
        shares = _swa_host_shares(110, full, _pool(50, 50), 0.2)
        self.assertEqual([round(s, 6) for s in shares], [100.0, 10.0])


if __name__ == "__main__":
    unittest.main()
