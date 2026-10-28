import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host import base
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

RATIO = 2.0
RANKS = 4


def _device_pool():
    return MHATokenToKVPool(
        size=128,
        page_size=2,
        dtype=torch.float16,
        head_num=2,
        head_dim=4,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )


def _host_pool(pool):
    return MHATokenToKVPoolHost(
        pool, RATIO, 0, 2, "layer_first", pin_memory=False, device="cpu"
    )


class TestExplicitHostMemoryBudget(CustomTestCase):
    def setUp(self):
        self.pool = _device_pool()
        with patch.object(base, "available_host_memory_bytes", return_value=1 << 50):
            host = _host_pool(self.pool)
        # What one rank's host pool pins, and free memory with room for every
        # rank's pool and some to spare -- before, and after the others pinned
        # theirs.
        self.pool_bytes = host.size * host.size_per_token
        self.whole_host = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + int(
            RANKS * self.pool_bytes * 1.2
        )
        self.after_peers = self.whole_host - (RANKS - 1) * self.pool_bytes

    def test_a_late_rank_is_refused_when_each_pool_samples_live(self):
        """The bug: the last rank divides what its peers left by every rank."""
        with (
            patch.object(base, "ranks_per_host", return_value=RANKS),
            patch.object(
                base, "available_host_memory_bytes", return_value=self.after_peers
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                _host_pool(self.pool)

    def test_a_late_rank_builds_its_pool_against_the_shared_snapshot(self):
        """Sampled before any rank allocated, the budget holds every rank's
        pool -- however late this rank builds its own."""
        samples = iter([self.whole_host] + [self.after_peers] * 8)
        with (
            patch.object(base, "ranks_per_host", return_value=RANKS),
            patch.object(
                base,
                "available_host_memory_bytes",
                side_effect=lambda **_: next(samples),
            ),
            base.explicit_host_memory_budget(enabled=True),
        ):
            host = _host_pool(self.pool)
            self.assertEqual(host.size * host.size_per_token, self.pool_bytes)
        self.assertIsNone(base._host_memory_budget.get())

    def test_inside_auto_sizing_it_does_nothing(self):
        """Auto-sizing opened its own snapshot; it is left exactly as it is."""
        with (
            base.host_memory_budget_scope(12345),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "all_reduce") as all_reduce,
            base.explicit_host_memory_budget(enabled=True),
        ):
            self.assertEqual(base._host_memory_budget.get(), 12345)
        all_reduce.assert_not_called()

    def test_every_rank_takes_the_smallest_budget_any_rank_sampled(self):
        """One collective before any pool: every rank waits in it for the
        last to sample, and books the minimum."""
        calls = []

        def all_reduce(tensor, op, group):
            calls.append((tensor.dtype, op))
            tensor.fill_(1)  # another rank sampled almost nothing

        with (
            patch.object(base, "ranks_per_host", return_value=1),
            patch.object(base, "available_host_memory_bytes", return_value=1 << 50),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "all_reduce", side_effect=all_reduce),
            patch.object(base, "get_parallel"),
            base.explicit_host_memory_budget(enabled=True),
        ):
            self.assertEqual(base._host_memory_budget.get(), 1)
            with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                _host_pool(self.pool)
        self.assertEqual(calls, [(torch.int64, torch.distributed.ReduceOp.MIN)])

    def test_without_host_pools_it_does_nothing(self):
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "all_reduce") as all_reduce,
            base.explicit_host_memory_budget(enabled=False),
        ):
            self.assertIsNone(base._host_memory_budget.get())
        all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
