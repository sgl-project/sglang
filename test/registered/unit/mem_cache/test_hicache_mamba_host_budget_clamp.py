"""HiCache: the Mamba host pool must degrade to the host budget, not crash.

Hybrid Mamba models build their host pool by ratio (--hicache-ratio) or by an
explicit --hicache-size split, and unlike the KV host pools they do not go
through auto_size_hicache. When the computed pool did not fit the available
host memory (e.g. a container whose cgroup limit is mostly consumed by the
weight load), MambaPoolHost aborted engine startup with "Not enough host
memory available". The pool now clamps to the available budget with a warning
instead, and only raises when not even one page fits.
"""

import unittest
from unittest import mock

from sglang.srt.mem_cache.pool_host.mamba import (
    MambaPoolHost,
    _clamp_host_pool_size,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SIZE_PER_TOKEN = 1_000_000  # 1 MB per slot
PAGE_SIZE = 64


def _make_host(size: int) -> MambaPoolHost:
    host = MambaPoolHost.__new__(MambaPoolHost)
    host.size = size
    host.size_per_token = SIZE_PER_TOKEN
    host.page_size = PAGE_SIZE
    host.page_num = size // PAGE_SIZE
    host.layout = "page_first"
    return host


def _budget(_requested: int) -> int:
    return 100 * SIZE_PER_TOKEN  # 100 slots' worth of host memory


class TestClampHostPoolSize(unittest.TestCase):
    def test_clamps_to_page_aligned_budget(self):
        # 100 slots of budget, 64-slot pages -> 64.
        self.assertEqual(
            _clamp_host_pool_size(1000, SIZE_PER_TOKEN, PAGE_SIZE, 100 * SIZE_PER_TOKEN),
            64,
        )

    def test_less_than_one_page_is_zero(self):
        self.assertEqual(
            _clamp_host_pool_size(1000, SIZE_PER_TOKEN, PAGE_SIZE, SIZE_PER_TOKEN),
            0,
        )


class TestFitHostPoolToBudget(CustomTestCase):
    def test_no_clamp_when_pool_fits(self):
        host = _make_host(64)  # 64 MB, budget 100 MB
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mamba.host_memory_budget_bytes",
            side_effect=_budget,
        ):
            host._fit_host_pool_to_budget(host_size=0)
        self.assertEqual(host.size, 64)

    def test_clamps_with_warning_when_over_budget(self):
        host = _make_host(640)  # 640 MB requested, 100 MB available
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mamba.host_memory_budget_bytes",
            side_effect=_budget,
        ):
            with self.assertLogs(
                "sglang.srt.mem_cache.pool_host.mamba", level="WARNING"
            ) as logs:
                host._fit_host_pool_to_budget(host_size=0)
        self.assertEqual(host.size, 64)  # 100 slots -> one 64-slot page
        self.assertEqual(host.page_num, 1)
        self.assertIn("shrinking it from 640 to 64 slots", logs.output[0])

    def test_raises_only_when_not_even_one_page_fits(self):
        host = _make_host(640)
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mamba.host_memory_budget_bytes",
            side_effect=lambda _requested: SIZE_PER_TOKEN,  # one slot, < one page
        ):
            with self.assertRaisesRegex(ValueError, "even one page"):
                host._fit_host_pool_to_budget(host_size=0)

    def test_clamped_size_is_booked_against_scope_budget(self):
        from sglang.srt.mem_cache.pool_host.base import (
            host_memory_budget_bytes,
            host_memory_budget_scope,
        )

        host = _make_host(640)  # 640 MB requested, scope allows 200 MB
        with host_memory_budget_scope(200 * SIZE_PER_TOKEN):
            host._fit_host_pool_to_budget(host_size=0)
            # 200 slots of budget -> three 64-slot pages = 192 booked.
            self.assertEqual(host.size, 192)
            self.assertEqual(
                host_memory_budget_bytes(0), (200 - 192) * SIZE_PER_TOKEN
            )


if __name__ == "__main__":
    unittest.main()
