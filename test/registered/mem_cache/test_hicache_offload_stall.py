"""A full host pool stops KV offload permanently, and must say so.

``write_backup`` returns 0 when the host pool cannot take another page. That is
normally transient -- ``evict_host`` frees something and the next attempt
succeeds -- but ``evict_host`` can only free a node that the *device* tier has
already dropped: ``_update_host_leaf_status`` admits a node to
``evictable_host_leaves`` only when ``node.evicted``. So when the device pool is
larger than the host pool, the host pool fills first, GPU eviction never fires,
nothing is ever evictable, and offload stops for the life of the process.

Measured on the 2-worker KVCR POC (Qwen3-8B, page-size 64, ~19-page prefixes):
offload froze at 1680 of 1696 host pages and never resumed; halving the host pool
moved the freeze to 844 pages; making the device pool smaller than the host pool
removed it entirely, and offload then ran past the host pool size.

Nothing errors on this path: L2 write-through and L3 offload just stop, and every
later request reports an ordinary cache miss. The warning is the fix available
here -- the reclaim rule itself is HiRadixCache's, but an operator who sees this
line knows to raise ``--hicache-size``.

What turns this red: deleting the ``_log_offload_stalled()`` call from
``write_backup``'s ``host_indices is None`` branch, or narrowing the warning so
it no longer names the host pool.

    python -m pytest test/registered/mem_cache/test_hicache_offload_stall.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.hiradix_cache import (
    _OFFLOAD_STALLED_LOG_INTERVAL_S,
    HiRadixCache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StalledCache:
    """The narrowest object ``write_backup`` needs, with a full host pool.

    Built by ``__new__`` rather than a real constructor: ``HiRadixCache.__init__``
    allocates device memory and starts controller threads, none of which this
    path touches.
    """

    @staticmethod
    def build() -> HiRadixCache:
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.cache_controller = MagicMock()
        # A full host pool: every allocation attempt declines.
        cache.cache_controller.write.return_value = None
        cache.evict_host = MagicMock()
        cache._last_offload_stalled_log_s = 0.0
        cache._get_extra_pools = lambda: {}
        return cache


def _node(*, backuped_parent: bool = True) -> MagicMock:
    node = MagicMock()
    node.value = [0, 1, 2, 3]
    node.key = [10, 11, 12, 13]
    node.parent.backuped = backuped_parent
    return node


class OffloadStallWarningTest(unittest.TestCase):
    def test_exhausted_host_pool_warns(self):
        cache = _StalledCache.build()
        with self.assertLogs("sglang.srt.mem_cache.hiradix_cache", "WARNING") as logs:
            written = cache.write_backup(_node(), write_back=True)
        self.assertEqual(written, 0)
        self.assertIn("host pool", "\n".join(logs.output).lower())

    def test_warning_names_the_setting_to_raise(self):
        """An operator has to learn *what to change*, not just that it broke."""
        cache = _StalledCache.build()
        with self.assertLogs("sglang.srt.mem_cache.hiradix_cache", "WARNING") as logs:
            cache.write_backup(_node(), write_back=True)
        self.assertIn("--hicache-size", "\n".join(logs.output))

    def test_warning_is_rate_limited(self):
        """Every write attempt fails once stalled; unthrottled this floods."""
        cache = _StalledCache.build()
        with self.assertLogs("sglang.srt.mem_cache.hiradix_cache", "WARNING") as logs:
            for _ in range(50):
                cache.write_backup(_node(), write_back=True)
        self.assertEqual(len(logs.output), 1)

    def test_warning_repeats_after_the_interval(self):
        """A stall that outlives the interval must keep being visible."""
        cache = _StalledCache.build()
        with self.assertLogs("sglang.srt.mem_cache.hiradix_cache", "WARNING") as logs:
            cache.write_backup(_node(), write_back=True)
            cache._last_offload_stalled_log_s -= _OFFLOAD_STALLED_LOG_INTERVAL_S + 1.0
            cache.write_backup(_node(), write_back=True)
        self.assertEqual(len(logs.output), 2)

    def test_successful_write_stays_quiet(self):
        """The warning must not fire on the healthy path."""
        cache = _StalledCache.build()
        host_indices = MagicMock()
        host_indices.clone.return_value = [7, 8, 9, 10]
        host_indices.__len__ = lambda _self: 4
        cache.cache_controller.write.return_value = host_indices
        cache._track_write_through_node = MagicMock()

        with self.assertNoLogs("sglang.srt.mem_cache.hiradix_cache", "WARNING"):
            written = cache.write_backup(_node(), write_back=True)
        self.assertEqual(written, 4)

    def test_retry_after_eviction_is_not_reported_as_a_stall(self):
        """Only a *second* failed attempt means nothing was evictable.

        ``write_backup`` retries once after ``evict_host``. A first-attempt
        failure that the retry rescues is ordinary pressure, not a stall, and
        warning on it would make the line meaningless.
        """
        cache = _StalledCache.build()
        host_indices = MagicMock()
        host_indices.clone.return_value = [7, 8, 9, 10]
        host_indices.__len__ = lambda _self: 4
        cache.cache_controller.write.side_effect = [None, host_indices]
        cache._track_write_through_node = MagicMock()

        with self.assertNoLogs("sglang.srt.mem_cache.hiradix_cache", "WARNING"):
            written = cache.write_backup(_node(), write_back=True)
        self.assertEqual(written, 4)
        cache.evict_host.assert_called_once()


if __name__ == "__main__":
    unittest.main()
