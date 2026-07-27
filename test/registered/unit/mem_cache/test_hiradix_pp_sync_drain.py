"""Unit test for HiRadixCache._drain_async_work PP-sync backpressure."""

import unittest

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _Holder:
    """Minimal carrier exposing only what _drain_async_work touches."""


class TestPPSyncDrain(unittest.TestCase):
    def _drain_fns(self):
        return (HiRadixCache._drain_async_work, UnifiedRadixCache._drain_async_work)

    def test_drain_waits_all_and_clears(self):
        for drain in self._drain_fns():
            holder = _Holder()
            works = [_FakeWork(), _FakeWork(), _FakeWork()]
            holder.work_list = list(works)

            drain(holder)

            self.assertTrue(all(w.waited for w in works))
            self.assertEqual(holder.work_list, [])

    def test_drain_empty_is_noop(self):
        for drain in self._drain_fns():
            holder = _Holder()
            holder.work_list = []

            drain(holder)

            self.assertEqual(holder.work_list, [])


if __name__ == "__main__":
    unittest.main()
