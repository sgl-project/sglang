"""Unit tests for HiRadixCache.bulk_check_prefetch_progress (fix for #30760)."""

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_bare_cache(enable_storage: bool = True) -> HiRadixCache:
    cache = object.__new__(HiRadixCache)
    cache.enable_storage = enable_storage
    return cache


class TestBulkCheckPrefetchProgress(unittest.TestCase):
    def test_returns_empty_dict_when_storage_disabled(self):
        cache = _make_bare_cache(enable_storage=False)
        cache.check_prefetch_progress = MagicMock(return_value=True)

        result = HiRadixCache.bulk_check_prefetch_progress(cache, ["r0", "r1", "r2"])

        self.assertEqual(result, {})
        cache.check_prefetch_progress.assert_not_called()

    def test_returns_empty_dict_for_empty_input(self):
        cache = _make_bare_cache(enable_storage=True)
        cache.check_prefetch_progress = MagicMock(return_value=True)

        result = HiRadixCache.bulk_check_prefetch_progress(cache, [])

        self.assertEqual(result, {})
        cache.check_prefetch_progress.assert_not_called()

    def test_delegates_to_per_req_check(self):
        cache = _make_bare_cache(enable_storage=True)
        returns = {"a": True, "b": False, "c": True}
        cache.check_prefetch_progress = MagicMock(side_effect=lambda rid: returns[rid])

        result = HiRadixCache.bulk_check_prefetch_progress(cache, ["a", "b", "c"])

        self.assertEqual(result, {"a": True, "b": False, "c": True})
        self.assertEqual(cache.check_prefetch_progress.call_count, 3)
        called_rids = [
            args[0] for args, _ in cache.check_prefetch_progress.call_args_list
        ]
        self.assertEqual(called_rids, ["a", "b", "c"])

    def test_call_count_equals_input_length(self):
        cache = _make_bare_cache(enable_storage=True)
        cache.check_prefetch_progress = MagicMock(return_value=True)

        for n in (1, 2, 5, 17, 128):
            with self.subTest(n=n):
                cache.check_prefetch_progress.reset_mock()
                rids = [f"req-{i}" for i in range(n)]
                result = HiRadixCache.bulk_check_prefetch_progress(cache, rids)
                self.assertEqual(len(result), n)
                self.assertEqual(cache.check_prefetch_progress.call_count, n)

    def test_duplicate_req_ids_still_call_per_slot(self):
        cache = _make_bare_cache(enable_storage=True)
        cache.check_prefetch_progress = MagicMock(return_value=True)

        result = HiRadixCache.bulk_check_prefetch_progress(cache, ["r0", "r0", "r1"])

        self.assertEqual(cache.check_prefetch_progress.call_count, 3)
        self.assertEqual(result, {"r0": True, "r1": True})


if __name__ == "__main__":
    unittest.main()
