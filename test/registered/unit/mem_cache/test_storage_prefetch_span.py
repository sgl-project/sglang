"""An L3 prefetch span must stop where the match key stops.

The trailing ``window_reprefill_tail_tokens()`` are re-prefilled, so the matcher
never consumes them. A prefetch reaching past that endpoint ends on a page
whose node carries no SWA host value: admission rejects the boundary and the
already-transferred span is dropped.
"""

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.mem_cache.common import _req_radix_key, storage_prefetch_end
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _req(n):
    return SimpleNamespace(
        full_untruncated_fill_ids=array("q", range(n)),
        extra_key=None,
        cache_salt=None,
        _user_cache_salt=None,
    )


class TestStoragePrefetchEnd(unittest.TestCase):
    def test_no_reprefill_tail_keeps_max_prefix_len(self):
        cache = SimpleNamespace(window_reprefill_tail_tokens=lambda req: 0)
        self.assertEqual(storage_prefetch_end(cache, _req(40), 39), 39)

    def test_tail_caps_the_span(self):
        cache = SimpleNamespace(window_reprefill_tail_tokens=lambda req: 4)
        for salt in (None, "tenant-a"):
            with self.subTest(request_cache_salt=salt):
                req = _req(40)
                req.extra_key = "adapter-a"
                req.cache_salt = "sglang-mm-v1:internal"
                req._user_cache_salt = salt
                key = _req_radix_key(cache, req, req.full_untruncated_fill_ids, 39)
                self.assertIs(key.token_ids, req.full_untruncated_fill_ids)
                self.assertEqual(key.extra_key, req.extra_key)
                self.assertEqual(key.cache_salt, req.cache_salt)
                self.assertEqual(key.request_cache_salt, salt)
                self.assertFalse(key.is_bigram)
                self.assertEqual(key.limit, 36)
                self.assertEqual(storage_prefetch_end(cache, req, 39), 36)

    def test_smaller_max_prefix_len_wins(self):
        cache = SimpleNamespace(window_reprefill_tail_tokens=lambda req: 4)
        self.assertEqual(storage_prefetch_end(cache, _req(40), 30), 30)

    def test_tail_longer_than_prompt(self):
        cache = SimpleNamespace(window_reprefill_tail_tokens=lambda req: 64)
        self.assertEqual(storage_prefetch_end(cache, _req(40), 39), 0)

    def test_boundary_case_stops_one_page_below_the_old_span(self):
        # L = 8216 -> L mod 256 = 24: the reported failure window [1, 127].
        # The old span floored (L - 1) to 8192; the key-driven span must floor
        # the capped end (8088) to 7936, the page the matcher can consume.
        cache = SimpleNamespace(window_reprefill_tail_tokens=lambda req: 128)
        req = _req(8216)
        end = storage_prefetch_end(cache, req, len(req.full_untruncated_fill_ids) - 1)
        self.assertEqual(end // 256 * 256, 8192 - 256)


if __name__ == "__main__":
    unittest.main()
