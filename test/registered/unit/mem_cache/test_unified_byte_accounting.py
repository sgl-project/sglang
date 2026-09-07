# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Byte-conservation verifier (`verify_byte_accounting`) for the unified
2-pool composites.

The unified pool's correctness rests on BYTE bookkeeping -- watermark spans,
holes, pending compaction, frontier ordering inside one shared buffer -- and
a drifted counter admits requests into memory that is not actually free.
Nothing about that is visible to the token-identity leak check, so this
verifier is the only idle-time tripwire for it.

The conservation identity: on a lazy end pool the watermark span equals
live + holes + pending pages at EVERY point of a lifecycle, not just at rest.
"""

import unittest

from test_multi_ended_allocator import TestPagedMultiEndedAllocator as _PagedFixture
from test_multi_ended_allocator import (
    TestUnifiedSWATokenToKVPoolAllocator as _SwaFixture,
)

from sglang.srt.mem_cache.allocator import unified_sub_pool as mea
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _swa_composite():
    inst = _SwaFixture([m for m in dir(_SwaFixture) if m.startswith("test_")][0])
    pool, allocator, kvcache = inst._build()
    return inst, allocator, kvcache


def _paged_pair(lazy: bool):
    inst = _PagedFixture([m for m in dir(_PagedFixture) if m.startswith("test_")][0])
    _pool, full, swa, _fkv, _skv = inst._build()
    full.lazy_compaction = lazy
    return full, swa


class TestHealthyLifecycleReportsClean(unittest.TestCase):
    def test_lazy_end_pool_clean_through_free_and_flush(self):
        full, _swa = _paged_pair(lazy=True)
        self.assertEqual(full._byte_accounting_violations(), [])
        v = full.alloc(full.page_size * 4)
        self.assertEqual(full._byte_accounting_violations(), [])
        full.free(v[: full.page_size * 2])  # lazy: holes, no compaction yet
        self.assertEqual(full._byte_accounting_violations(), [])
        full._flush(urgent=True)
        self.assertEqual(full._byte_accounting_violations(), [])


class TestDriftReportsLoudly(unittest.TestCase):
    """Each mutation models a distinct bookkeeping bug; the verifier must name
    the drifted sub-pool. Without these, a regression in any single counter
    passes every other test -- the pool still 'works', it just lies about
    capacity."""

    def _lazy_full(self):
        full, _swa = _paged_pair(lazy=True)
        v = full.alloc(full.page_size * 4)
        full.free(v[: full.page_size])  # one hole so all three terms are live
        self.assertEqual(full._byte_accounting_violations(), [])
        return full

    def test_any_drifted_term_reports(self):
        def drift_live_count(full):
            full.live_page_count += 1

        def leak_hole(full):
            full._free_phys_pages = full._free_phys_pages[:-1]  # hole vanished

        def drift_watermark(full):
            full.watermark_physical += 1

        for term, mutate in (
            ("live_count", drift_live_count),
            ("leaked_hole", leak_hole),
            ("watermark", drift_watermark),
        ):
            with self.subTest(term=term):
                full = self._lazy_full()
                mutate(full)
                self.assertTrue(
                    any("span" in s for s in full._byte_accounting_violations())
                )

    def test_composite_report_names_the_sub_pool(self):
        """Frontier-bounds drift (checked in BOTH lazy and eager modes): push
        the swa band's watermark outside the buffer."""
        inst, allocator, kvcache = _swa_composite()
        inst._alloc(allocator, kvcache, 8)
        swa = allocator.swa_attn_allocator
        # grow-down member: low frontier = (wm+1)*bytes; wm == num_pages puts
        # it past the buffer top.
        self.assertEqual(swa.grow_direction, "down")
        swa.watermark_physical = swa.num_pages
        out = allocator.verify_byte_accounting()
        self.assertTrue(out and any("[swa]" in s for s in out), out)


class TestChainFrontierOrder(unittest.TestCase):
    def test_overlapping_frontiers_report(self):
        """Both bands hold pages, then the up member's watermark is pushed
        past the down member's LIVE low frontier. Both sides must be populated
        for this to be real corruption: an empty down band cannot overlap,
        its low frontier IS the buffer top."""
        full, swa = _paged_pair(lazy=False)
        chain = mea._end_pair_chain(full, swa)
        up, down = chain
        self.assertEqual(up.grow_direction, "up")
        self.assertIsNotNone(down.alloc(down.page_size * 2))  # down side live
        self.assertLess(down._byte_low_frontier(), up.unified_buffer.total_bytes)
        up.watermark_physical = up.num_pages  # up band swallows the buffer
        out = mea._chain_byte_accounting_violations(chain)
        self.assertTrue(any("overlap" in s for s in out), out)


if __name__ == "__main__":
    unittest.main()
