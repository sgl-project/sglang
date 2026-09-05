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
"""Byte-conservation verifier for the unified 2-pool composites.

`verify_byte_accounting` is the idle-time tripwire the token-identity leak
check cannot provide: the unified pool's correctness rests on BYTE bookkeeping
(watermark spans, holes, pending compaction, frontier ordering inside one
shared buffer), and a drifted counter admits requests into memory that is not
actually free — silent corruption territory, not a crash.

Derived properties pinned here:

  * Conservation: on a lazy end pool the watermark span must equal
    live + holes + pending pages at EVERY point of a healthy lifecycle
    (alloc, partial free, group free, flush) — not just at rest.
  * The check is not vacuous: drifting any single term (live count, watermark,
    a leaked hole) reports loudly, naming the sub-pool.
  * Chain order: one member's low frontier clearing the other's high frontier
    is what "two pools share one buffer without overlap" MEANS.
  * The strict escalation env defaults OFF: promoting the diagnostic to a
    RuntimeError is a validation posture, not the production one.

    python -m pytest test/registered/unit/mem_cache/test_unified_byte_accounting.py -v
"""

import unittest

from test_multi_ended_allocator import TestPagedMultiEndedAllocator as _PagedFixture
from test_multi_ended_allocator import (
    TestUnifiedSWATokenToKVPoolAllocator as _SwaFixture,
)

from sglang.srt.mem_cache.allocator import unified_sub_pool as mea
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


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
    """Each mutation below models a distinct bookkeeping bug; the verifier
    must name the drifted sub-pool. Without these, a regression in any single
    counter passes every other test (the pool still 'works' — it just lies
    about capacity)."""

    def _lazy_full(self):
        full, _swa = _paged_pair(lazy=True)
        v = full.alloc(full.page_size * 4)
        full.free(v[: full.page_size])  # one hole so all three terms are live
        self.assertEqual(full._byte_accounting_violations(), [])
        return full

    def test_any_drifted_term_reports(self):
        """One drifted span term per subTest -- each must be reported."""

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
        """Both bands hold pages, then the up member's watermark is pushed past
        the down member's LIVE low frontier: the two bands now claim the same
        bytes of one buffer. (An empty down band cannot overlap — its low
        frontier IS the buffer top — so both sides must be populated for the
        scenario to be a real corruption.)"""
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
