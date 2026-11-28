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
2-pool composites, and the SWA composite's shared virtual id space.

The unified pool's correctness rests on BYTE bookkeeping -- watermark spans,
holes, pending compaction, frontier ordering inside one shared buffer -- and
a drifted counter admits requests into memory that is not actually free.
Nothing about that is visible to the token-identity leak check, so this
verifier is the only idle-time tripwire for it.

The conservation identity: on a lazy end pool the watermark span equals
live + holes + pending pages at EVERY point of a lifecycle, not just at rest.
"""

import unittest

from sglang.srt.mem_cache.kv.alloc import band as mea
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.unified_allocator_fixtures import (
    build_paged_band_pair,
    build_window_chain,
)

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _paged_pair(lazy: bool):
    _pool, full, swa, _fkv, _skv = build_paged_band_pair(page_size=8)
    full.lazy_compaction = lazy
    return full, swa


class TestHealthyLifecycleReportsClean(unittest.TestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

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

    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

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
        _pool, allocator, _kvcache = build_window_chain()
        self.assertIsNotNone(allocator.alloc(8))
        swa = allocator.swa_attn_allocator
        # grow-down member: low frontier = (wm+1)*bytes; wm == num_pages puts
        # it past the buffer top.
        self.assertEqual(swa.grow_direction, "down")
        swa.watermark_physical = swa.num_pages
        out = allocator.verify_byte_accounting()
        self.assertTrue(out and any("[swa]" in s for s in out), out)


class TestChainFrontierOrder(unittest.TestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_overlapping_frontiers_report(self):
        """Both bands hold pages, then the up member's watermark is pushed
        past the down member's LIVE low frontier. Both sides must be populated
        for this to be real corruption: an empty down band cannot overlap,
        its low frontier IS the buffer top."""
        full, swa = _paged_pair(lazy=True)
        chain = mea._end_pair_chain(full, swa)
        up, down = chain
        self.assertEqual(up.grow_direction, "up")
        self.assertIsNotNone(down.alloc(down.page_size * 2))  # down side live
        self.assertLess(down._byte_low_frontier(), up.arena.total_bytes)
        up.watermark_physical = up.num_pages  # up band swallows the buffer
        out = mea.chain_byte_accounting_violations(chain)
        self.assertTrue(any("overlap" in s for s in out), out)


class TestSharedVirtualIdSpace(unittest.TestCase):
    """The SWA composite mints one virtual page id and binds it on both sides,
    so the swa side's `virtual_to_physical` is indexed by the FULL side's ids.
    Which side gets more pages out of a shared byte budget depends on the layer
    split (gemma-4: 10 full and 50 sliding layers), and a narrow table fails
    only after churn, on GPU as an unchecked `tl.store` past its end."""

    @staticmethod
    def _build(n_full, n_swa, full_layers, swa_layers):
        from sglang.test.unified_allocator_fixtures import setup_allocator_context

        setup_allocator_context()
        _arena, allocator, _kvcache = build_window_chain(
            n_full_slots=n_full,
            n_swa_slots=n_swa,
            full_layer_num=full_layers,
            swa_layer_num=swa_layers,
        )
        return allocator

    def test_swa_table_spans_the_owners_id_space(self):
        """Static form: the table has to be wide enough before any alloc runs."""
        for full_layers, swa_layers in ((1, 5), (5, 1), (2, 2)):
            with self.subTest(full_layers=full_layers, swa_layers=swa_layers):
                alloc = self._build(200, 20, full_layers, swa_layers)
                owner = alloc.full_attn_allocator
                swa = alloc.swa_attn_allocator
                self.assertGreaterEqual(
                    int(swa.ids.virtual_to_physical.shape[0]),
                    owner.ids.num_virtual_ids + 1,
                    "swa v2p cannot address every id the owner can mint",
                )
                # p2v stays this pool's own business: it is indexed by physical id.
                self.assertEqual(
                    int(swa.ids.physical_to_virtual.shape[0]), swa.num_pages + 1
                )

    def test_churn_past_the_swa_page_count_binds_cleanly(self):
        """Dynamic form: alloc/free until the owner's cursor passes the swa
        side's page count, which is where the narrow table used to be written
        off the end."""
        alloc = self._build(200, 20, full_layers=1, swa_layers=5)
        swa_pages = alloc.swa_attn_allocator.num_pages
        highest = 0
        for _ in range(40):
            v = alloc.alloc(4)
            if v is None:
                break
            highest = max(highest, int(v.max()) // alloc.page_size)
            alloc.free(v)
        self.assertGreater(
            highest,
            swa_pages,
            f"churn never reached past the swa side's {swa_pages} pages, so this "
            "test would pass on a narrow table too",
        )


if __name__ == "__main__":
    unittest.main()
