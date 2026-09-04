"""Tests for the model-agnostic KV region descriptors (host offload addressing).

CPU-only: these exercise the four addressing schemes and the save/load engine on
plain tensors, plus a cross-check that ``ReqScoped`` reproduces the row math the
PD disaggregation path already uses for DSV4 C128 state.

    python -m pytest test/registered/mem_cache/test_kv_region.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.kv_region import (
    KVRegion,
    PageAligned,
    ReqScoped,
    RequestCtx,
    SwaMapped,
    SwaPageRing,
    load_regions,
    save_regions,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _ctx(*, start: int, seq_len: int, req_pool_idx: int = 0) -> RequestCtx:
    return RequestCtx(
        token_indices=torch.arange(start, start + seq_len, dtype=torch.int64),
        req_pool_idx=req_pool_idx,
    )


class TestPageAligned(unittest.TestCase):
    def test_rows_cover_every_touched_row(self):
        rows = PageAligned(stride=256).save_plan(_ctx(start=0, seq_len=600))[0]
        self.assertEqual(rows.tolist(), [0, 1, 2])

    def test_row_count_depends_only_on_seq_len(self):
        # Save and load see different absolute indices; the count must match so
        # that row k means the same thing on both sides.
        addressing = PageAligned(stride=256)
        saved = addressing.save_plan(_ctx(start=0, seq_len=600))[0]
        loaded = addressing.save_plan(_ctx(start=2560, seq_len=600))[0]
        self.assertEqual(saved.numel(), loaded.numel())
        self.assertEqual(loaded.tolist(), [10, 11, 12])

    def test_single_row_when_shorter_than_stride(self):
        rows = PageAligned(stride=256).save_plan(_ctx(start=512, seq_len=3))[0]
        self.assertEqual(rows.tolist(), [2])

    def test_exact_multiple_of_stride(self):
        rows = PageAligned(stride=256).save_plan(_ctx(start=0, seq_len=512))[0]
        self.assertEqual(rows.tolist(), [0, 1])


class TestSwaMapped(unittest.TestCase):
    def _mapping(self, mapped_from: int) -> torch.Tensor:
        """Tokens below ``mapped_from`` sit on the reserved dummy slot 0."""
        mapping = torch.zeros(600, dtype=torch.int64)
        mapping[mapped_from:600] = torch.arange(mapped_from, 600, dtype=torch.int64)
        return mapping

    def test_unmapped_pages_are_dropped(self):
        rows, side = SwaMapped(mapping=self._mapping(300), page_size=100).save_plan(
            _ctx(start=0, seq_len=600)
        )
        # Token pages 3,4,5 of the request are mapped; 0,1,2 are not.
        self.assertEqual(rows.tolist(), [3, 4, 5])
        self.assertEqual(side.tolist(), [False, False, False, True, True, True])

    def test_all_unmapped_gives_no_rows(self):
        mapping = torch.zeros(600, dtype=torch.int64)
        rows, _ = SwaMapped(mapping=mapping, page_size=100).save_plan(
            _ctx(start=0, seq_len=600)
        )
        self.assertEqual(rows.numel(), 0)

    def test_load_restores_only_pages_mapped_on_both_sides(self):
        """Regression: a request retracted with three SWA pages live can resume
        with only one still mapped -- the allocator's ring state decides, not
        seq_len. The load side must restore the intersection and pick the
        matching saved rows instead of assuming the counts agree."""
        save_side = SwaMapped(mapping=self._mapping(300), page_size=100)
        load_side = SwaMapped(mapping=self._mapping(500), page_size=100)
        ctx = _ctx(start=0, seq_len=600)

        saved_rows, side = save_side.save_plan(ctx)
        self.assertEqual(saved_rows.numel(), 3)

        rows, source = load_side.load_plan(ctx, side)
        self.assertEqual(rows.tolist(), [5])
        # Saved rows were pages 3,4,5 in order, so page 5 is saved row 2.
        self.assertEqual(source.tolist(), [2])

    def test_load_skips_pages_that_were_unmapped_at_save(self):
        save_side = SwaMapped(mapping=self._mapping(500), page_size=100)
        load_side = SwaMapped(mapping=self._mapping(300), page_size=100)
        ctx = _ctx(start=0, seq_len=600)
        _, side = save_side.save_plan(ctx)
        rows, source = load_side.load_plan(ctx, side)
        self.assertEqual(rows.tolist(), [5])
        self.assertEqual(source.tolist(), [0])


class TestSwaPageRing(unittest.TestCase):
    def test_expands_each_page_to_a_ring_block(self):
        mapping = torch.arange(400, dtype=torch.int64)
        mapping[0] = 1  # keep slot 0 out of the way of the >0 filter
        rows, _ = SwaPageRing(
            mapping=mapping, swa_page_size=100, ring_size=4
        ).save_plan(_ctx(start=0, seq_len=200))
        # pages {0, 1} -> blocks [0..3] and [4..7]
        self.assertEqual(rows.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])

    def test_load_intersects_and_expands_source_blocks(self):
        mapping_save = torch.arange(400, dtype=torch.int64)
        mapping_save[0] = 1
        mapping_load = torch.zeros(400, dtype=torch.int64)
        mapping_load[100:200] = torch.arange(100, 200, dtype=torch.int64)
        ctx = _ctx(start=0, seq_len=200)

        _, side = SwaPageRing(
            mapping=mapping_save, swa_page_size=100, ring_size=4
        ).save_plan(ctx)
        rows, source = SwaPageRing(
            mapping=mapping_load, swa_page_size=100, ring_size=4
        ).load_plan(ctx, side)
        # Only token page 1 survives: its ring block is rows 4..7, fed by the
        # second saved block (saved rows 4..7).
        self.assertEqual(rows.tolist(), [4, 5, 6, 7])
        self.assertEqual(source.tolist(), [4, 5, 6, 7])


class TestReqScoped(unittest.TestCase):
    def test_plain_slot_row(self):
        rows = ReqScoped().save_plan(_ctx(start=0, seq_len=600, req_pool_idx=7))[0]
        self.assertEqual(rows.tolist(), [7])

    def test_no_rows_on_block_boundary(self):
        rows, _ = ReqScoped(rows_per_req=1, block_rows=1, block_tokens=128).save_plan(
            _ctx(start=0, seq_len=512, req_pool_idx=7)
        )
        self.assertEqual(rows.numel(), 0)

    def test_multi_row_block_is_returned_whole(self):
        # A per-request ring of 256 rows in two 128-row blocks; seq_len 600 has
        # 599 % 256 == 87 -> block 0 of slot 2.
        rows, _ = ReqScoped(
            rows_per_req=256, block_rows=128, block_tokens=128
        ).save_plan(_ctx(start=0, seq_len=600, req_pool_idx=2))
        self.assertEqual(rows.numel(), 128)
        self.assertEqual(rows[0].item(), 2 * 256)
        self.assertEqual(rows[-1].item(), 2 * 256 + 127)

    def test_second_block_of_the_ring(self):
        # 727 % 256 == 215 -> block 1.
        rows, _ = ReqScoped(
            rows_per_req=256, block_rows=128, block_tokens=128
        ).save_plan(_ctx(start=0, seq_len=728, req_pool_idx=2))
        self.assertEqual(rows[0].item(), 2 * 256 + 128)

    def test_matches_pd_c128_state_indices(self):
        """``get_dsv4_c128_state_indices`` returns a *block* index (the PD state
        component's item is 128 rows offline, 1 row online); check the rows we
        touch are exactly that block."""
        from sglang.srt.disaggregation.utils import get_dsv4_c128_state_indices

        for req_pool_idx in (0, 3, 11):
            for seq_len in (1, 127, 128, 129, 255, 256, 600, 728, 1024):
                with self.subTest(req_pool_idx=req_pool_idx, seq_len=seq_len):
                    ctx = _ctx(start=0, seq_len=seq_len, req_pool_idx=req_pool_idx)

                    online = ReqScoped(rows_per_req=1, block_rows=1, block_tokens=128)
                    self.assertEqual(
                        online.save_plan(ctx)[0].tolist(),
                        get_dsv4_c128_state_indices(
                            req_pool_idx, seq_len, online=True, ring_size=1
                        ).tolist(),
                    )

                    for ring_size in (128, 256):
                        offline = ReqScoped(
                            rows_per_req=ring_size, block_rows=128, block_tokens=128
                        )
                        blocks = get_dsv4_c128_state_indices(
                            req_pool_idx, seq_len, online=False, ring_size=ring_size
                        )
                        expected = [
                            int(block) * 128 + offset
                            for block in blocks
                            for offset in range(128)
                        ]
                        self.assertEqual(offline.save_plan(ctx)[0].tolist(), expected)


class TestRegionRoundTrip(unittest.TestCase):
    def _regions(self, paged, per_req, reset=None):
        return [
            KVRegion(
                name="paged",
                tensors=(paged,),
                addressing=PageAligned(stride=4),
            ),
            KVRegion(
                name="per_req",
                tensors=(per_req,),
                addressing=ReqScoped(),
                reset_before_load=reset,
            ),
        ]

    def test_round_trip_to_new_rows_and_new_req_slot(self):
        paged = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        per_req = torch.arange(5, dtype=torch.float32)
        regions = self._regions(paged, per_req)

        save_ctx = _ctx(start=0, seq_len=8, req_pool_idx=1)
        host = save_regions(regions=regions, ctx=save_ctx)
        expected_paged = paged[[0, 1]].clone()
        expected_per_req = per_req[[1]].clone()

        paged.zero_()
        per_req.zero_()

        # Resume lands on different pages and a different req slot.
        load_ctx = _ctx(start=16, seq_len=8, req_pool_idx=4)
        load_regions(regions=regions, host=host, ctx=load_ctx)

        torch.testing.assert_close(paged[[4, 5]], expected_paged)
        torch.testing.assert_close(per_req[[4]], expected_per_req)
        self.assertTrue(torch.all(paged[[0, 1, 2, 3, 6, 7, 8, 9]] == 0))

    def test_empty_region_is_saved_as_none(self):
        per_req = torch.zeros(5)
        regions = [
            KVRegion(
                name="gated",
                tensors=(per_req,),
                addressing=ReqScoped(rows_per_req=1, block_rows=1, block_tokens=128),
            )
        ]
        host = save_regions(regions=regions, ctx=_ctx(start=0, seq_len=256))
        self.assertIsNone(host["gated"])
        # Loading a None region is a no-op rather than an error.
        load_regions(regions=regions, host=host, ctx=_ctx(start=0, seq_len=256))

    def test_reset_before_load_runs_even_when_nothing_was_saved(self):
        seen = []
        regions = [
            KVRegion(
                name="gated",
                tensors=(torch.zeros(5),),
                addressing=ReqScoped(rows_per_req=1, block_rows=1, block_tokens=128),
                reset_before_load=seen.append,
            )
        ]
        host = save_regions(regions=regions, ctx=_ctx(start=0, seq_len=256))
        load_regions(
            regions=regions, host=host, ctx=_ctx(start=0, seq_len=256, req_pool_idx=9)
        )
        self.assertEqual(seen, [9])

    def test_asymmetric_row_count_is_rejected(self):
        paged = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        regions = [
            KVRegion(name="paged", tensors=(paged,), addressing=PageAligned(stride=4))
        ]
        host = save_regions(regions=regions, ctx=_ctx(start=0, seq_len=8))
        with self.assertRaisesRegex(AssertionError, "not save/load symmetric"):
            load_regions(regions=regions, host=host, ctx=_ctx(start=0, seq_len=4))


if __name__ == "__main__":
    unittest.main()
