"""Unit tests for heterogeneous-TP head slicing: compute_head_slice_params and the
paged segment layouts built on it.

Wrong head indices do not fail loudly: the transfer delivers the wrong channels
and the only symptom is a garbled end-to-end accuracy score. The replication
cases are the ones no end-to-end test can reach -- every heterogeneous-TP suite
runs tp <= total_kv_heads on both sides, where a modulo map and the correct
divide-by-replication map agree.

Expected values are derived by hand from the head-distribution rules, never by
calling the implementation, so a bug in it cannot make both sides agree.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


import unittest

import numpy as np

from sglang.srt.disaggregation.common.head_slice_layout import (
    page_row_head_segments,
    paged_segment_addresses,
    token_major_head_segments,
)
from sglang.srt.disaggregation.common.staging_buffer import compute_head_slice_params
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import (
    main_page_components,
    page_row_layout,
)
from sglang.test.test_utils import CustomTestCase


class TestHeadSliceParamsScatterReplication(CustomTestCase):
    """More decode ranks than KV heads: consecutive decode ranks share a head."""

    SRC_TP, DST_TP, KV_HEADS = 1, 4, 2

    def _call(self, dst_rank):
        return compute_head_slice_params(
            self.SRC_TP, self.DST_TP, 0, dst_rank, self.KV_HEADS
        )

    def test_replicating_decode_ranks_read_the_same_head(self):
        starts = [self._call(d)[0] for d in range(4)]
        self.assertEqual(starts, [0, 0, 1, 1])
        self.assertNotEqual(starts, [0, 1, 0, 1], "a modulo map would give this")
        for dst_rank in range(self.DST_TP):
            self.assertEqual(
                self._call(dst_rank)[1], 1, "max(1, 2 // 4) clamps to one head"
            )


class TestHeadSliceParamsGatherReplication(CustomTestCase):
    """More prefill ranks than KV heads -- the src_replication branch."""

    SRC_TP, DST_TP, KV_HEADS = 8, 2, 4

    def _call(self, src_rank):
        return compute_head_slice_params(
            self.SRC_TP, self.DST_TP, src_rank, 0, self.KV_HEADS
        )

    def test_replicating_prefill_ranks_write_the_same_head(self):
        starts = [self._call(r)[2] for r in range(8)]
        self.assertEqual(starts, [0, 0, 1, 1, 0, 0, 1, 1])
        self.assertNotEqual(
            starts, [0, 1, 0, 1, 0, 1, 0, 1], "a modulo map would give this"
        )
        for src_rank in range(self.SRC_TP):
            self.assertEqual(
                self._call(src_rank)[1], 1, "max(1, 4 // 8) clamps to one head"
            )


def _simulate_paged_send(*, segments_for, src_tp, dst_tp, layout, pages):
    """Copy every prefill rank's pages into decode rows through the computed segments.

    ``layout(tp)`` returns (row bytes, fill(rank) -> rows of shape [pages, row bytes]).
    """
    src_row, src_fill = layout(src_tp)
    dst_row, _ = layout(dst_tp)
    dst_rows = {d: np.zeros((pages, dst_row), dtype=np.uint8) for d in range(dst_tp)}
    page_ids = np.arange(pages)
    for dst_rank in range(dst_tp):
        if dst_tp >= src_tp:
            src_ranks = [dst_rank // (dst_tp // src_tp)]
        else:
            ratio = src_tp // dst_tp
            src_ranks = range(dst_rank * ratio, (dst_rank + 1) * ratio)
        for src_rank in src_ranks:
            src = src_fill(src_rank).reshape(-1)
            out = dst_rows[dst_rank].reshape(-1)
            s_addr, d_addr, lengths = paged_segment_addresses(
                src_base_ptr=0,
                dst_base_ptr=0,
                src_page_indices=page_ids,
                dst_page_indices=page_ids,
                src_page_bytes=src_row,
                dst_page_bytes=dst_row,
                segments=segments_for(src_rank, dst_rank),
            )
            for a, b, n in zip(s_addr, d_addr, lengths):
                out[b : b + n] = src[a : a + n]
    return dst_rows


class TestPageRowHeadSlice(CustomTestCase):
    """The NVFP4 main KV row is [K, K scale, V, V scale], each [heads, page, bytes]
    and padded to 128 B, so the row width and every component offset depend on
    the per-rank head count. A send that reuses the prefill offsets, or slices
    per token as for token-major KV, lands bytes in the wrong component or head;
    nothing fails, only accuracy drops."""

    KV_HEADS, HEAD_DIM, PAGE, PAGES = 4, 128, 128, 3

    def _head_bytes(self, head, component, page, nbytes):
        rng = np.random.default_rng(1000 * head + 10 * component + page)
        return rng.integers(1, 256, size=nbytes, dtype=np.uint8)

    def _layout(self, tp):
        heads = self.KV_HEADS // tp
        components = main_page_components(heads, self.HEAD_DIM, self.PAGE)
        offsets, row = page_row_layout(components)

        def fill(rank):
            rows = np.zeros((self.PAGES, row), dtype=np.uint8)
            for page in range(self.PAGES):
                for c, ((_, (_, tokens, nbytes)), base) in enumerate(
                    zip(components, offsets)
                ):
                    run = tokens * nbytes
                    for local in range(heads):
                        head = rank * heads + local
                        start = base + local * run
                        rows[page, start : start + run] = self._head_bytes(
                            head, c, page, run
                        )
            return rows

        return row, fill

    def _check(self, src_tp, dst_tp):
        def segments_for(src_rank, dst_rank):
            segments, _, _ = page_row_head_segments(
                total_kv_heads=self.KV_HEADS,
                head_dim=self.HEAD_DIM,
                page_size=self.PAGE,
                src_attn_tp_size=src_tp,
                dst_attn_tp_size=dst_tp,
                src_tp_rank=src_rank,
                dst_tp_rank=dst_rank,
            )
            return segments

        received = _simulate_paged_send(
            segments_for=segments_for,
            src_tp=src_tp,
            dst_tp=dst_tp,
            layout=self._layout,
            pages=self.PAGES,
        )
        _, expected_fill = self._layout(dst_tp)
        for dst_rank, rows in received.items():
            np.testing.assert_array_equal(rows, expected_fill(dst_rank))

    def test_scatter_to_more_decode_ranks(self):
        self._check(src_tp=2, dst_tp=4)

    def test_gather_to_fewer_decode_ranks(self):
        self._check(src_tp=4, dst_tp=2)

    def test_row_widths_follow_each_side_head_count(self):
        _, src_row, dst_row = page_row_head_segments(
            total_kv_heads=4,
            head_dim=128,
            page_size=128,
            src_attn_tp_size=2,
            dst_attn_tp_size=4,
            src_tp_rank=0,
            dst_tp_rank=1,
        )
        # Two heads: 2 x (16 KiB payload + 2 KiB scale); one head: half of that.
        self.assertEqual((src_row, dst_row), (36864, 18432))

    def test_rejects_replicated_heads(self):
        with self.assertRaises(ValueError):
            page_row_head_segments(
                total_kv_heads=4,
                head_dim=128,
                page_size=128,
                src_attn_tp_size=2,
                dst_attn_tp_size=8,
                src_tp_rank=0,
                dst_tp_rank=0,
            )


class TestTokenMajorHeadSlice(CustomTestCase):
    """Draft KV is [token, head, head_bytes]: one run per token, at offsets that
    use each side's own token stride."""

    KV_HEADS, HEAD_BYTES, PAGE, PAGES = 4, 128, 8, 2

    def _layout(self, tp):
        heads = self.KV_HEADS // tp
        item = self.PAGE * heads * self.HEAD_BYTES

        def fill(rank):
            rows = np.zeros((self.PAGES, self.PAGE, heads, self.HEAD_BYTES), np.uint8)
            for local in range(heads):
                rng = np.random.default_rng(rank * heads + local)
                rows[:, :, local, :] = rng.integers(
                    1, 256, size=(self.PAGES, self.PAGE, self.HEAD_BYTES)
                )
            return rows.reshape(self.PAGES, item)

        return item, fill

    def _check(self, src_tp, dst_tp):
        src_item, _ = self._layout(src_tp)
        dst_item, _ = self._layout(dst_tp)

        def segments_for(src_rank, dst_rank):
            return token_major_head_segments(
                total_kv_heads=self.KV_HEADS,
                src_item_len=src_item,
                dst_item_len=dst_item,
                page_size=self.PAGE,
                src_attn_tp_size=src_tp,
                dst_attn_tp_size=dst_tp,
                src_tp_rank=src_rank,
                dst_tp_rank=dst_rank,
            )

        received = _simulate_paged_send(
            segments_for=segments_for,
            src_tp=src_tp,
            dst_tp=dst_tp,
            layout=self._layout,
            pages=self.PAGES,
        )
        _, expected_fill = self._layout(dst_tp)
        for dst_rank, rows in received.items():
            np.testing.assert_array_equal(rows, expected_fill(dst_rank))

    def test_scatter_to_more_decode_ranks(self):
        self._check(src_tp=2, dst_tp=4)

    def test_gather_to_fewer_decode_ranks(self):
        self._check(src_tp=4, dst_tp=2)



if __name__ == "__main__":
    unittest.main()
