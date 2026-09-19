import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.dcp_pack import (
    dcp_pack_buffer_bytes,
    try_pack_dcp_src,
)
from sglang.srt.disaggregation.common.utils import (
    build_dcp_token_transfer_plan,
    group_concurrent_contiguous,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _plan(*, src, dst, page_size, dcp_size, dcp_rank, **kwargs):
    return build_dcp_token_transfer_plan(
        np.asarray(src, dtype=np.int32),
        np.asarray(dst, dtype=np.int32),
        physical_page_size=page_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        **kwargs,
    )


class TestDcpTokenTransferPlan(CustomTestCase):
    def test_one_virtual_page_explicit_rows(self):
        # P=2, N=4. Prefill pages 5,2,11,4; decode virtual page 7.
        # pos 0..7 src rows: 10,11, 4,5, 22,23, 8,9
        # draft dest page is P*N=8 → 56..63
        # each rank stores local rows 14,15 (page P=2)
        expected_draft_src = [10, 11, 4, 5, 22, 23, 8, 9]
        expected_draft_dst = list(range(56, 64))
        expected_target_src = {
            0: [10, 22],
            1: [11, 23],
            2: [4, 8],
            3: [5, 9],
        }
        seen_src = []
        for rank, src in expected_target_src.items():
            plan = _plan(
                src=[5, 2, 11, 4],
                dst=[7],
                page_size=2,
                dcp_size=4,
                dcp_rank=rank,
                num_kv_tokens=8,
            )
            np.testing.assert_array_equal(
                plan.draft_src_token_indices, expected_draft_src
            )
            np.testing.assert_array_equal(
                plan.draft_dst_token_indices, expected_draft_dst
            )
            np.testing.assert_array_equal(plan.target_src_token_indices, src)
            np.testing.assert_array_equal(plan.target_dst_token_indices, [14, 15])
            seen_src.extend(plan.target_src_token_indices.tolist())
        self.assertEqual(sorted(seen_src), sorted(expected_draft_src))

    def test_second_chunk_crosses_dest_pages(self):
        # P=2, N=2 (virtual page = 4). Decode already holds a 4-token prefix;
        # dst=[4, 6] is the full send-range page list. This chunk is the second
        # prefill page of the send range (src_page_offset=1), so its 4 tokens
        # sit at send-range pos 2..5 (absolute 6..9) and straddle virtual page
        # 4 (rows 16..19) and virtual page 6 (rows 24..27).
        plan = _plan(
            src=[9, 3],
            dst=[4, 6],
            page_size=2,
            dcp_size=2,
            dcp_rank=0,
            src_page_offset=1,
            decode_prefix_len=4,
            num_kv_tokens=4,
        )
        np.testing.assert_array_equal(plan.draft_src_token_indices, [18, 19, 6, 7])
        np.testing.assert_array_equal(plan.draft_dst_token_indices, [18, 19, 24, 25])
        # rank 0 owns absolute pos 6, 8 -> per-rank slots 1, 2 -> pages 4, 6.
        np.testing.assert_array_equal(plan.target_src_token_indices, [18, 6])
        np.testing.assert_array_equal(plan.target_dst_token_indices, [9, 12])

        plan_r1 = _plan(
            src=[9, 3],
            dst=[4, 6],
            page_size=2,
            dcp_size=2,
            dcp_rank=1,
            src_page_offset=1,
            decode_prefix_len=4,
            num_kv_tokens=4,
        )
        np.testing.assert_array_equal(plan_r1.draft_src_token_indices, [18, 19, 6, 7])
        np.testing.assert_array_equal(plan_r1.draft_dst_token_indices, [18, 19, 24, 25])
        np.testing.assert_array_equal(plan_r1.target_src_token_indices, [19, 7])
        np.testing.assert_array_equal(plan_r1.target_dst_token_indices, [9, 12])

    def test_rejects_unaligned_prefix(self):
        with self.assertRaisesRegex(ValueError, "align"):
            _plan(
                src=[0],
                dst=[0],
                page_size=2,
                dcp_size=4,
                dcp_rank=0,
                decode_prefix_len=1,
                num_kv_tokens=2,
            )

    def test_empty_tokens(self):
        plan = _plan(
            src=[0], dst=[0], page_size=2, dcp_size=4, dcp_rank=0, num_kv_tokens=0
        )
        self.assertTrue(plan.empty())


class TestPackedDcpGrouping(CustomTestCase):
    def test_target_needs_pack_draft_does_not(self):
        plan = _plan(
            src=[0, 1, 2, 3],
            dst=[0],
            page_size=2,
            dcp_size=4,
            dcp_rank=0,
            num_kv_tokens=8,
        )
        np.testing.assert_array_equal(plan.target_src_token_indices, [0, 4])
        np.testing.assert_array_equal(plan.target_dst_token_indices, [0, 1])
        target_src, _ = group_concurrent_contiguous(
            plan.target_src_token_indices, plan.target_dst_token_indices
        )
        self.assertEqual(target_src, [[0], [4]])

        packed_src, packed_dst = group_concurrent_contiguous(
            np.arange(2, dtype=np.int64), plan.target_dst_token_indices
        )
        self.assertEqual(packed_src, [[0, 1]])
        self.assertEqual(packed_dst, [[0, 1]])

        draft_src, draft_dst = group_concurrent_contiguous(
            plan.draft_src_token_indices, plan.draft_dst_token_indices
        )
        self.assertEqual(draft_src, [[0, 1, 2, 3, 4, 5, 6, 7]])
        self.assertEqual(draft_dst, [[0, 1, 2, 3, 4, 5, 6, 7]])


def _dcp_kv_manager_stub(*, page_size, kv_item_lens, num_draft_entries):
    return SimpleNamespace(
        kv_args=SimpleNamespace(
            page_size=page_size,
            kv_item_lens=kv_item_lens,
            num_draft_entries=num_draft_entries,
        )
    )


class TestPrepareDcpTokenItemLens(CustomTestCase):
    def test_draft_tail_scales_by_dst_dcp_size(self):
        mgr = _dcp_kv_manager_stub(
            page_size=64,
            kv_item_lens=[64 * 32, 64 * 32, 64 * 16],
            num_draft_entries=1,
        )
        token_lens = CommonKVManager.prepare_dcp_token_item_lens(
            mgr, [64 * 32, 64 * 32, 4 * 64 * 16], dst_dcp_size=4
        )
        self.assertEqual(token_lens, [32, 32, 16])

    def test_rejects_unscaled_draft_item_len(self):
        mgr = _dcp_kv_manager_stub(
            page_size=64,
            kv_item_lens=[64 * 32, 64 * 16],
            num_draft_entries=1,
        )
        with self.assertRaisesRegex(RuntimeError, "geometry differs at entry 1"):
            CommonKVManager.prepare_dcp_token_item_lens(
                mgr, [64 * 32, 64 * 16], dst_dcp_size=4
            )


class TestDcpPackBufferBytes(CustomTestCase):
    def test_sizes_a_full_max_tokens_region_per_dcp_rank(self):
        # A rank packs only its 1/dcp_size share, but it packs at offset
        # rank * rank_stride, so the highest rank is left just rank_stride
        # bytes. Per-rank capacity is therefore rank_stride / per-token, and
        # num_kv_tokens is not bounded by max_tokens (a prefix cached on
        # prefill but missing on decode ships as one unbounded chunk). Size
        # each region for max_tokens, not max_tokens / dcp_size.
        self.assertEqual(
            dcp_pack_buffer_bytes(
                [64 * 16, 64 * 16],
                page_size=64,
                max_tokens=10,
                dcp_size=4,
            ),
            4 * 10 * (16 + 16),
        )

    def test_rank_region_holds_a_full_chunk(self):
        """The invariant try_pack_dcp_src.fits() depends on.

        `_pack_dcp_rank_once` packs rank r at `offset = r * rank_stride` where
        `rank_stride = size // dcp_size`, so fits() leaves the highest rank
        exactly one stride. A rank's stride must therefore cover the largest
        pack it can ever see. Undersizing it makes fits() fail on that rank
        alone and silently degrade to per-token RDMA -- which is why rank 0
        looks healthy while the top rank regresses.
        """
        item_lens = [64 * 16, 64 * 16]
        per_token = sum(x // 64 for x in item_lens)
        for dcp_size in (1, 2, 4, 8):
            for max_tokens in (1, 10, 16384):
                size = dcp_pack_buffer_bytes(
                    item_lens, page_size=64, max_tokens=max_tokens, dcp_size=dcp_size
                )
                rank_stride = size // dcp_size
                self.assertGreaterEqual(
                    rank_stride,
                    max_tokens * per_token,
                    f"dcp_size={dcp_size} max_tokens={max_tokens}: rank region "
                    f"{rank_stride} B is below the max_tokens pack "
                    f"({max_tokens * per_token} B) -- fits() would fail on the "
                    f"highest rank and fall back to per-token RDMA",
                )

    def test_documents_the_supported_prefix_threshold(self):
        """This sizing raises a threshold; it does not remove one.

        A transfer chunk of `num_kv_tokens` shards to `num_kv_tokens /
        dcp_size` per rank, so the highest rank overflows once
        `num_kv_tokens > dcp_size * (rank_stride / per-token bytes)`. Pinning
        the number here means a future change to the formula has to restate
        what it now supports, instead of silently moving the cliff.
        """
        item_lens = [64 * 16, 64 * 16]
        per_token = sum(x // 64 for x in item_lens)
        max_tokens, dcp_size = 16384, 8
        size = dcp_pack_buffer_bytes(
            item_lens, page_size=64, max_tokens=max_tokens, dcp_size=dcp_size
        )
        supported = dcp_size * ((size // dcp_size) // per_token)
        # Was ceil(max_tokens / dcp_size) * dcp_size == 16,384 before this fix.
        self.assertEqual(supported, 131072)

    def test_rejects_invalid_item_lens(self):
        with self.assertRaisesRegex(ValueError, "at least one page"):
            dcp_pack_buffer_bytes([0], page_size=64, max_tokens=8)
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            dcp_pack_buffer_bytes([100], page_size=64, max_tokens=8)


class TestTryDcpPack(CustomTestCase):
    def test_try_pack_uses_requested_region_and_dense_indices(self):
        dim = 4
        kv = torch.arange(16 * dim, dtype=torch.float32).view(16, 1, dim)
        item_len = int(kv[0].nbytes)
        pack = torch.zeros(8 * item_len, dtype=torch.uint8)
        gather_stream = Mock()
        buf = type(
            "Buf",
            (),
            {
                "buffer": pack,
                "fits": lambda self, n: n <= pack.numel(),
                "get_ptr": lambda self: 0x1000,
                "get_size": lambda self: pack.numel(),
                "get_gather_stream": lambda self: gather_stream,
            },
        )()
        src = np.array([1, 5, 9, 13], dtype=np.int64)
        pack_offset = 2 * item_len
        with (
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.torch.cuda.default_stream"
            ),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.torch.cuda.stream",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.copy_mla_rows_into_pack"
            ) as copy_mock,
        ):
            packed = try_pack_dcp_src(
                pack_buffer=buf,
                kv_data_ptrs=[kv.data_ptr()],
                src_token_indices=src,
                token_item_lens=[item_len],
                pack_offset_bytes=pack_offset,
            )

        gather_stream.synchronize.assert_called_once_with()
        self.assertIsNotNone(packed)
        ptrs, indices = packed
        self.assertEqual(ptrs, [0x1000 + pack_offset])
        np.testing.assert_array_equal(indices, np.arange(4))
        pack_view = copy_mock.call_args.args[2]
        self.assertEqual(pack_view.storage_offset(), pack_offset)
        self.assertEqual(pack_view.numel(), src.size * item_len)


if __name__ == "__main__":
    unittest.main()
