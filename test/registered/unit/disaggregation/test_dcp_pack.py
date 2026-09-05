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

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
        # Prefix already filled one virtual page (P*N=4). This chunk's 4 tokens
        # start at dest pos 4 and spill from virtual page 4 onto page 6.
        plan = _plan(
            src=[9, 3],
            dst=[4, 6],
            page_size=2,
            dcp_size=2,
            dcp_rank=0,
            src_page_offset=2,
            decode_prefix_len=4,
            num_kv_tokens=4,
        )
        np.testing.assert_array_equal(plan.draft_src_token_indices, [18, 19, 6, 7])
        np.testing.assert_array_equal(plan.draft_dst_token_indices, [16, 17, 26, 27])
        np.testing.assert_array_equal(plan.target_src_token_indices, [18, 6])
        np.testing.assert_array_equal(plan.target_dst_token_indices, [12, 13])

        plan_r1 = _plan(
            src=[9, 3],
            dst=[4, 6],
            page_size=2,
            dcp_size=2,
            dcp_rank=1,
            src_page_offset=2,
            decode_prefix_len=4,
            num_kv_tokens=4,
        )
        np.testing.assert_array_equal(plan_r1.draft_src_token_indices, [18, 19, 6, 7])
        np.testing.assert_array_equal(plan_r1.target_src_token_indices, [19, 7])
        np.testing.assert_array_equal(plan_r1.target_dst_token_indices, [12, 13])

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
    def test_sizes_fixed_regions_for_each_dcp_rank(self):
        self.assertEqual(
            dcp_pack_buffer_bytes(
                [64 * 16, 64 * 16],
                page_size=64,
                max_tokens=10,
                dcp_size=4,
            ),
            4 * 3 * (16 + 16),
        )

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
