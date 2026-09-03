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


class TestPackedDcpGrouping(CustomTestCase):
    def test_packed_groups_collapse_cyclic_src(self):
        page_size = 64
        dcp_size = 4
        src_pages = np.arange(4, dtype=np.int32)
        dst_pages = np.array([7], dtype=np.int32)
        plan = build_dcp_token_transfer_plan(
            src_pages,
            dst_pages,
            physical_page_size=page_size,
            dcp_size=dcp_size,
            dcp_rank=0,
            num_kv_tokens=256,
        )
        raw_src, _ = group_concurrent_contiguous(
            plan.target_src_token_indices, plan.target_dst_token_indices
        )
        self.assertEqual(len(raw_src), 64)
        self.assertTrue(all(len(group) == 1 for group in raw_src))

        packed_src = np.arange(plan.target_dst_token_indices.size, dtype=np.int64)
        packed_groups, _ = group_concurrent_contiguous(
            packed_src, plan.target_dst_token_indices
        )
        self.assertEqual(len(packed_groups), 1)
        self.assertEqual(len(packed_groups[0]), 64)


class TestReplicatedDcpPlan(CustomTestCase):
    def test_replicated_rows_consistent_with_strided_plan(self):
        page_size = 64
        dcp_size = 4
        virtual_page_size = page_size * dcp_size
        src_pages = np.array([5, 2, 11, 4], dtype=np.int32)
        dst_pages = np.array([7], dtype=np.int32)

        offsets = np.arange(256, dtype=np.int64)
        for dcp_rank in range(dcp_size):
            plan = build_dcp_token_transfer_plan(
                src_pages,
                dst_pages,
                physical_page_size=page_size,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                num_kv_tokens=256,
            )
            np.testing.assert_array_equal(
                plan.draft_src_token_indices,
                src_pages.astype(np.int64)[offsets // page_size] * page_size
                + offsets % page_size,
            )
            np.testing.assert_array_equal(
                plan.draft_dst_token_indices, 7 * virtual_page_size + offsets
            )
            owned = plan.draft_dst_token_indices % dcp_size == dcp_rank
            np.testing.assert_array_equal(
                plan.draft_dst_token_indices[owned] // dcp_size,
                plan.target_dst_token_indices,
            )
            np.testing.assert_array_equal(
                plan.draft_src_token_indices[owned], plan.target_src_token_indices
            )

    def test_second_chunk_continues_the_delta_space(self):
        page_size = 64
        dcp_size = 2
        virtual_page_size = page_size * dcp_size
        src_pages = np.array([9, 3], dtype=np.int32)
        dst_pages = np.array([4, 6], dtype=np.int32)
        plan = build_dcp_token_transfer_plan(
            src_pages,
            dst_pages,
            physical_page_size=page_size,
            dcp_size=dcp_size,
            dcp_rank=0,
            src_page_offset=2,
            decode_prefix_len=virtual_page_size,
            num_kv_tokens=128,
        )
        relative = 2 * page_size + np.arange(128, dtype=np.int64)
        np.testing.assert_array_equal(
            plan.draft_dst_token_indices,
            np.where(relative < virtual_page_size, 4, 6) * virtual_page_size
            + relative % virtual_page_size,
        )


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
