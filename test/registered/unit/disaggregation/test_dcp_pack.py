"""CPU unit tests for DCP1→DCP-N packed PD transfer planning and gather."""

import unittest

import numpy as np
import torch

from sglang.srt.disaggregation.common.dcp_pack import (
    dcp_pack_buffer_bytes,
    gather_mla_owned_tokens,
    plan_packed_dcp_blocks,
    try_pack_dcp_src,
)
from sglang.srt.disaggregation.common.utils import (
    build_dcp_token_transfer_plan,
    group_concurrent_contiguous,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPlanPackedDcpBlocks(CustomTestCase):
    def test_empty(self):
        self.assertEqual(plan_packed_dcp_blocks(np.array([], dtype=np.int64)), [])

    def test_one_contiguous_dest_page(self):
        dst = np.arange(64, dtype=np.int64) + 10 * 64
        self.assertEqual(plan_packed_dcp_blocks(dst), [(0, 640, 64)])

    def test_splits_on_dest_page_gap(self):
        dst = np.concatenate(
            [
                np.arange(64, dtype=np.int64) + 3 * 64,
                np.arange(64, dtype=np.int64) + 8 * 64,
            ]
        )
        self.assertEqual(
            plan_packed_dcp_blocks(dst),
            [(0, 192, 64), (64, 512, 64)],
        )

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
        raw_src, raw_dst = group_concurrent_contiguous(
            plan.src_token_indices, plan.dst_token_indices
        )
        self.assertEqual(len(raw_src), 64)
        self.assertTrue(all(len(g) == 1 for g in raw_src))

        packed_src = np.arange(plan.dst_token_indices.size, dtype=np.int64)
        packed_groups, _ = group_concurrent_contiguous(
            packed_src, plan.dst_token_indices
        )
        self.assertEqual(len(packed_groups), 1)
        self.assertEqual(len(packed_groups[0]), 64)


class TestDcpPackBufferBytes(CustomTestCase):
    def test_sizes_from_page_item_lens(self):
        with envs.SGLANG_DISAGG_DCP_PACK_MAX_TOKENS.override(128):
            self.assertEqual(
                dcp_pack_buffer_bytes([64 * 16, 64 * 16], page_size=64),
                128 * (16 + 16),
            )

    def test_rejects_non_page_aligned_item_lens(self):
        with envs.SGLANG_DISAGG_DCP_PACK_MAX_TOKENS.override(8):
            with self.assertRaises(ValueError):
                dcp_pack_buffer_bytes([100], page_size=64)


class TestGatherMlaOwnedTokens(CustomTestCase):
    def test_gathers_strided_rows_layer_major(self):
        dim = 8
        kv0 = torch.arange(32 * dim, dtype=torch.float32).view(32, 1, dim)
        kv1 = kv0 + 1000
        src = np.array([0, 4, 8, 12], dtype=np.int64)
        item_len = int(kv0[0].nbytes)
        pack = torch.zeros(2 * src.size * item_len, dtype=torch.uint8)
        gather_mla_owned_tokens([kv0, kv1], src, pack, [item_len, item_len], gpu_id=0)

        packed0 = pack[: src.size * item_len].view(torch.float32).view(4, 1, dim)
        packed1 = pack[src.size * item_len :].view(torch.float32).view(4, 1, dim)
        torch.testing.assert_close(packed0, kv0[src])
        torch.testing.assert_close(packed1, kv1[src])

    def test_try_pack_returns_dense_src_indices(self):
        dim = 4
        kv = torch.arange(16 * dim, dtype=torch.float32).view(16, 1, dim)
        item_len = int(kv[0].nbytes)
        pack = torch.zeros(8 * item_len, dtype=torch.uint8)
        buf = type(
            "Buf",
            (),
            {
                "buffer": pack,
                "fits": lambda self, n: n <= pack.numel(),
                "get_ptr": lambda self: 0x1000,
                "get_size": lambda self: pack.numel(),
            },
        )()
        src = np.array([1, 5, 9, 13], dtype=np.int64)
        packed = try_pack_dcp_src(
            pack_buffer=buf,
            kv_buffers=[kv],
            src_token_indices=src,
            token_item_lens=[item_len],
            gpu_id=0,
        )
        self.assertIsNotNone(packed)
        ptrs, indices = packed
        self.assertEqual(ptrs, [0x1000])
        np.testing.assert_array_equal(indices, np.arange(4))


if __name__ == "__main__":
    unittest.main()
