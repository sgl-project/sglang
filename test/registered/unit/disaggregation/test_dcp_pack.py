import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import numpy as np
import torch

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
        self.assertEqual(
            dcp_pack_buffer_bytes([64 * 16, 64 * 16], page_size=64, max_tokens=128),
            128 * (16 + 16),
        )

    def test_rejects_non_page_aligned_item_lens(self):
        with self.assertRaises(ValueError):
            dcp_pack_buffer_bytes([100], page_size=64, max_tokens=8)


class TestTryDcpPack(CustomTestCase):
    def test_try_pack_returns_dense_src_indices(self):
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
        with (
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.torch.cuda.default_stream"
            ),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.torch.cuda.stream",
                return_value=nullcontext(),
            ),
            patch("sglang.srt.disaggregation.common.dcp_pack.copy_mla_rows_into_pack"),
        ):
            packed = try_pack_dcp_src(
                pack_buffer=buf,
                kv_data_ptrs=[kv.data_ptr()],
                src_token_indices=src,
                token_item_lens=[item_len],
            )
        gather_stream.synchronize.assert_called_once_with()
        self.assertIsNotNone(packed)
        ptrs, indices = packed
        self.assertEqual(ptrs, [0x1000])
        np.testing.assert_array_equal(indices, np.arange(4))


if __name__ == "__main__":
    unittest.main()
