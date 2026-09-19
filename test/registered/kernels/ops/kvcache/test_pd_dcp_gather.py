import unittest

import numpy as np
import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import (
    copy_dsa_pages_into_pack,
    copy_mla_rows_into_pack,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestPdDcpGather(CustomTestCase):
    def test_dsa_page_layout_and_tail_canaries(self):
        page_bytes = 8448
        rng = np.random.default_rng(81)
        host = rng.integers(0, 256, size=40 * page_bytes, dtype=np.uint8)
        source = torch.tensor(host, device="cuda")
        for size in (2, 3, 4, 8):
            for page_count in (1, size, 2 * size + 1):
                pages = rng.permutation(40)[:page_count]
                metadata = torch.tensor(
                    [source.data_ptr(), *pages.tolist()],
                    dtype=torch.int64,
                    device="cuda",
                )
                for rank in range(size):
                    with self.subTest(size=size, pages=page_count, rank=rank):
                        n = (page_count * 64 + size - 1 - rank) // size
                        packed_bytes = ((n + 63) // 64) * page_bytes
                        # Check nonzero output offset and guard bytes on both sides.
                        storage = torch.full(
                            (packed_bytes + 128,), 173, dtype=torch.uint8, device="cuda"
                        )
                        output = storage.narrow(0, 64, packed_bytes)
                        before = torch.cuda.memory_allocated()
                        copy_dsa_pages_into_pack(metadata, output, n, size, rank)
                        torch.cuda.synchronize()
                        self.assertEqual(torch.cuda.memory_allocated(), before)
                        expected = np.zeros(packed_bytes, dtype=np.uint8)
                        for local, token in enumerate(
                            range(rank, page_count * 64, size)
                        ):
                            sp, slot = divmod(token, 64)
                            dp, ds = divmod(local, 64)
                            for width, base in ((128, 0), (4, 8192)):
                                src = int(pages[sp]) * page_bytes + base + slot * width
                                dst = dp * page_bytes + base + ds * width
                                expected[dst : dst + width] = host[src : src + width]
                        result = storage.cpu().numpy()
                        np.testing.assert_array_equal(result[64:-64], expected)
                        self.assertTrue(np.all(result[:64] == 173))
                        self.assertTrue(np.all(result[-64:] == 173))

    def test_gathers_strided_rows_layer_major(self):
        dim = 8
        kv0 = torch.arange(32 * dim, dtype=torch.float32, device="cuda").view(
            32, 1, dim
        )
        kv1 = torch.arange(32 * 5, dtype=torch.float16, device="cuda").view(32, 1, 5)
        row_indices = torch.tensor([0, 4, 9, 12], dtype=torch.int64, device="cuda")
        item_lens = [int(kv0[0].nbytes), int(kv1[0].nbytes)]
        pack = torch.zeros(
            row_indices.numel() * sum(item_lens), dtype=torch.uint8, device="cuda"
        )

        copy_mla_rows_into_pack(
            [kv0.data_ptr(), kv1.data_ptr()],
            row_indices,
            pack,
            item_lens,
        )
        torch.cuda.synchronize()

        split = row_indices.numel() * item_lens[0]
        packed0 = pack[:split].view(torch.float32).view(4, 1, dim)
        packed1 = pack[split:].view(torch.float16).view(4, 1, 5)
        torch.testing.assert_close(packed0, kv0[row_indices], rtol=0, atol=0)
        torch.testing.assert_close(packed1, kv1[row_indices], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
