import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsa.index_buf_accessor import GetKAndS
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-c", runner_config="4-gpu-h100")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestGetKAndSLargePage(CustomTestCase):
    def test_int32_page_index_preserves_large_byte_offset(self):
        page_size = 64
        index_head_dim = 128
        page_bytes = page_size * (index_head_dim + 4)
        page_index = 262_144
        self.assertGreater(page_index * page_bytes, 2**31)

        buf = torch.empty(
            (page_index + 1, page_bytes), dtype=torch.uint8, device="cuda"
        )
        expected_k = torch.arange(index_head_dim, dtype=torch.uint8, device="cuda")
        expected_s = torch.tensor([11, 22, 33, 44], dtype=torch.uint8, device="cuda")
        buf[page_index, :index_head_dim].copy_(expected_k)
        buf[
            page_index, page_size * index_head_dim : page_size * index_head_dim + 4
        ].copy_(expected_s)

        k, scale = GetKAndS.triton(
            SimpleNamespace(page_size=page_size, index_head_dim=index_head_dim),
            buf,
            page_indices=torch.tensor([[page_index]], dtype=torch.int32, device="cuda"),
            seq_len_tensor=torch.tensor([1], dtype=torch.int32, device="cuda"),
            seq_len_sum=1,
            max_seq_len=1,
        )
        torch.testing.assert_close(k[0], expected_k, atol=0, rtol=0)
        torch.testing.assert_close(scale[0], expected_s, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
