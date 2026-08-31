import unittest

import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_mla_rows_into_pack
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestPdDcpGather(CustomTestCase):
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
