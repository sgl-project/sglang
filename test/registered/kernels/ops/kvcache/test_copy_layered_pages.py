import unittest

import torch

from sglang.kernels.ops.kvcache.copy_pages import copy_layered_pages
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestCopyLayeredPages(CustomTestCase):
    def test_copies_disjoint_rows_with_request_major_strides(self):
        state = torch.randn(8, 3, 16, 128, 16, device="cuda").transpose(0, 1)
        expected = state.clone()
        expected[:, [1, 4, 7]] = state[:, [2, 2, 6]]
        slot_ids = torch.tensor([2, 0, 2, 6], dtype=torch.int32, device="cuda")
        dst_ids = torch.tensor([1, 0, 4, 7], dtype=torch.int32, device="cuda")
        copy_layered_pages(state, src=slot_ids, dst=dst_ids)
        torch.testing.assert_close(state, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
