import unittest

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    transform_raw_c4_indices_to_page_indices,
    transform_raw_c4_indices_to_page_indices_torch,
    transform_raw_c4_indices_to_page_indices_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSV4CSAIndexCacheTransform(CustomTestCase):
    def _assert_triton_matches_torch(
        self,
        raw_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        page_size: int,
    ):
        device = torch.device("cuda")
        raw_indices = raw_indices.to(device=device, dtype=torch.int32)
        page_table = page_table.to(device=device, dtype=torch.int32)
        seq_lens = seq_lens.to(device=device, dtype=torch.int32)
        expected = torch.empty_like(raw_indices)
        actual = torch.empty_like(raw_indices)

        transform_raw_c4_indices_to_page_indices_torch(
            raw_indices, seq_lens, page_table, expected, page_size
        )
        transform_raw_c4_indices_to_page_indices_triton(
            raw_indices, seq_lens, page_table, actual, page_size
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_c4_index_transform_triton_matches_torch_basic_cases(self):
        raw_indices = torch.tensor(
            [
                [0, 63, 64, 127, 128, -1, 512],
                [7, 65, 129, 256, 300, 999, -4],
                [1, 2, 3, 4, 5, 6, 7],
            ]
        )
        page_table = torch.tensor(
            [
                [100, 101, 102, -1, 104, 105, 106, 107],
                [200, -1, 202, 203, 204, 205, 206, 207],
                [300, 301, 302, 303, 304, 305, 306, 307],
            ]
        )
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([256, 320, 4]), page_table, page_size=64
        )
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([[256], [320], [4]]), page_table, page_size=64
        )

    def test_c4_index_transform_triton_matches_torch_different_page_size(self):
        raw_indices = torch.tensor(
            [
                [0, 127, 128, 255, 256, -1, 1024],
                [3, 129, 257, 384, 500, 999, -8],
            ]
        )
        page_table = torch.tensor(
            [
                [10, 11, 12, -1, 14, 15, 16, 17],
                [20, -1, 22, 23, 24, 25, 26, 27],
            ]
        )
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([512, 640]), page_table, page_size=128
        )

    def test_c4_index_transform_triton_matches_torch_large_odd_topk(self):
        batch_size = 3
        topk = 513
        page_size = 64
        raw_indices = (
            torch.arange(batch_size * topk, dtype=torch.int32).reshape(batch_size, topk)
            % 700
        )
        raw_indices[0, 17] = -1
        raw_indices[1, 257] = 4096
        raw_indices[2, 512] = 63
        page_table = (
            torch.arange(batch_size * 16, dtype=torch.int32).reshape(batch_size, 16)
            + 100
        )
        page_table[1, 3] = -1
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([640, 768, 256]), page_table, page_size
        )

    def test_c4_index_transform_triton_matches_torch_non_contiguous_inputs(self):
        raw_base = torch.tensor(
            [
                [0, 999, 64, 999, 128, 999, 192, 999],
                [3, 999, 67, 999, 131, 999, 195, 999],
            ]
        )
        page_table_base = torch.tensor(
            [
                [10, 999, 11, 999, 12, 999, 13, 999],
                [20, 999, 21, 999, -1, 999, 23, 999],
            ]
        )
        raw_indices = raw_base[:, ::2]
        page_table = page_table_base[:, ::2]
        self.assertFalse(raw_indices.is_contiguous())
        self.assertFalse(page_table.is_contiguous())
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([256, 256]), page_table, page_size=64
        )

    def test_c4_index_transform_triton_handles_empty_topk(self):
        raw_indices = torch.empty((2, 0), dtype=torch.int32)
        page_table = torch.tensor([[1, 2], [3, 4]])
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([64, 64]), page_table, page_size=64
        )

    def test_c4_index_transform_wrapper_falls_back_to_torch_on_cpu(self):
        raw_indices = torch.tensor([[0, 63, 64, -1], [5, 128, 256, 1024]])
        seq_lens = torch.tensor([128, 300])
        page_table = torch.tensor([[10, 11, 12, 13], [20, 21, -1, 23]])
        expected = torch.empty_like(raw_indices)
        actual = torch.empty_like(raw_indices)

        transform_raw_c4_indices_to_page_indices_torch(
            raw_indices, seq_lens, page_table, expected, page_size=64
        )
        transform_raw_c4_indices_to_page_indices(
            raw_indices, seq_lens, page_table, actual, page_size=64
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
