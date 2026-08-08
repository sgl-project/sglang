import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    prepare_trtllm_nope_sparse_metadata,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_prepare_trtllm_nope_sparse_metadata() -> None:
    topk = 2048
    page_table = torch.full((3, topk), -1, dtype=torch.int32, device="cuda")
    page_table[0] = torch.arange(topk, dtype=torch.int32, device="cuda")
    page_table[1, :3] = torch.tensor([11, 17, 23], dtype=torch.int32, device="cuda")
    expected_page_table = page_table.clone()
    expected_page_table[2, 0] = 0

    topk_lens = prepare_trtllm_nope_sparse_metadata(page_table)

    torch.testing.assert_close(
        topk_lens,
        torch.tensor([topk, 3, 1], dtype=torch.int32, device="cuda"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(page_table, expected_page_table, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
