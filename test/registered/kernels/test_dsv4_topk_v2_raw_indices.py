import sys

import pytest
import torch

from sglang.jit_kernel.dsv4 import (
    plan_topk_v2,
    topk_transform_512,
    topk_transform_512_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, suite="nightly-1-gpu", nightly=True)


@pytest.mark.parametrize("topk", [512, 1024])
@pytest.mark.parametrize("seq_lens", [[17, 511, 512], [513, 4096, 33000]])
def test_topk_v2_optional_raw_indices_matches_v1(topk, seq_lens):
    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA")

    torch.manual_seed(topk + sum(seq_lens))
    page_size = 64
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    num_pages = (max_seq_len + page_size - 1) // page_size

    scores = torch.randn(batch_size, num_pages * page_size, device="cuda")
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    page_table = torch.empty(batch_size, num_pages, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        page_table[i] = torch.randperm(num_pages, device="cuda", dtype=torch.int32)

    out_page_v1 = torch.empty(batch_size, topk, dtype=torch.int32, device="cuda")
    out_raw_v1 = torch.empty_like(out_page_v1)
    topk_transform_512(
        scores,
        seq_lens_tensor,
        page_table,
        out_page_v1,
        page_size,
        out_raw_v1,
    )

    out_page_v2 = torch.empty_like(out_page_v1)
    out_raw_v2 = torch.empty_like(out_page_v1)
    topk_transform_512_v2(
        scores,
        seq_lens_tensor,
        page_table,
        out_page_v2,
        page_size,
        plan_topk_v2(seq_lens_tensor),
        out_raw_v2,
    )

    assert torch.equal(
        torch.sort(out_raw_v1, dim=1).values,
        torch.sort(out_raw_v2, dim=1).values,
    )
    assert torch.equal(
        torch.sort(out_page_v1, dim=1).values,
        torch.sort(out_page_v2, dim=1).values,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
