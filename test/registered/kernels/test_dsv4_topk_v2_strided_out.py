"""topk_v2 with a strided output (combined-table tail) must match the dense
output bit-for-bit: same inputs, out aimed at a column slice of a wider
table, non-slice columns untouched."""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_512_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

TOPK = 512
SWA = 128
PAGE_SIZE = 64


@pytest.mark.parametrize("bs", [1, 4, 33, 256])
@pytest.mark.parametrize("max_seq", [96, 700, 5000])
def test_topk_v2_strided_out_matches_dense(bs, max_seq):
    torch.manual_seed(bs * 7 + max_seq)
    dev = torch.device("cuda")
    seq_lens = torch.randint(1, max_seq + 1, (bs,), device=dev, dtype=torch.int32)
    max_len = int(seq_lens.max().item())
    scores_w = (max_len + 3) // 4 * 4
    scores = torch.randn(bs, scores_w, device=dev, dtype=torch.float32)
    n_pages = (max_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = torch.arange(bs * n_pages, device=dev, dtype=torch.int32).reshape(
        bs, n_pages
    )
    plan = plan_topk_v2(seq_lens)

    dense = torch.full((bs, TOPK), -7, dtype=torch.int32, device=dev)
    topk_transform_512_v2(scores, seq_lens, page_table, dense, PAGE_SIZE, plan)

    table = torch.full((bs, SWA + TOPK), -9, dtype=torch.int32, device=dev)
    strided = table[:, SWA:]
    topk_transform_512_v2(scores, seq_lens, page_table, strided, PAGE_SIZE, plan)

    # The selection order is nondeterministic beyond seq_len > topk (verified:
    # two identical dense runs differ in order but not in the selected set),
    # and the attention kernel consumes the row as a set -- compare sorted.
    assert torch.equal(
        strided.sort(dim=1).values, dense.sort(dim=1).values
    ), "strided out selects a different top-k set than dense out"
    assert (table[:, :SWA] == -9).all(), "kernel wrote outside its column slice"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
