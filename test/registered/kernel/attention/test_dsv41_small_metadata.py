"""Integer metadata equivalence with changing CUDA graph inputs."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    BuildPageTablePositions,
)
from sglang.kernels.ops.attention.dsv41_small_metadata import (
    page_table_positions_small,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("rows", [1, 5, 6, 8])
@pytest.mark.parametrize("pages", [1, 17, 4098])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_pages_graph(rows, pages, dtype):
    # Rows have a larger physical stride than the logical table.
    mapping = torch.randint(-256, 1 << 24, (9, pages * 256 + 512), device="cuda")
    reqs = torch.arange(rows, device="cuda", dtype=dtype)
    lens = torch.arange(rows, device="cuda", dtype=dtype)
    args = dict(
        req_to_token=mapping,
        req_pool_indices_repeated=reqs,
        seq_lens_casual=lens,
        max_seq_len=pages * 256,
        page_size=256,
        swa_window=128,
    )
    for _ in range(3):
        page_table_positions_small(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = page_table_positions_small(**args)
    for replay in range(5):
        mapping.random_(-256, 1 << 24)
        reqs.copy_((torch.arange(rows, device="cuda") + replay) % 9)
        lens.copy_(torch.arange(rows, device="cuda") * 127 + replay - 1)
        graph.replay()
        ref = BuildPageTablePositions.triton(**args)
        for name in (
            "seq_lens_casual",
            "positions_casual",
            "page_table",
            "swa_topk_lengths",
        ):
            assert torch.equal(getattr(out, name), getattr(ref, name)), name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
