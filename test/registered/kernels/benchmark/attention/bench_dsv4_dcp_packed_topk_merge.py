import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4 import (
    dcp_topk_candidates,
    dcp_topk_merge,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-benchmark", runner_config="amd")

TOPK = 1024
DCP_SIZE = 8
PAGE_SIZE = 64


def _merge(
    candidates: torch.Tensor,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    local_lens: torch.Tensor,
    local_raw: torch.Tensor,
) -> None:
    dcp_topk_merge(
        candidates,
        page_table,
        page_indices,
        local_lens,
        PAGE_SIZE,
        DCP_SIZE,
        0,
        local_raw,
    )


@marker.parametrize("total_valid", [256, 512, 1024, 1025, 2048])
@marker.benchmark("implementation", ["dcp_merge"])
def benchmark_partial(total_valid: int, implementation: str):
    assert implementation == "dcp_merge"
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260829 + total_valid)
    rank_candidates = []
    max_local_count = (total_valid + DCP_SIZE - 1) // DCP_SIZE
    for rank in range(DCP_SIZE):
        local_count = total_valid // DCP_SIZE + (rank < total_valid % DCP_SIZE)
        scores = torch.randn(
            (1, max_local_count),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        local_lens = torch.tensor([local_count], dtype=torch.int32, device=device)
        candidates = torch.empty((1, TOPK), dtype=torch.int64, device=device)
        dcp_topk_candidates(scores, local_lens, candidates, DCP_SIZE, rank)
        rank_candidates.append(candidates)

    candidates = torch.cat(rank_candidates, dim=0)
    page_table = torch.arange(
        (max_local_count + PAGE_SIZE - 1) // PAGE_SIZE,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    page_indices = torch.empty((1, TOPK), dtype=torch.int32, device=device)
    local_lens = torch.empty((1,), dtype=torch.int32, device=device)
    local_raw = torch.empty_like(page_indices)

    return marker.do_bench(
        _merge,
        input_args=(
            candidates,
            page_table,
            page_indices,
            local_lens,
            local_raw,
        ),
        graph_clone_args=(0, 1),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark_partial.run()
