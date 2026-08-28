import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import (
    dcp_topk_candidates,
    dcp_topk_merge,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_hip() or not is_gfx95_supported(),
    reason="Packed DSV4 DCP top-k requires ROCm gfx950",
)

C4_PAGE_SIZE = 64


def _local_lens(global_lens: torch.Tensor, dcp_size: int, rank: int) -> torch.Tensor:
    return (global_lens // dcp_size + (rank < global_lens % dcp_size)).to(torch.int32)


def test_candidate_mapping_uses_compressed_row_owner_order() -> None:
    candidates = torch.empty((1, 2), dtype=torch.int64, device="cuda")
    dcp_topk_candidates(
        torch.tensor([[1.0, 2.0]], device="cuda"),
        torch.tensor([2], dtype=torch.int32, device="cuda"),
        candidates,
        dcp_size=2,
        dcp_rank=1,
    )
    global_indices = candidates.view(torch.int32).reshape(1, 2, 2)[..., 0]
    torch.testing.assert_close(
        global_indices,
        torch.tensor([[1, 3]], dtype=torch.int32, device="cuda"),
    )


@pytest.mark.parametrize("topk", [512, 1024])
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("tied", [False, True])
def test_packed_dcp_topk_matches_global_score_set(
    topk: int, dcp_size: int, tied: bool
) -> None:
    device = torch.device("cuda")
    global_lens = torch.tensor(
        [1, topk - 1, topk, topk + 1, 4 * topk + 3, 8 * topk + 1],
        dtype=torch.int32,
        device=device,
    )
    batch_size = global_lens.numel()
    global_width = int(global_lens.max())
    generator = torch.Generator(device=device).manual_seed(20260828 + topk + dcp_size)
    if tied:
        global_scores = torch.zeros(
            (batch_size, global_width), dtype=torch.float32, device=device
        )
    else:
        global_scores = torch.randn(
            (batch_size, global_width),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

    rank_candidates = []
    rank_page_tables = []
    for rank in range(dcp_size):
        local_scores = global_scores[:, rank::dcp_size].contiguous()
        local_lens = _local_lens(global_lens, dcp_size, rank)
        candidates = torch.empty((batch_size, topk), dtype=torch.int64, device=device)
        dcp_topk_candidates(
            local_scores,
            local_lens,
            candidates,
            dcp_size,
            rank,
        )
        rank_candidates.append(candidates)

        num_pages = (local_scores.shape[1] + C4_PAGE_SIZE - 1) // C4_PAGE_SIZE
        page_table = (
            torch.arange(num_pages, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
        )
        page_table += (
            rank * 10000
            + torch.arange(batch_size, dtype=torch.int32, device=device).unsqueeze(1)
            * 100
        )
        rank_page_tables.append(page_table)

    gathered = torch.cat(rank_candidates, dim=0)
    per_rank_raw = []
    for rank in range(dcp_size):
        page_indices = torch.empty((batch_size, topk), dtype=torch.int32, device=device)
        local_raw = torch.empty_like(page_indices)
        local_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
        dcp_topk_merge(
            gathered,
            rank_page_tables[rank],
            page_indices,
            local_lens,
            C4_PAGE_SIZE,
            dcp_size,
            rank,
            local_raw,
        )
        torch.cuda.synchronize()

        for batch_idx in range(batch_size):
            count = int(local_lens[batch_idx])
            raw = local_raw[batch_idx, :count].to(torch.int64)
            expected_pages = (
                rank_page_tables[rank][batch_idx, raw // C4_PAGE_SIZE] * C4_PAGE_SIZE
                + raw % C4_PAGE_SIZE
            ).to(torch.int32)
            torch.testing.assert_close(page_indices[batch_idx, :count], expected_pages)
            assert torch.all(page_indices[batch_idx, count:] == -1)
            assert torch.all(local_raw[batch_idx, count:] == -1)
        per_rank_raw.append(local_raw)

    for batch_idx, length_tensor in enumerate(global_lens):
        length = int(length_tensor)
        selected = []
        for rank in range(dcp_size):
            count = int((per_rank_raw[rank][batch_idx] >= 0).sum().item())
            local_raw = per_rank_raw[rank][batch_idx, :count].to(torch.int64)
            selected.extend((local_raw * dcp_size + rank).tolist())

        expected_count = min(length, topk)
        assert len(selected) == expected_count
        assert len(set(selected)) == expected_count
        assert all(0 <= index < length for index in selected)
        if tied:
            assert set(selected) == set(range(expected_count))
        elif length <= topk:
            assert set(selected) == set(range(length))
        else:
            actual_scores = torch.sort(global_scores[batch_idx, selected]).values
            expected_scores = torch.sort(
                torch.topk(global_scores[batch_idx, :length], topk).values
            ).values
            torch.testing.assert_close(actual_scores, expected_scores)
