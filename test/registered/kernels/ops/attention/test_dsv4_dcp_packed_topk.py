import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import (
    dcp_topk_candidates,
    dcp_topk_merge,
)
from sglang.srt.layers.attention.dsv4.dcp import (
    combined_q_topk_candidate_view,
    combined_q_topk_rank_major_q_view,
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


def test_candidate_clamps_metadata_length_to_score_width() -> None:
    candidates = torch.empty((1, 4), dtype=torch.int64, device="cuda")
    dcp_topk_candidates(
        torch.tensor([[3.0, 1.0]], device="cuda"),
        torch.tensor([99], dtype=torch.int32, device="cuda"),
        candidates,
        dcp_size=4,
        dcp_rank=2,
    )
    global_indices = candidates.view(torch.int32).reshape(1, 4, 2)[..., 0]
    torch.testing.assert_close(
        global_indices,
        torch.tensor([[2, 6, -1, -1]], dtype=torch.int32, device="cuda"),
    )


def test_candidate_clamps_negative_metadata_length_to_zero() -> None:
    candidates = torch.empty((1, 4), dtype=torch.int64, device="cuda")
    dcp_topk_candidates(
        torch.tensor([[3.0, 1.0]], device="cuda"),
        torch.tensor([-1], dtype=torch.int32, device="cuda"),
        candidates,
        dcp_size=4,
        dcp_rank=2,
    )
    global_indices = candidates.view(torch.int32).reshape(1, 4, 2)[..., 0]
    torch.testing.assert_close(
        global_indices,
        torch.full((1, 4), -1, dtype=torch.int32, device="cuda"),
    )


@pytest.mark.parametrize("tied", [False, True])
def test_candidate_large_width_preserves_guard_rows(tied: bool) -> None:
    batch_size, score_width, topk = 64, 32768, 1024
    sentinel = 0x123456789ABCDEF
    storage = torch.full(
        (batch_size + 2, topk),
        sentinel,
        dtype=torch.int64,
        device="cuda",
    )
    candidates = storage[1:-1]
    scores = (
        torch.zeros((batch_size, score_width), dtype=torch.float32, device="cuda")
        if tied
        else torch.randn((batch_size, score_width), dtype=torch.float32, device="cuda")
    )
    dcp_topk_candidates(
        scores,
        torch.full((batch_size,), score_width, dtype=torch.int32, device="cuda"),
        candidates,
        dcp_size=4,
        dcp_rank=3,
    )
    torch.cuda.synchronize()

    assert torch.all(storage[0] == sentinel)
    assert torch.all(storage[-1] == sentinel)
    global_indices = candidates.view(torch.int32).reshape(batch_size, topk, 2)[..., 0]
    assert torch.all(global_indices >= 0)
    assert torch.all(global_indices < score_width * 4)
    assert torch.all(torch.remainder(global_indices, 4) == 3)
    repeated = torch.empty_like(candidates)
    dcp_topk_candidates(
        scores,
        torch.full((batch_size,), score_width, dtype=torch.int32, device="cuda"),
        repeated,
        dcp_size=4,
        dcp_rank=3,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(repeated, candidates, rtol=0, atol=0)


def test_candidate_large_concentrated_scores_matches_topk() -> None:
    score_width, topk, dcp_size, dcp_rank = 32768, 1024, 4, 0
    scores = (
        1.0
        + torch.arange(score_width, dtype=torch.float32, device="cuda").unsqueeze(0)
        * 1.0e-7
    )
    candidates = torch.empty((1, topk), dtype=torch.int64, device="cuda")
    dcp_topk_candidates(
        scores,
        torch.tensor([score_width], dtype=torch.int32, device="cuda"),
        candidates,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    torch.cuda.synchronize()

    global_indices = candidates.view(torch.int32).reshape(1, topk, 2)[0, :, 0]
    expected = (
        torch.argsort(scores[0], descending=True, stable=True)[:topk].to(torch.int32)
        * dcp_size
        + dcp_rank
    )
    torch.testing.assert_close(
        torch.sort(global_indices).values,
        torch.sort(expected).values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("score_width", [8193, 16384, 32768, 65536])
def test_candidate_long_random_scores_match_exact_topk(score_width: int) -> None:
    topk, dcp_size, dcp_rank = 1024, 4, 3
    generator = torch.Generator(device="cuda").manual_seed(20260828 + score_width)
    scores = torch.randn(
        (1, score_width),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    candidates = torch.empty((1, topk), dtype=torch.int64, device="cuda")
    dcp_topk_candidates(
        scores,
        torch.tensor([score_width], dtype=torch.int32, device="cuda"),
        candidates,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    torch.cuda.synchronize()

    global_indices = candidates.view(torch.int32).reshape(1, topk, 2)[0, :, 0]
    expected = (
        torch.argsort(scores[0], descending=True, stable=True)[:topk].to(torch.int32)
        * dcp_size
        + dcp_rank
    )
    torch.testing.assert_close(
        torch.sort(global_indices).values,
        torch.sort(expected).values,
        rtol=0,
        atol=0,
    )


def test_candidate_special_scores_match_canonical_reference() -> None:
    score_width, topk, dcp_size, dcp_rank = 16384, 1024, 4, 1
    scores = (
        1.0
        + torch.arange(score_width, dtype=torch.float32, device="cuda").unsqueeze(0)
        * 1.0e-7
    )
    scores[0, 0] = float("nan")
    scores[0, 1] = -0.0
    scores[0, 2] = 0.0
    scores[0, 3] = float("inf")
    scores[0, 4] = -float("inf")
    candidates = torch.empty((1, topk), dtype=torch.int64, device="cuda")
    dcp_topk_candidates(
        scores,
        torch.tensor([score_width], dtype=torch.int32, device="cuda"),
        candidates,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    torch.cuda.synchronize()

    canonical = torch.nan_to_num(scores[0], nan=-float("inf"))
    canonical = torch.where(canonical == 0.0, 0.0, canonical)
    expected = (
        torch.argsort(canonical, descending=True, stable=True)[:topk].to(torch.int32)
        * dcp_size
        + dcp_rank
    )
    global_indices = candidates.view(torch.int32).reshape(1, topk, 2)[0, :, 0]
    torch.testing.assert_close(
        torch.sort(global_indices).values,
        torch.sort(expected).values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_combined_q_candidate_layout_is_zero_copy_and_bitwise(
    batch_size: int,
) -> None:
    local_heads, head_dim, topk, dcp_size, dcp_rank = 16, 512, 1024, 8, 3
    candidate_heads = topk * 4 // head_dim
    combined_heads = local_heads + candidate_heads
    generator = torch.Generator(device="cuda").manual_seed(20260828 + batch_size)

    local_combined = torch.empty(
        (batch_size, combined_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q = torch.randn(
        (batch_size, local_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    q_out = local_combined[:, :local_heads, :]
    assert q_out.stride() == (combined_heads * head_dim, head_dim, 1)
    q_out.copy_(q)
    local_candidates = combined_q_topk_candidate_view(
        local_combined, local_heads=local_heads, topk=topk
    )
    assert local_candidates.stride() == (3072, 1)
    assert local_candidates.data_ptr() == (
        local_combined.data_ptr() + local_heads * head_dim * 2
    )

    score_width = 2 * topk + 17
    scores = torch.randn(
        (batch_size, score_width),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    local_lens = torch.linspace(
        topk - 1,
        score_width,
        batch_size,
        dtype=torch.int32,
        device="cuda",
    )
    contiguous_candidates = torch.empty(
        (batch_size, topk), dtype=torch.int64, device="cuda"
    )
    dcp_topk_candidates(scores, local_lens, contiguous_candidates, dcp_size, dcp_rank)
    dcp_topk_candidates(scores, local_lens, local_candidates, dcp_size, dcp_rank)
    torch.cuda.synchronize()

    # Candidate output order above the threshold is allowed to follow atomic
    # arrival order. Sorting compares the packed score+index bits themselves.
    torch.testing.assert_close(
        torch.sort(local_candidates, dim=1).values,
        torch.sort(contiguous_candidates, dim=1).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(q_out, q, rtol=0, atol=0)

    gathered = torch.empty(
        (dcp_size * batch_size, combined_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    for rank in range(dcp_size):
        gathered[rank * batch_size : (rank + 1) * batch_size, :local_heads, :].copy_(
            q + rank
        )
    rank_major_q = combined_q_topk_rank_major_q_view(
        gathered,
        batch_size=batch_size,
        local_heads=local_heads,
        topk=topk,
        dcp_size=dcp_size,
    )
    assert rank_major_q.shape == (batch_size, dcp_size, local_heads, head_dim)
    assert rank_major_q.stride() == (
        combined_heads * head_dim,
        batch_size * combined_heads * head_dim,
        head_dim,
        1,
    )
    for rank in range(dcp_size):
        torch.testing.assert_close(rank_major_q[:, rank], q + rank, rtol=0, atol=0)


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
    head_dim, local_heads = 512, 16
    gathered_combined = torch.empty(
        (
            dcp_size * batch_size,
            local_heads + topk * 4 // head_dim,
            head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    strided_gathered = combined_q_topk_candidate_view(
        gathered_combined,
        local_heads=local_heads,
        topk=topk,
    )
    strided_gathered.copy_(gathered)
    per_rank_raw = []
    for rank in range(dcp_size):
        page_indices = torch.empty((batch_size, topk), dtype=torch.int32, device=device)
        local_raw = torch.empty_like(page_indices)
        local_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
        dcp_topk_merge(
            strided_gathered,
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
        repeated_page_indices = torch.empty_like(page_indices)
        repeated_local_raw = torch.empty_like(local_raw)
        repeated_local_lens = torch.empty_like(local_lens)
        dcp_topk_merge(
            gathered,
            rank_page_tables[rank],
            repeated_page_indices,
            repeated_local_lens,
            C4_PAGE_SIZE,
            dcp_size,
            rank,
            repeated_local_raw,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(repeated_local_lens, local_lens, rtol=0, atol=0)
        torch.testing.assert_close(repeated_page_indices, page_indices, rtol=0, atol=0)
        torch.testing.assert_close(repeated_local_raw, local_raw, rtol=0, atol=0)
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
