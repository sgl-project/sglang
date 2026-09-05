"""MiniMax-H3 TP2 x U2 grouped-QKV ownership gates."""

import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _copy_grouped_qkv_tp_ulysses_shard,
    _reorder_grouped_qkv_to_qkv,
    _reorder_grouped_qkv_to_ulysses_qkv,
)

NUM_HEADS = 56
TP_SIZE = 2
ULYSSES_SIZE = 2
HEAD_DIM = 2
HIDDEN = 3


def _dense_weight() -> torch.Tensor:
    return torch.arange(
        NUM_HEADS * 3 * HEAD_DIM * HIDDEN,
        dtype=torch.bfloat16,
    ).reshape(NUM_HEADS * 3 * HEAD_DIM, HIDDEN)


def _rank_shard(
    dense: torch.Tensor, *, tp_rank: int, ulysses_rank: int
) -> torch.Tensor:
    local_heads = NUM_HEADS // (TP_SIZE * ULYSSES_SIZE)
    shard = torch.empty(
        3 * local_heads * HEAD_DIM,
        HIDDEN,
        dtype=torch.bfloat16,
    )
    shard.output_dim = 0
    assert _copy_grouped_qkv_tp_ulysses_shard(
        shard,
        dense,
        num_query_groups=NUM_HEADS,
        head_dim=HEAD_DIM,
        tp_rank=tp_rank,
        tp_size=TP_SIZE,
        ulysses_rank=ulysses_rank,
        ulysses_size=ULYSSES_SIZE,
    )
    return shard.reshape(3, local_heads, HEAD_DIM, HIDDEN)


def test_tp2_u2_shards_reconstruct_dense_qkv() -> None:
    dense = _dense_weight()
    baseline = _reorder_grouped_qkv_to_qkv(
        dense,
        num_query_groups=NUM_HEADS,
        heads_per_group=1,
        head_dim=HEAD_DIM,
    ).reshape(3, NUM_HEADS, HEAD_DIM, HIDDEN)
    reconstructed = torch.cat(
        [
            torch.cat(
                [
                    _rank_shard(dense, tp_rank=tp_rank, ulysses_rank=ulysses_rank)
                    for ulysses_rank in range(ULYSSES_SIZE)
                ],
                dim=1,
            )
            for tp_rank in range(TP_SIZE)
        ],
        dim=1,
    )
    torch.testing.assert_close(reconstructed, baseline, rtol=0, atol=0)

    for ulysses_rank in range(ULYSSES_SIZE):
        reordered = _reorder_grouped_qkv_to_ulysses_qkv(
            dense,
            num_query_groups=NUM_HEADS,
            head_dim=HEAD_DIM,
            tp_size=TP_SIZE,
            ulysses_size=ULYSSES_SIZE,
            ulysses_rank=ulysses_rank,
        ).reshape(3, NUM_HEADS // ULYSSES_SIZE, HEAD_DIM, HIDDEN)
        expected = torch.cat(
            [
                _rank_shard(dense, tp_rank=tp_rank, ulysses_rank=ulysses_rank)
                for tp_rank in range(TP_SIZE)
            ],
            dim=1,
        )
        torch.testing.assert_close(reordered, expected, rtol=0, atol=0)


def test_tp2_u2_gather_project_matches_tp_local_projection() -> None:
    dense = _dense_weight()
    gathered_x = torch.tensor(
        [
            [0.25, -0.50, 0.75],
            [1.00, 0.50, -0.25],
            [-1.00, 0.25, 0.50],
            [0.75, -0.75, 0.25],
        ],
        dtype=torch.float32,
    )
    for tp_rank in range(TP_SIZE):
        u_weights = [
            _rank_shard(dense, tp_rank=tp_rank, ulysses_rank=ulysses_rank)
            for ulysses_rank in range(ULYSSES_SIZE)
        ]
        tp_weight = torch.cat(u_weights, dim=1)
        baseline = torch.einsum("si,qhdi->sqhd", gathered_x, tp_weight.float())
        candidate = torch.cat(
            [
                torch.einsum("si,qhdi->sqhd", gathered_x, weight.float())
                for weight in u_weights
            ],
            dim=2,
        )
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
