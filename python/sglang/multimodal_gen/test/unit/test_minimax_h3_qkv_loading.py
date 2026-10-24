# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for dense grouped-QKV loading; no communication runtime."""

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _copy_grouped_qkv_tp_shard,
    _reorder_grouped_qkv_to_qkv,
)


@pytest.mark.parametrize("tp_size,ulysses_size", [(1, 2), (2, 2), (2, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_composed_rank_shards_reconstruct_grouped_qkv(tp_size, ulysses_size, dtype):
    heads, head_dim, hidden = 56, 2, 3
    dense = ((torch.arange(heads * 3 * head_dim * hidden) % 127 - 63) / 4).to(dtype)
    dense = dense.reshape(heads * 3 * head_dim, hidden)
    local_heads = heads // (tp_size * ulysses_size)
    shards = []
    x = torch.tensor([[0.25, -0.5, 0.75], [1.0, 0.5, -0.25]])
    projected = []
    for tp_rank in range(tp_size):
        for ulysses_rank in range(ulysses_size):
            rank = tp_rank * ulysses_size + ulysses_rank
            shard = torch.empty(3 * local_heads * head_dim, hidden, dtype=dtype)
            shard.output_dim = 0
            assert _copy_grouped_qkv_tp_shard(
                shard,
                dense,
                num_query_groups=heads,
                head_dim=head_dim,
                tp_rank=rank,
                tp_size=tp_size * ulysses_size,
            )
            # Independent native [head, Q/K/V, dim, hidden] ownership oracle.
            start = rank * local_heads
            expected = dense.view(heads, 3, head_dim, hidden)[
                start : start + local_heads
            ]
            expected = expected.permute(1, 0, 2, 3).reshape_as(shard)
            assert torch.equal(shard.view(torch.uint8), expected.view(torch.uint8))
            shards.append(shard.reshape(3, local_heads, head_dim, hidden))
            projected.append(torch.einsum("si,qhdi->sqhd", x, shards[-1].float()))

    full = _reorder_grouped_qkv_to_qkv(
        dense,
        num_query_groups=heads,
        heads_per_group=1,
        head_dim=head_dim,
    ).reshape(3, heads, head_dim, hidden)
    reconstructed = torch.cat(shards, dim=1)
    assert torch.equal(reconstructed.view(torch.uint8), full.view(torch.uint8))
    torch.testing.assert_close(
        torch.cat(projected, dim=2),
        torch.einsum("si,qhdi->sqhd", x, full.float()),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "boundary",
    [
        "output_dim",
        "sharded",
        "packed",
        "negative_rank",
        "rank_outside",
        "zero_size",
        "nondivisible_size",
        "source_shape",
        "target_shape",
        "source_stride",
        "target_stride",
        "unsupported_dtype",
        "dtype_mismatch",
    ],
)
def test_unsupported_direct_copy_leaves_source_and_destination_unchanged(boundary):
    source = torch.arange(144, dtype=torch.float32).reshape(48, 3).to(torch.bfloat16)
    target = torch.full((12, 3), -7, dtype=torch.bfloat16)
    rank, size = 0, 4
    if boundary == "negative_rank":
        rank = -1
    elif boundary == "rank_outside":
        rank = size
    elif boundary == "zero_size":
        size = 0
    elif boundary == "nondivisible_size":
        size = 3
    elif boundary == "source_shape":
        source = source[:-1]
    elif boundary == "target_shape":
        target = target[:-1]
    elif boundary == "source_stride":
        source = source.t().contiguous().t()
    elif boundary == "target_stride":
        target = target.t().contiguous().t()
    elif boundary == "unsupported_dtype":
        source, target = source.float(), target.float()
    elif boundary == "dtype_mismatch":
        target = target.to(torch.float8_e4m3fn)
    target.output_dim = 1 if boundary == "output_dim" else 0
    if boundary == "sharded":
        target.is_sharded_weight = True
    if boundary == "packed":
        target.packed_dim = 0
    source_before, target_before = source.clone(), target.clone()
    assert not _copy_grouped_qkv_tp_shard(
        target,
        source,
        num_query_groups=8,
        head_dim=2,
        tp_rank=rank,
        tp_size=size,
    )
    assert torch.equal(source.float(), source_before.float())
    assert torch.equal(target.float(), target_before.float())
