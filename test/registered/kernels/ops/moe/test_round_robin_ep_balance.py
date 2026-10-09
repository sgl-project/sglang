"""GPU coverage for the round-robin benchmark routing override's EP-rank balance."""

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _simulate_balanced_routing,
    select_experts,
)
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 256
TOPK = 8
EP_SIZE = 16


def _reference(
    num_tokens: int,
    *,
    num_ranks: int,
    token_shard_rank: int = 0,
    num_token_shards: int = 1,
    layer_id: int = 0,
    topk: int = TOPK,
    num_experts: int = NUM_EXPERTS,
) -> torch.Tensor:
    """Plain-torch statement of the dealing: slot ``t*k + j`` of shard ``r``
    goes to rank ``d % P`` and expert ``(d // P) % (E // P)`` in that rank,
    with ``d`` offset by the layer, a rank rotation of ``r`` (spread evenly
    when there are fewer shards than ranks) and an expert rotation of ``r``."""
    experts_per_rank = num_experts // num_ranks
    rank_rotation = (
        token_shard_rank * max(num_ranks, num_token_shards) // num_token_shards
    )
    offset = layer_id + rank_rotation + token_shard_rank * num_ranks
    tokens = torch.arange(num_tokens, device="cuda").unsqueeze(1)
    slots = torch.arange(topk, device="cuda").unsqueeze(0)
    deal = tokens * topk + slots + offset
    return (deal % num_ranks) * experts_per_rank + (
        deal // num_ranks
    ) % experts_per_rank


def _route(
    num_tokens: int,
    *,
    num_ranks: int = EP_SIZE,
    token_shard_rank: int = 0,
    num_token_shards: int = 1,
    layer_id: int = 0,
    topk: int = TOPK,
    num_experts: int = NUM_EXPERTS,
    dtype: torch.dtype = torch.int32,
):
    ids = torch.full((num_tokens, topk), -1, dtype=dtype, device="cuda")
    weights = torch.zeros((num_tokens, topk), dtype=torch.float32, device="cuda")
    _simulate_balanced_routing(
        ids,
        weights,
        num_experts,
        random=False,
        num_ranks=num_ranks,
        layer_id=layer_id,
        token_shard_rank=token_shard_rank,
        num_token_shards=num_token_shards,
    )
    return ids, weights


def _rank_loads(ids: torch.Tensor, num_ranks: int = EP_SIZE) -> torch.Tensor:
    return torch.bincount(
        ids.flatten().long() // (NUM_EXPERTS // num_ranks), minlength=num_ranks
    )


def _sharded_ids(counts: list[int], *, layer_id: int = 0) -> torch.Tensor:
    return torch.cat(
        [
            _route(
                count,
                token_shard_rank=rank,
                num_token_shards=len(counts),
                layer_id=layer_id,
            )[0]
            for rank, count in enumerate(counts)
        ]
    )


@pytest.mark.parametrize(
    "num_ranks, num_shards, tokens, topk, layer_id, dtype",
    [
        (16, 1, 300, 8, 0, torch.int32),  # one shard
        (1, 1, 17, 8, 3, torch.int32),  # single rank: sequential experts
        (16, 16, 3, 8, 5, torch.int32),  # as many shards as ranks
        (16, 4, 2, 8, 0, torch.int64),  # fewer shards than ranks, int64 ids
        (16, 24, 1, 8, 0, torch.int32),  # more shards than ranks
        (16, 2, 5, 6, 3, torch.int32),  # non-power-of-two top-k
    ],
)
def test_kernel_matches_reference_on_every_shard(
    num_ranks, num_shards, tokens, topk, layer_id, dtype
):
    for shard in range(num_shards):
        ids, weights = _route(
            tokens,
            num_ranks=num_ranks,
            token_shard_rank=shard,
            num_token_shards=num_shards,
            layer_id=layer_id,
            topk=topk,
            dtype=dtype,
        )
        expected = _reference(
            tokens,
            num_ranks=num_ranks,
            token_shard_rank=shard,
            num_token_shards=num_shards,
            layer_id=layer_id,
            topk=topk,
        )
        assert torch.equal(ids.long(), expected), (shard, ids, expected)
        assert torch.equal(weights, torch.full_like(weights, 1.0 / topk))


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 300])
@pytest.mark.parametrize("num_ranks", [4, 8, 16])
def test_single_shard_balances_within_one_slot(num_tokens, num_ranks):
    ids, _ = _route(num_tokens, num_ranks=num_ranks)
    loads = _rank_loads(ids, num_ranks)
    assert int(loads.max() - loads.min()) <= 1
    assert all(row.unique().numel() == TOPK for row in ids)


@pytest.mark.parametrize("tokens_per_shard", [1, 2, 64])
@pytest.mark.parametrize("layer_id", [0, 5])
def test_equal_shards_balance_exactly(tokens_per_shard, layer_id):
    # Every CUDA-graph decode step with equal per-rank batch sizes: the
    # rotated per-shard streams tile the EP ranks with no remainder.
    loads = _rank_loads(_sharded_ids([tokens_per_shard] * EP_SIZE, layer_id=layer_id))
    assert torch.equal(loads, torch.full_like(loads, tokens_per_shard * TOPK))


@pytest.mark.parametrize(
    "counts",
    [
        [2] + [1] * 15,
        [2, 1] * 8,
        [65] * 8 + [64] * 8,
        [1, 3, 5, 7] + [0] * 12,
        # Synthetic uneven prefill: 1001, 2002, 3003, 4004, 1005, 2006, 3007, 4008.
        [1000 * (r % 4 + 1) + r + 1 for r in range(8)],
    ],
)
def test_uneven_shards_spread_is_at_most_one_slot_per_partial_shard(counts):
    # Without cross-shard token counts, a shard whose ``tokens * k`` is not a
    # multiple of the rank count leaves one partial block, worth at most one
    # slot of imbalance; shards with whole blocks contribute none.
    loads = _rank_loads(_sharded_ids(counts))
    partial_shards = sum(1 for c in counts if (c * TOPK) % EP_SIZE)
    assert int(loads.max() - loads.min()) <= partial_shards
    assert int(loads.sum()) == sum(counts) * TOPK


def test_small_batches_spread_over_experts_within_each_rank():
    # Sixteen 1-token shards: the expert rotation keeps them from all landing on
    # the first expert of every rank (which would make weight-bandwidth-bound
    # decode measurements optimistic).
    ids = _sharded_ids([1] * EP_SIZE)
    experts_per_rank = NUM_EXPERTS // EP_SIZE
    for rank in range(EP_SIZE):
        on_rank = ids[ids // experts_per_rank == rank]  # 8 slots from 8 shards
        assert on_rank.unique().numel() >= on_rank.numel() * 3 // 4, (rank, on_rank)


def _fixed_router(hidden_states, gating_output, topk, renormalize):
    del hidden_states, renormalize
    ids = torch.full(
        (gating_output.shape[0], topk), 77, dtype=torch.int32, device="cuda"
    )
    weights = torch.full(
        (gating_output.shape[0], topk), 0.75, dtype=torch.float32, device="cuda"
    )
    return weights, ids


def _select(
    top_k: int = TOPK,
    num_fused_shared_experts: int = 0,
    *,
    num_tokens: int = 2,
    dp_rank: int = 0,
    dp_size: int = 1,
    tp_rank: int = 0,
    tp_size: int = 1,
    ep_size: int = EP_SIZE,
    scattered: bool = True,
    uniform: bool = False,
    **forward_flags,
):
    with (
        envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(uniform),
        envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(not uniform),
        get_parallel().override(
            attn_dp_rank=dp_rank,
            attn_dp_size=dp_size,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp_size,
            moe_ep_size=ep_size,
        ),
        get_forward().scoped(**forward_flags),
        patch(
            "sglang.srt.layers.moe.topk.is_moe_input_scattered_across_dp_ranks",
            return_value=scattered,
        ),
    ):
        return select_experts(
            hidden_states=torch.zeros((num_tokens, 4), device="cuda"),
            router_logits=torch.zeros((num_tokens, NUM_EXPERTS), device="cuda"),
            topk_config=TopKConfig(
                top_k=top_k,
                num_fused_shared_experts=num_fused_shared_experts,
                custom_routing_function=_fixed_router,
            ),
            layer_id=0,
        )


def test_select_experts_gathered_single_rank_deals_experts_in_order():
    output = _select(num_tokens=4, ep_size=1, scattered=False)
    expected = _reference(4, num_ranks=1)
    assert torch.equal(output.topk_ids.long(), expected)


def test_select_experts_scattered_dp_shards_tile_the_ranks():
    ids = torch.cat(
        [
            _select(num_tokens=1, dp_rank=r, dp_size=EP_SIZE).topk_ids
            for r in range(EP_SIZE)
        ]
    )
    loads = _rank_loads(ids)
    assert torch.equal(loads, torch.full_like(loads, TOPK))


@pytest.mark.parametrize("flag", ["attn_tp_sequence_sharded", "attn_input_scattered"])
def test_select_experts_sharded_attention_tp_ranks_are_distinct_shards(flag):
    # DP1 x attention-TP2 with attention-TP-local rows: each attention-TP rank
    # deals its own shard, and the two 1-token slices cover the ranks exactly.
    outputs = [
        _select(num_tokens=1, tp_rank=tp_rank, tp_size=2, **{flag: True}).topk_ids
        for tp_rank in range(2)
    ]
    assert not torch.equal(outputs[0], outputs[1])
    loads = _rank_loads(torch.cat(outputs))
    assert torch.equal(loads, torch.ones_like(loads))


def test_select_experts_replicated_attention_tp_ranks_agree():
    # Without attention-TP-local rows the attention-TP ranks hold the same
    # tokens and must route them identically.
    outputs = [_select(tp_rank=tp_rank, tp_size=2).topk_ids for tp_rank in range(2)]
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("uniform", [False, True])
def test_select_experts_preserves_fused_shared_expert_column(uniform):
    output = _select(TOPK + 1, num_fused_shared_experts=1, uniform=uniform)
    assert torch.equal(output.topk_ids[:, -1], torch.full((2,), 77, device="cuda"))
    assert torch.equal(
        output.topk_weights[:, -1], torch.full((2,), 0.75, device="cuda")
    )
    routed_weights = output.topk_weights[:, :TOPK]
    assert torch.equal(routed_weights, torch.full_like(routed_weights, 1.0 / TOPK))
    if not uniform:
        loads = _rank_loads(output.topk_ids[:, :TOPK])
        assert torch.equal(loads, torch.ones_like(loads))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
