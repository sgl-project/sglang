"""GPU coverage for the round-robin benchmark routing override's EP-rank balance."""

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _simulate_balanced_routing,
    select_experts,
)
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

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


def test_kernel_masks_padded_rows_and_scales_weights():
    tokens, real, scale = 300, 250, 1.5
    ids = torch.zeros((tokens, TOPK), dtype=torch.int64, device="cuda")
    weights = torch.ones((tokens, TOPK), dtype=torch.float32, device="cuda")
    _simulate_balanced_routing(
        ids,
        weights,
        NUM_EXPERTS,
        random=False,
        num_ranks=EP_SIZE,
        layer_id=3,
        num_token_non_padded=torch.tensor([real], device="cuda"),
        weight_scale=scale,
    )
    expected_ids, expected_weights = _route(tokens, layer_id=3)
    assert torch.equal(ids[:real], expected_ids[:real].long())
    torch.testing.assert_close(weights[:real], expected_weights[:real] * scale)
    assert torch.equal(ids[real:], torch.full_like(ids[real:], -1))
    assert torch.equal(weights[real:], torch.zeros_like(weights[real:]))


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
    routed_scaling_factor=None,
    apply_routed_scaling_factor_on_output: bool = False,
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
                shared_experts=SharedExpertsConfig(
                    num_fused_shared_experts=num_fused_shared_experts
                ),
                custom_routing_function=_fixed_router,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
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


@pytest.mark.parametrize("on_output", [False, True])
def test_select_experts_carries_a_routed_scale_folded_into_routing(on_output):
    # A scale folded into routing is skipped by the MoE epilogue, so simulated
    # weights must carry it, as the in-kernel routers' encoded logits do; otherwise
    # the epilogue applies it.
    output = _select(
        routed_scaling_factor=2.5, apply_routed_scaling_factor_on_output=on_output
    )
    expected = (2.5 if on_output else 1.0) / TOPK
    weights = output.topk_weights
    assert torch.equal(weights, torch.full_like(weights, expected))


# (num_experts, top_k, (num_expert_group, topk_group)). DeepSeek-V3 splits its 8
# slots evenly over 4 kept groups; the others are synthetic uneven splits.
GROUPED_CONFIGS = [
    (256, 8, (8, 4)),
    (64, 6, (8, 4)),
    (256, 8, (8, 3)),
    (32, 3, (4, 2)),
]


def _route_grouped(
    num_tokens,
    num_experts,
    topk,
    groups,
    *,
    random=False,
    seed=None,
    num_ranks=1,
    token_shard_rank=0,
    num_token_shards=1,
    layer_id=0,
    num_token_non_padded=None,
):
    ids = torch.full((num_tokens, topk), -2, dtype=torch.int32, device="cuda")
    weights = torch.zeros((num_tokens, topk), dtype=torch.float32, device="cuda")
    _simulate_balanced_routing(
        ids,
        weights,
        num_experts,
        random=random,
        seed=seed,
        num_ranks=num_ranks,
        layer_id=layer_id,
        token_shard_rank=token_shard_rank,
        num_token_shards=num_token_shards,
        num_token_non_padded=num_token_non_padded,
        groups=groups,
    )
    return ids, weights


@pytest.mark.parametrize("num_experts, topk, groups", GROUPED_CONFIGS)
@pytest.mark.parametrize("random", [False, True])
def test_grouped_sets_stay_within_the_kept_groups(num_experts, topk, groups, random):
    # A grouped router can only select experts from topk_group groups.
    num_groups, topk_group = groups
    ids, weights = _route_grouped(
        1000, num_experts, topk, groups, random=random, seed=7, layer_id=3
    )
    assert int(ids.min()) >= 0 and int(ids.max()) < num_experts
    for row in ids.long():
        assert row.unique().numel() == topk, row
        assert (row // (num_experts // num_groups)).unique().numel() <= topk_group
    assert torch.equal(weights, torch.full_like(weights, 1.0 / topk))


@pytest.mark.parametrize("num_experts, topk, groups", GROUPED_CONFIGS)
@pytest.mark.parametrize("num_shards, layer_id", [(1, 0), (4, 5)])
def test_grouped_round_robin_balances_every_expert(
    num_experts, topk, groups, num_shards, layer_id
):
    # num_experts global tokens are num_experts // num_groups whole cycles, each
    # giving every group k slots, so every expert takes exactly k slots, also when
    # equal DP shards interleave their global token numbers.
    per_shard = num_experts // num_shards
    ids = torch.cat(
        [
            _route_grouped(
                per_shard,
                num_experts,
                topk,
                groups,
                token_shard_rank=rank,
                num_token_shards=num_shards,
                layer_id=layer_id,
            )[0]
            for rank in range(num_shards)
        ]
    )
    loads = torch.bincount(ids.flatten().long(), minlength=num_experts)
    assert torch.equal(loads, torch.full_like(loads, topk))


@pytest.mark.parametrize("ep_size", [1, 2, 4, 8, 16, 32])
def test_grouped_round_robin_balances_ep_ranks_and_bounds_fan_out(ep_size):
    # DeepSeek-V3: with one group per EP rank (EP8) a token reaches at most 4 ranks.
    ids, _ = _route_grouped(NUM_EXPERTS, NUM_EXPERTS, TOPK, (8, 4), num_ranks=ep_size)
    loads = _rank_loads(ids, ep_size)
    assert torch.equal(loads, torch.full_like(loads, NUM_EXPERTS * TOPK // ep_size))
    if ep_size == 8:
        ranks = ids.long() // (NUM_EXPERTS // ep_size)
        assert max(row.unique().numel() for row in ranks) == 4


@pytest.mark.parametrize("ep_size", [16, 32])
@pytest.mark.parametrize("num_tokens", [8, 16])
def test_grouped_round_robin_spreads_small_batches_over_ranks_sharing_a_group(
    ep_size, num_tokens
):
    # DeepSeek-V3 decode with more EP ranks than groups: one cycle of 8 tokens gives
    # each group 8 slots, which must alternate over the group's ranks rather than
    # fill its first rank and leave the others idle.
    ids, _ = _route_grouped(
        num_tokens, NUM_EXPERTS, TOPK, (8, 4), num_ranks=ep_size, layer_id=3
    )
    loads = _rank_loads(ids, ep_size)
    assert torch.equal(loads, torch.full_like(loads, num_tokens * TOPK // ep_size))


@pytest.mark.parametrize("num_shards", [3, 8, 16])
def test_grouped_round_robin_balances_each_uneven_shard_on_its_own(num_shards):
    # DeepSeek-V3 under scattered DP: a shard's whole 8-token cycles must load every
    # EP rank equally whatever the other shards hold, rather than pinning a shard to
    # the few groups its global token numbers would reach.
    for rank in range(num_shards):
        num_tokens = 8 * (rank % 3)
        ids, _ = _route_grouped(
            num_tokens,
            NUM_EXPERTS,
            TOPK,
            (8, 4),
            num_ranks=8,
            token_shard_rank=rank,
            num_token_shards=num_shards,
            layer_id=2,
        )
        loads = _rank_loads(ids, 8)
        assert torch.equal(loads, torch.full_like(loads, num_tokens)), (rank, loads)


def test_grouped_round_robin_masks_padded_rows():
    ids, weights = _route_grouped(
        10,
        NUM_EXPERTS,
        TOPK,
        (8, 4),
        num_token_non_padded=torch.tensor([7], dtype=torch.int32).cuda(),
    )
    expected, _ = _route_grouped(10, NUM_EXPERTS, TOPK, (8, 4))
    assert torch.equal(ids[:7], expected[:7])
    assert torch.equal(ids[7:], torch.full_like(ids[7:], -1))
    assert torch.equal(weights[7:], torch.zeros_like(weights[7:]))


def _bypassed_config(method, top_k, groups=None, **kwargs):
    from sglang.srt.layers.moe.topk import TopKOutputFormat
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    return TopKConfig(
        top_k=top_k,
        use_grouped_topk=groups is not None,
        num_expert_group=None if groups is None else groups[0],
        topk_group=None if groups is None else groups[1],
        runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
        routing_method_type=method,
        output_format=TopKOutputFormat.BYPASSED,
        layer_id=3,
        **kwargs,
    )


def _simulation_scope(*, uniform, ep_size=1):
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(uniform))
    stack.enter_context(envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(not uniform))
    stack.enter_context(get_parallel().override(moe_ep_size=ep_size))
    stack.enter_context(
        patch(
            "sglang.srt.layers.moe.topk.is_moe_input_scattered_across_dp_ranks",
            return_value=False,
        )
    )
    return stack


def _encode(config, num_tokens, *, seed=None, dtype=torch.bfloat16):
    from sglang.srt.layers.moe.topk import (
        encode_simulated_router_logits,
        simulated_logit_encoding,
    )

    logits = torch.zeros((num_tokens, NUM_EXPERTS), dtype=dtype, device="cuda")
    return encode_simulated_router_logits(
        logits,
        encoding=simulated_logit_encoding(config),
        layer_id=config.layer_id,
        seed=seed,
    )


def _targets(encoded):
    rows = [torch.nonzero(row > -64.0).flatten() for row in encoded]
    return torch.stack(rows).int()


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("groups", [None, (8, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ep_size", [1, 8])
def test_encoder_marks_exactly_the_generated_experts(uniform, groups, dtype, ep_size):
    from sglang.srt.layers.moe.utils import RoutingMethodType

    config = _bypassed_config(RoutingMethodType.DeepSeekV3, TOPK, groups)
    with _simulation_scope(uniform=uniform, ep_size=ep_size):
        encoded = _encode(config, 33, seed=11, dtype=dtype)
    ids = torch.full((33, TOPK), -1, dtype=torch.int32, device="cuda")
    _simulate_balanced_routing(
        ids,
        torch.empty((33, TOPK), device="cuda"),
        NUM_EXPERTS,
        random=uniform,
        seed=11,
        num_ranks=ep_size,
        layer_id=3,
        groups=groups,
    )
    assert encoded.dtype == dtype
    assert torch.equal(_targets(encoded), ids.sort(-1).values)
    assert int((encoded == -64.0).sum()) == 33 * (NUM_EXPERTS - TOPK)


@pytest.mark.parametrize("contiguous", [True, False])
def test_encoder_in_place_reuses_contiguous_logits_only(contiguous):
    # The bypass path overwrites the raw logits instead of allocating per forward;
    # rows it cannot address by one stride get a fresh buffer and stay untouched.
    from sglang.srt.layers.moe.topk import (
        encode_simulated_router_logits,
        simulated_logit_encoding,
    )
    from sglang.srt.layers.moe.utils import RoutingMethodType

    config = _bypassed_config(RoutingMethodType.DeepSeekV3, TOPK, (8, 4))
    shape = (9, NUM_EXPERTS) if contiguous else (NUM_EXPERTS, 9)
    raw = torch.full(shape, 3.0, dtype=torch.bfloat16, device="cuda")
    logits = raw if contiguous else raw.t()
    with _simulation_scope(uniform=False, ep_size=8):
        encoded = encode_simulated_router_logits(
            logits,
            encoding=simulated_logit_encoding(config),
            layer_id=config.layer_id,
            in_place=True,
        )
        expected = _encode(config, 9)
    assert (encoded.data_ptr() == raw.data_ptr()) == contiguous
    assert bool((raw == 3.0).all()) != contiguous
    assert torch.equal(encoded, expected)


def test_encoder_from_ids_marks_exactly_the_given_experts_and_skips_padding():
    # gfx1250 W4A8 routes from logits even when TopK materialized simulated ids, so
    # the logits must select those ids (and padded -1 rows must not fault).
    from sglang.srt.layers.moe.topk import encode_router_logits_from_topk_ids

    ids, _ = _route(5)
    ids[3:] = -1
    encoded = encode_router_logits_from_topk_ids(
        ids, num_experts=NUM_EXPERTS, dtype=torch.bfloat16
    )
    assert encoded.dtype == torch.bfloat16
    assert torch.equal(_targets(encoded[:3]), ids[:3].sort(-1).values)
    assert bool((encoded[3:] == -64.0).all())
    weights = encoded[:3].float().softmax(-1).topk(TOPK).values
    assert torch.equal(weights, torch.full_like(weights, 1.0 / TOPK))


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize(
    "method_name, groups, kwargs",
    [
        (
            "DeepSeekV3",
            (8, 4),
            dict(scoring_func="sigmoid", routed_scaling_factor=2.5),
        ),
        ("Renormalize", None, {}),
    ],
)
def test_select_experts_reproduces_encoded_ids_without_resimulating(
    uniform, method_name, groups, kwargs
):
    # The FP8 runner materializes bypassed routing (to_standard) on encoded logits
    # and the simulated config; a second simulation would reseed uniform mode.
    from dataclasses import replace

    from sglang.srt.layers.moe.utils import RoutingMethodType

    method = RoutingMethodType[method_name]
    bias = torch.zeros(NUM_EXPERTS, device="cuda") if groups else None
    config = _bypassed_config(method, TOPK, groups, correction_bias=bias, **kwargs)
    with _simulation_scope(uniform=uniform):
        encoded = _encode(config, 64, seed=5)
        output = select_experts(
            hidden_states=torch.zeros((64, 16), device="cuda", dtype=torch.bfloat16),
            router_logits=encoded,
            topk_config=replace(config, routing_simulation_encoded=True),
            layer_id=3,
        )
    expected = _targets(encoded)
    assert torch.equal(output.topk_ids.int().sort(-1).values, expected)
    torch.testing.assert_close(
        output.topk_weights.float(), torch.full((64, TOPK), 1.0 / TOPK).cuda()
    )


def _eplb_physical_maps(num_redundant=32):
    # Laid out like ExpertLocationMetadata._init_raw: each logical expert's replica
    # slots first, padded with -1 to the physical expert count.
    generator = torch.Generator().manual_seed(0)
    num_physical = NUM_EXPERTS + num_redundant
    physical_to_logical = torch.cat(
        [
            torch.arange(NUM_EXPERTS),
            torch.randperm(NUM_EXPERTS, generator=generator)[:num_redundant],
        ]
    )[torch.randperm(num_physical, generator=generator)]
    to_physical_map = torch.full((NUM_EXPERTS, num_physical), -1, dtype=torch.int32)
    for expert in range(NUM_EXPERTS):
        replicas = (physical_to_logical == expert).nonzero().flatten()
        to_physical_map[expert, : len(replicas)] = replicas.int()
    logical_count = (to_physical_map != -1).sum(-1).int()
    return to_physical_map.cuda(), logical_count.cuda()


def _megagate_replicas(logical_ids, to_physical_map, logical_count, ep_rank):
    tokens = torch.arange(logical_ids.shape[0], device="cuda").unsqueeze(1)
    logical = logical_ids.long()
    replica = (ep_rank + tokens * 23333) % logical_count[logical]
    return to_physical_map[logical, replica], replica


@pytest.mark.parametrize("groups", [None, (8, 4)])
@pytest.mark.parametrize("random", [False, True])
@pytest.mark.parametrize("ep_rank", [0, 5])
def test_physical_map_places_generated_ids_on_megagate_replicas(
    groups, random, ep_rank
):
    to_physical_map, logical_count = _eplb_physical_maps()
    num_tokens, num_non_padded = 300, 290

    def route(**physical):
        ids = torch.full((num_tokens, TOPK), -2, dtype=torch.int32, device="cuda")
        weights = torch.zeros((num_tokens, TOPK), device="cuda")
        _simulate_balanced_routing(
            ids,
            weights,
            NUM_EXPERTS,
            random=random,
            seed=7,
            num_ranks=8,
            layer_id=2,
            num_token_non_padded=torch.tensor(
                [num_non_padded], dtype=torch.int32, device="cuda"
            ),
            groups=groups,
            **physical,
        )
        return ids, weights

    logical, _ = route()
    ids, weights = route(
        to_physical_map=to_physical_map, logical_count=logical_count, ep_rank=ep_rank
    )
    expected, replica = _megagate_replicas(
        logical[:num_non_padded], to_physical_map, logical_count, ep_rank
    )
    assert (replica > 0).any(), "no token reached a redundant replica"
    assert torch.equal(ids[:num_non_padded], expected)
    assert torch.equal(ids[num_non_padded:], torch.full_like(ids[num_non_padded:], -1))
    assert torch.equal(
        weights[:num_non_padded], torch.full((num_non_padded, TOPK), 1.0 / TOPK).cuda()
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
