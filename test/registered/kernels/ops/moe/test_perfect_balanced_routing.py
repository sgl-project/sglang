"""GPU coverage for perfect-balanced benchmark routing."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens_live_gpu,
    set_dp_buffer_len,
    update_dp_global_num_tokens_live_gpu,
)
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _simulate_balanced_routing,
    select_experts,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 256
TOPK = 8
EP_SIZE = 16


def _allocate(num_tokens: int, topk: int = TOPK):
    ids = torch.full((num_tokens, topk), -1, dtype=torch.int32, device="cuda")
    weights = torch.zeros((num_tokens, topk), dtype=torch.float32, device="cuda")
    return ids, weights


def _perfect_route(
    num_tokens: int,
    *,
    token_shard_rank: int = 0,
    num_token_shards: int = 1,
    sequence_shard_rank: int = 0,
    token_counts: torch.Tensor | None = None,
):
    ids, weights = _allocate(num_tokens)
    _simulate_balanced_routing(
        ids,
        weights,
        NUM_EXPERTS,
        random=False,
        perfect_balanced=True,
        num_ranks=EP_SIZE,
        token_shard_rank=token_shard_rank,
        num_token_shards=num_token_shards,
        sequence_shard_rank=sequence_shard_rank,
        dp_token_counts=token_counts,
    )
    return ids, weights


def test_uneven_dp_shards_match_one_contiguous_token_stream():
    live_counts = [2] + [1] * 15
    counts_gpu = torch.tensor(live_counts, dtype=torch.int32, device="cuda")
    sharded_ids = []
    for rank, count in enumerate(live_counts):
        ids, _ = _perfect_route(
            count,
            token_shard_rank=rank,
            num_token_shards=len(live_counts),
            token_counts=counts_gpu,
        )
        sharded_ids.append(ids)

    expected_ids, _ = _perfect_route(sum(live_counts))
    assert torch.equal(torch.cat(sharded_ids), expected_ids)


def test_indivisible_expert_count_keeps_every_expert_reachable():
    ids, weights = _allocate(25)
    _simulate_balanced_routing(
        ids,
        weights,
        200,
        random=False,
        perfect_balanced=True,
        num_ranks=EP_SIZE,
    )
    assert ids.unique().numel() == 200


def test_round_robin_mapping_is_unchanged():
    num_tokens, layer_id = 17, 3
    ids, weights = _allocate(num_tokens)
    _simulate_balanced_routing(
        ids,
        weights,
        NUM_EXPERTS,
        random=False,
        layer_id=layer_id,
    )
    tokens = torch.arange(num_tokens, dtype=torch.int32, device="cuda").unsqueeze(1)
    slots = torch.arange(TOPK, dtype=torch.int32, device="cuda").unsqueeze(0)
    expected = (tokens + layer_id + slots * (NUM_EXPERTS // TOPK)) % NUM_EXPERTS
    assert torch.equal(ids, expected)
    assert torch.equal(weights, torch.full_like(weights, 1.0 / TOPK))


def test_uniform_mapping_preserves_structural_contract():
    ids, weights = _allocate(4096)
    _simulate_balanced_routing(
        ids,
        weights,
        NUM_EXPERTS,
        random=True,
        seed=17,
    )
    assert int(ids.min()) >= 0
    assert int(ids.max()) < NUM_EXPERTS
    assert all(row.unique().numel() == TOPK for row in ids[:64])
    assert torch.equal(weights, torch.full_like(weights, 1.0 / TOPK))


def test_sequence_parallel_shards_match_one_contiguous_token_stream():
    live_tokens = 5
    local_padded_tokens = 2
    counts_gpu = torch.tensor([live_tokens], dtype=torch.int32, device="cuda")
    sharded_ids = []
    for rank, local_live_tokens in enumerate((2, 2, 1, 0)):
        ids, _ = _perfect_route(
            local_padded_tokens,
            sequence_shard_rank=rank,
            token_counts=counts_gpu,
        )
        sharded_ids.append(ids[:local_live_tokens])

    expected_ids, _ = _perfect_route(live_tokens)
    assert torch.equal(torch.cat(sharded_ids), expected_ids)


def test_cuda_graph_replay_reads_live_counts_not_padded_counts():
    live = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
    padded = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
    with envs.SGLANG_SIMULATE_PERFECT_BALANCED_EXPERTS.override(True):
        update_dp_global_num_tokens_live_gpu(live)
        output = torch.empty_like(live)

        set_dp_buffer_len(8, 4, True, [4, 4], padded, padded)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            set_dp_buffer_len(8, 4, True, [4, 4], padded, padded)
            output.copy_(get_dp_global_num_tokens_live_gpu())

        update_dp_global_num_tokens_live_gpu(
            torch.tensor([1, 2], dtype=torch.int32, device="cuda")
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            output, torch.tensor([1, 2], dtype=torch.int32, device="cuda")
        )


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
    top_k: int,
    num_fused_shared_experts: int = 0,
    *,
    dp_size: int = 1,
    token_counts: torch.Tensor | None = None,
    expected_error: str | None = None,
):
    parallel = SimpleNamespace(
        attn_dp_rank=0,
        attn_dp_size=dp_size,
        attn_tp_rank=0,
        moe_ep_size=EP_SIZE,
    )
    with (
        envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(False),
        envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(False),
        envs.SGLANG_SIMULATE_PERFECT_BALANCED_EXPERTS.override(True),
        patch("sglang.srt.layers.moe.topk.get_parallel", return_value=parallel),
        patch(
            "sglang.srt.layers.moe.topk.get_forward",
            return_value=SimpleNamespace(sp_active=False),
        ),
        patch(
            "sglang.srt.layers.moe.topk.is_moe_input_scattered_across_dp_ranks",
            return_value=True,
        ),
        patch(
            "sglang.srt.layers.moe.topk.get_dp_global_num_tokens_live_gpu",
            return_value=token_counts,
        ),
    ):

        def run_select_experts():
            return select_experts(
                hidden_states=torch.zeros((2, 4), device="cuda"),
                router_logits=torch.zeros((2, NUM_EXPERTS), device="cuda"),
                topk_config=TopKConfig(
                    top_k=top_k,
                    num_fused_shared_experts=num_fused_shared_experts,
                    custom_routing_function=_fixed_router,
                ),
                layer_id=0,
            )

        if expected_error is not None:
            with pytest.raises(RuntimeError, match=expected_error):
                run_select_experts()
            return None
        return run_select_experts()


def test_select_experts_does_not_require_token_counts_for_dp1():
    output = _select(TOPK)
    rank_counts = torch.bincount(
        output.topk_ids.flatten().long() // (NUM_EXPERTS // EP_SIZE),
        minlength=EP_SIZE,
    )
    assert torch.equal(rank_counts, torch.ones_like(rank_counts))


def test_select_experts_requires_live_token_counts_for_scattered_dp():
    _select(TOPK, dp_size=2, expected_error="requires live DP token counts")


def test_select_experts_preserves_fused_shared_expert_column():
    output = _select(TOPK + 1, num_fused_shared_experts=1)
    assert torch.equal(output.topk_ids[:, -1], torch.full((2,), 77, device="cuda"))
    assert torch.equal(
        output.topk_weights[:, -1], torch.full((2,), 0.75, device="cuda")
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
