# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.moe import (
    LingBotVideoSparseMoeBlock,
    MoeExpertParallelInfo,
    resolve_moe_expert_parallel,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_video_moe import (
    is_expert_parallel_param,
    pack_expert_weights,
)


def _ep_info(ep_size: int, ep_rank: int, num_experts: int) -> MoeExpertParallelInfo:
    num_local = num_experts // ep_size
    return MoeExpertParallelInfo(
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_local_experts=num_local,
        local_expert_start=ep_rank * num_local,
    )


def _make_moe_block(**overrides):
    kwargs = dict(
        hidden_size=8,
        intermediate_size=4,
        num_experts=8,
        top_k=2,
        score_func="sigmoid",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=1,
    )
    kwargs.update(overrides)
    return LingBotVideoSparseMoeBlock(**kwargs)


def test_expert_parallel_shards_tile_the_dense_checkpoint():
    num_experts, intermediate_size, hidden_size, ep_size = 8, 3, 4, 4
    w1 = torch.arange(
        num_experts * intermediate_size * hidden_size, dtype=torch.float32
    ).reshape(num_experts, intermediate_size, hidden_size)
    w2 = torch.arange(
        num_experts * hidden_size * intermediate_size, dtype=torch.float32
    ).reshape(num_experts, hidden_size, intermediate_size)
    w3 = w1 + 100.0
    router = torch.randn(num_experts, hidden_size)
    src = [
        ("blocks.0.ffn.experts.w1", w1),
        ("blocks.0.ffn.experts.w2", w2),
        ("blocks.0.ffn.experts.w3", w3),
        ("blocks.0.ffn.router.weight", router),
    ]

    shards = [
        dict(
            pack_expert_weights(iter(src), ep_info=_ep_info(ep_size, rank, num_experts))
        )
        for rank in range(ep_size)
    ]

    for shard in shards:
        assert (
            shard["blocks.0.ffn.experts.w13_weight"].shape[0] == num_experts // ep_size
        )
        assert shard["blocks.0.ffn.experts.w2"].shape[0] == num_experts // ep_size
        for key in (
            "blocks.0.ffn.experts.w13_weight",
            "blocks.0.ffn.experts.w2",
        ):
            tensor = shard[key]
            assert tensor.untyped_storage().nbytes() == (
                tensor.numel() * tensor.element_size()
            )
        torch.testing.assert_close(shard["blocks.0.ffn.router.weight"], router)

    for key, dense in (
        ("blocks.0.ffn.experts.w13_weight", torch.cat((w1, w3), dim=1)),
        ("blocks.0.ffn.experts.w2", w2),
    ):
        torch.testing.assert_close(torch.cat([s[key] for s in shards]), dense)


def test_is_expert_parallel_param_matches_only_expert_weights():
    assert is_expert_parallel_param("blocks.0.ffn.experts.w13_weight")
    assert is_expert_parallel_param("blocks.7.ffn.experts.w2")
    assert not is_expert_parallel_param("blocks.0.ffn.router.weight")
    assert not is_expert_parallel_param("blocks.0.ffn.shared_experts.up_proj.weight")
    assert not is_expert_parallel_param("blocks.0.attn.to_q.weight")


def test_moe_block_without_expert_parallel_owns_every_expert():
    block = _make_moe_block()

    assert not block.ep_info.enabled
    assert block.ep_info.ep_size == 1
    assert block.num_local_experts == 8
    assert block.experts.w13_weight.shape[0] == 8


def test_moe_block_rejects_expert_parallel_without_an_sp_aligned_group():
    with pytest.raises(RuntimeError, match="reuses the SP group"):
        _make_moe_block(ep_info=_ep_info(2, 0, 8))


@pytest.mark.parametrize("ep_rank", [0, 3])
def test_moe_block_with_expert_parallel_owns_only_its_shard(ep_rank):
    with patch(
        "sglang.multimodal_gen.runtime.layers.moe.create_moe_token_dispatcher",
        return_value=object(),
    ):
        block = _make_moe_block(ep_info=_ep_info(4, ep_rank, 8))

    assert block.ep_info.enabled
    assert block.num_local_experts == 2
    assert block.ep_info.local_expert_start == ep_rank * 2
    assert block.experts.w13_weight.shape[0] == 2
    assert block.experts.w2.shape[0] == 2
    assert block.router.weight.shape[0] == 8


def test_expert_parallel_rejects_indivisible_expert_count():
    with pytest.raises(ValueError, match="divisible"):
        resolve_moe_expert_parallel(8, ep_size=3)


def test_resolve_ep_disabled_ignores_sp():
    info = resolve_moe_expert_parallel(8, ep_size=1)
    assert not info.enabled
    assert info.ep_size == 1
    assert info.ep_rank == 0
    assert info.num_local_experts == 8


def test_resolve_ep_uses_sp_rank_for_larger_degree():
    with (
        patch(
            "sglang.multimodal_gen.runtime.layers.moe.get_sp_world_size",
            return_value=4,
        ),
        patch(
            "sglang.multimodal_gen.runtime.layers.moe.get_sp_parallel_rank",
            return_value=3,
        ),
    ):
        info = resolve_moe_expert_parallel(8, ep_size=4)

    assert info.enabled
    assert info.ep_size == 4
    assert info.ep_rank == 3
    assert info.num_local_experts == 2
    assert info.local_expert_start == 6


def test_resolve_ep_requires_matching_sp_world_size():
    with patch(
        "sglang.multimodal_gen.runtime.layers.moe.get_sp_world_size",
        return_value=4,
    ):
        with pytest.raises(RuntimeError, match="SP world size"):
            resolve_moe_expert_parallel(8, ep_size=2)
