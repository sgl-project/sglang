"""Unit tests for ParameterMapper."""

import pytest

from sglang.srt.model_loader.parameter_mapper import ParameterMapper


@pytest.fixture
def mapper():
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    expert_params_mapping = [
        ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
        ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
        ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
        ("experts.w13_", "experts.1.gate_proj.", 1, "w1"),
        ("experts.w2_", "experts.1.down_proj.", 1, "w2"),
        ("experts.w13_", "experts.1.up_proj.", 1, "w3"),
    ]
    return ParameterMapper(
        stacked_params_mapping=stacked_params_mapping,
        expert_params_mapping=expert_params_mapping,
        num_local_experts=2,
    )


@pytest.mark.parametrize(
    "ckpt_name,expected_mapped,expected_shard,expected_num_shards,expected_expert,expected_num_local_experts",
    [
        # QKV fusion
        (
            "layers.0.attn.q_proj.weight",
            "layers.0.attn.qkv_proj.weight",
            "q",
            3,
            None,
            None,
        ),
        (
            "layers.0.attn.k_proj.weight",
            "layers.0.attn.qkv_proj.weight",
            "k",
            3,
            None,
            None,
        ),
        (
            "layers.0.attn.v_proj.weight",
            "layers.0.attn.qkv_proj.weight",
            "v",
            3,
            None,
            None,
        ),
        # Gate/Up fusion
        (
            "layers.2.mlp.gate_proj.weight",
            "layers.2.mlp.gate_up_proj.weight",
            0,
            2,
            None,
            None,
        ),
        (
            "layers.2.mlp.up_proj.weight",
            "layers.2.mlp.gate_up_proj.weight",
            1,
            2,
            None,
            None,
        ),
        # Expert weights (MoE) - w13_ has 2 shards (w1, w3), w2_ has 1 shard (w2)
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.w13_weight",
            "w1",
            2,
            0,
            2,
        ),
        (
            "model.layers.0.mlp.experts.0.down_proj.weight",
            "model.layers.0.mlp.experts.w2_weight",
            "w2",
            1,
            0,
            2,
        ),
        (
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.w13_weight",
            "w3",
            2,
            0,
            2,
        ),
        (
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.layers.0.mlp.experts.w13_weight",
            "w1",
            2,
            1,
            2,
        ),
        # No match - returns original
        (
            "layers.0.mlp.down_proj.weight",
            "layers.0.mlp.down_proj.weight",
            None,
            1,
            None,
            None,
        ),
        ("embed_tokens.weight", "embed_tokens.weight", None, 1, None, None),
        # Scale remapping - Standard patterns
        (
            "model.layers.0.self_attn.k_scale",
            "model.layers.0.self_attn.attn.k_scale",
            None,
            1,
            None,
            None,
        ),
        (
            "model.layers.0.self_attn.v_scale",
            "model.layers.0.self_attn.attn.v_scale",
            None,
            1,
            None,
            None,
        ),
    ],
)
def test_map_weight_name(
    mapper,
    ckpt_name,
    expected_mapped,
    expected_shard,
    expected_num_shards,
    expected_expert,
    expected_num_local_experts,
):
    result = mapper.map(ckpt_name)
    assert result.sglang_name == expected_mapped
    assert result.shard_id == expected_shard
    assert result.num_shards == expected_num_shards
    assert result.expert_id == expected_expert
    assert result.num_local_experts == expected_num_local_experts
