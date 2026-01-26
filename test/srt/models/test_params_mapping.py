"""Unit tests for RemapParamsMixin.map_weight_name()."""

import pytest

from sglang.srt.models.remap_params_mixin import RemapParamsMixin


class DummyModel(RemapParamsMixin):
    def __init__(self):
        self.stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # 2 experts with 3 shards each (gate, down, up)
        self.expert_params_mapping = [
            ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
            ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
            ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
            ("experts.w13_", "experts.1.gate_proj.", 1, "w1"),
            ("experts.w2_", "experts.1.down_proj.", 1, "w2"),
            ("experts.w13_", "experts.1.up_proj.", 1, "w3"),
        ]


@pytest.fixture
def model():
    return DummyModel()


@pytest.mark.parametrize(
    "ckpt_name,expected_mapped,expected_shard,expected_num_shards,expected_expert,expected_num_experts",
    [
        # QKV fusion
        (
            "layers.0.attn.q_proj.weight",
            "layers.0.attn.qkv_proj.weight",
            "q",
            3,
            None,
            0,
        ),
        (
            "layers.0.attn.k_proj.weight",
            "layers.0.attn.qkv_proj.weight",
            "k",
            3,
            None,
            0,
        ),
        (
            "layers.0.attn.v_proj.weight",
            "layers.0.attn.qkv_proj.weight",
            "v",
            3,
            None,
            0,
        ),
        # Gate/Up fusion
        (
            "layers.2.mlp.gate_proj.weight",
            "layers.2.mlp.gate_up_proj.weight",
            0,
            2,
            None,
            0,
        ),
        (
            "layers.2.mlp.up_proj.weight",
            "layers.2.mlp.gate_up_proj.weight",
            1,
            2,
            None,
            0,
        ),
        # Expert weights (MoE) - now with 2 experts
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.w13_weight",
            "w1",
            1,
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
            1,
            0,
            2,
        ),
        (
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.layers.0.mlp.experts.w13_weight",
            "w1",
            1,
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
            0,
        ),
        ("embed_tokens.weight", "embed_tokens.weight", None, 1, None, 0),
        # Scale remapping - Standard patterns
        (
            "model.layers.0.self_attn.k_scale",
            "model.layers.0.self_attn.attn.k_scale",
            None,
            1,
            None,
            0,
        ),
        (
            "model.layers.0.self_attn.v_scale",
            "model.layers.0.self_attn.attn.v_scale",
            None,
            1,
            None,
            0,
        ),
    ],
)
def test_map_weight_name(
    model,
    ckpt_name,
    expected_mapped,
    expected_shard,
    expected_num_shards,
    expected_expert,
    expected_num_experts,
):
    mapped, shard, num_shards, expert, num_experts = model.map_weight_name(ckpt_name)
    assert mapped == expected_mapped
    assert shard == expected_shard
    assert num_shards == expected_num_shards
    assert expert == expected_expert
    assert num_experts == expected_num_experts
