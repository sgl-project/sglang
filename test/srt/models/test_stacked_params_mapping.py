"""Unit tests for StackedParamsMixin.map_weight_name()."""

import pytest

from sglang.srt.models.stacked_params_mixin import StackedParamsMixin


class DummyModel(StackedParamsMixin):
    def __init__(self):
        self.stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        self.expert_params_mapping = [
            ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
            ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
            ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
        ]


@pytest.fixture
def model():
    return DummyModel()


@pytest.mark.parametrize(
    "ckpt_name,expected_mapped,expected_shard,expected_num_shards,expected_expert",
    [
        # QKV fusion
        ("layers.0.attn.q_proj.weight", "layers.0.attn.qkv_proj.weight", "q", 3, None),
        ("layers.0.attn.k_proj.weight", "layers.0.attn.qkv_proj.weight", "k", 3, None),
        ("layers.0.attn.v_proj.weight", "layers.0.attn.qkv_proj.weight", "v", 3, None),
        # Gate/Up fusion
        (
            "layers.2.mlp.gate_proj.weight",
            "layers.2.mlp.gate_up_proj.weight",
            0,
            2,
            None,
        ),
        ("layers.2.mlp.up_proj.weight", "layers.2.mlp.gate_up_proj.weight", 1, 2, None),
        # Expert weights (MoE)
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.w13_weight",
            "w1",
            1,
            0,
        ),
        (
            "model.layers.0.mlp.experts.0.down_proj.weight",
            "model.layers.0.mlp.experts.w2_weight",
            "w2",
            1,
            0,
        ),
        (
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.w13_weight",
            "w3",
            1,
            0,
        ),
        # No match - returns original
        (
            "layers.0.mlp.down_proj.weight",
            "layers.0.mlp.down_proj.weight",
            None,
            1,
            None,
        ),
        ("embed_tokens.weight", "embed_tokens.weight", None, 1, None),
    ],
)
def test_map_weight_name(
    model,
    ckpt_name,
    expected_mapped,
    expected_shard,
    expected_num_shards,
    expected_expert,
):
    mapped, shard, num_shards, expert = model.map_weight_name(ckpt_name)
    assert mapped == expected_mapped
    assert shard == expected_shard
    assert num_shards == expected_num_shards
    assert expert == expected_expert
