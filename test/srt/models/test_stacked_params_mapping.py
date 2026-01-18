"""Unit tests for StackedParamsMixin."""

from sglang.srt.models.stacked_params_mixin import StackedParamsMixin


class DummyModel(StackedParamsMixin):
    """Dummy model for testing the mixin."""

    def __init__(self):
        self.stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        self.expert_params_mapping = [
            ("mlp.experts.{}.gate_up_proj", "mlp.experts.gate_proj", 0, 0),
        ]


class DummyModelNoMapping(StackedParamsMixin):
    """Dummy model with no mappings defined."""

    pass


def test_map_weight_name_qkv():
    """Test mapping q/k/v projections to fused qkv_proj."""
    m = DummyModel()

    mapped, shard, num_shards, expert = m.map_weight_name("layers.0.attn.q_proj.weight")
    assert "qkv_proj" in mapped
    assert shard == "q"
    assert num_shards == 3
    assert expert is None

    mapped, shard, num_shards, expert = m.map_weight_name("layers.0.attn.k_proj.weight")
    assert "qkv_proj" in mapped
    assert shard == "k"
    assert num_shards == 3
    assert expert is None

    mapped, shard, num_shards, expert = m.map_weight_name("layers.0.attn.v_proj.weight")
    assert "qkv_proj" in mapped
    assert shard == "v"
    assert num_shards == 3
    assert expert is None


def test_map_weight_name_gate_up():
    """Test mapping gate/up projections to fused gate_up_proj."""
    m = DummyModel()

    mapped, shard, num_shards, expert = m.map_weight_name(
        "layers.2.mlp.gate_proj.weight"
    )
    assert "gate_up_proj" in mapped
    assert shard == 0
    assert num_shards == 2
    assert expert is None

    mapped, shard, num_shards, expert = m.map_weight_name("layers.2.mlp.up_proj.weight")
    assert "gate_up_proj" in mapped
    assert shard == 1
    assert num_shards == 2
    assert expert is None


def test_map_weight_name_no_match():
    """Test that unmatched weights return the original name."""
    m = DummyModel()

    mapped, shard, num_shards, expert = m.map_weight_name(
        "layers.0.mlp.down_proj.weight"
    )
    assert mapped == "layers.0.mlp.down_proj.weight"
    assert shard is None
    assert num_shards == 1
    assert expert is None


def test_expert_mapping():
    """Test expert mapping for MoE models."""
    m = DummyModel()

    mapped, shard, num_shards, expert = m.map_weight_name(
        "mlp.experts.gate_proj.weight"
    )
    # The mixin does a direct replace; expert detection returns expert id from mapping if matched
    assert expert == 0
    assert shard == 0


def test_get_packed_modules_mapping():
    """Test the inverse mapping generation."""
    m = DummyModel()

    packed = m.get_packed_modules_mapping()
    assert "qkv_proj" in packed
    assert len(packed["qkv_proj"]) == 3
    assert ("q_proj", "q") in packed["qkv_proj"]
    assert ("k_proj", "k") in packed["qkv_proj"]
    assert ("v_proj", "v") in packed["qkv_proj"]

    assert "gate_up_proj" in packed
    assert len(packed["gate_up_proj"]) == 2
    assert ("gate_proj", 0) in packed["gate_up_proj"]
    assert ("up_proj", 1) in packed["gate_up_proj"]


def test_get_num_shards_for_param():
    """Test shard counting for parameters."""
    m = DummyModel()

    assert m.get_num_shards_for_param("qkv_proj") == 3
    assert m.get_num_shards_for_param("gate_up_proj") == 2
    assert m.get_num_shards_for_param("nonexistent") == 1  # Default to 1


def test_no_mapping_model():
    """Test that models without mappings return sensible defaults."""
    m = DummyModelNoMapping()

    assert m.get_stacked_params_mapping() == []
    assert m.get_expert_params_mapping() == []
    assert m.get_packed_modules_mapping() == {}

    mapped, shard, num_shards, expert = m.map_weight_name("any.weight.name")
    assert mapped == "any.weight.name"
    assert shard is None
    assert num_shards == 1
    assert expert is None


def test_get_stacked_params_mapping():
    """Test the get_stacked_params_mapping method returns a list."""
    m = DummyModel()
    mapping = m.get_stacked_params_mapping()
    assert isinstance(mapping, list)
    assert len(mapping) == 5


def test_get_expert_params_mapping():
    """Test the get_expert_params_mapping method returns a list."""
    m = DummyModel()
    mapping = m.get_expert_params_mapping()
    assert isinstance(mapping, list)
    assert len(mapping) == 1
