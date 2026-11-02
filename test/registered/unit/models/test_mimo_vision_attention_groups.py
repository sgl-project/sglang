"""MiMo TP vision must stay within its independent attention group."""

from types import SimpleNamespace

import pytest
from torch import nn

from sglang.srt import server_args
from sglang.srt.models import mimo_vl, qwen2_5_vl
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class ParallelSpy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.options = kwargs


@pytest.fixture
def groups(monkeypatch):
    monkeypatch.setattr(
        server_args, "_global_server_args", SimpleNamespace(rl_on_policy_target=None)
    )
    monkeypatch.setattr(qwen2_5_vl, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(qwen2_5_vl, "get_tensor_model_parallel_rank", lambda: 5)
    monkeypatch.setattr(qwen2_5_vl, "get_attention_tp_size", lambda: 4)
    monkeypatch.setattr(qwen2_5_vl, "get_attention_tp_rank", lambda: 1)
    for name in (
        "ColumnParallelLinear",
        "MergedColumnParallelLinear",
        "RowParallelLinear",
    ):
        monkeypatch.setattr(qwen2_5_vl, name, ParallelSpy)
    monkeypatch.setattr(mimo_vl, "VisionAttention", ParallelSpy)


@pytest.mark.parametrize("encoder_dp", [False, True])
def test_mimo_vision_uses_one_consistent_group(groups, monkeypatch, encoder_dp):
    monkeypatch.setattr(
        mimo_vl,
        "get_global_server_args",
        lambda: SimpleNamespace(mm_enable_dp_encoder=encoder_dp),
    )
    config = mimo_vl.MiMoVLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        num_key_value_heads=4,
        qk_channels=4,
        kv_channels=4,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=16,
        fullatt_block_indexes=[0],
    )
    model = mimo_vl.MiMoVisionTransformer(config)
    block = model.blocks[0]
    width, rank = (1, 0) if encoder_dp else (4, 1)
    assert block.attn.options.get("use_dp_attention_reduce", False) is (not encoder_dp)
    assert (block.mlp.tp_size, block.mlp.tp_rank) == (width, rank)
    for projection in [
        block.mlp.gate_up_proj,
        block.mlp.down_proj,
        model.merger.mlp[0],
        model.merger.mlp[2],
    ]:
        assert projection.options["tp_size"] == width
        assert projection.options["tp_rank"] == rank
    for projection in [block.mlp.down_proj, model.merger.mlp[2]]:
        assert projection.options["use_dp_attention_reduce"] is (not encoder_dp)


@pytest.mark.parametrize("component", ["mlp", "merger"])
def test_existing_qwen_default_group_is_preserved(groups, component):
    if component == "mlp":
        layer = qwen2_5_vl.Qwen2_5_VLMLP(16, 32)
        projection = layer.down_proj
    else:
        layer = qwen2_5_vl.Qwen2_5_VisionPatchMerger(16, 16)
        projection = layer.mlp[2]
    assert projection.options["tp_size"] == 8
    assert projection.options["tp_rank"] == 5
    assert projection.options["use_dp_attention_reduce"] is False
