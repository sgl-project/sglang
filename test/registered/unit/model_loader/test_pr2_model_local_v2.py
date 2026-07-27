"""CPU-only checks for model-local weight-loader-v2 special cases."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn


# Model imports transitively import optional kernel extension namespaces.  Stub
# them before importing any model so this module remains a lightweight CPU test.
sys.modules["sgl_kernel"] = MagicMock()
for _submodule in (
    "elementwise",
    "flash_attn",
    "flash_mla",
    "kvcacheio",
    "mamba",
    "quantization",
    "scalar_type",
    "sparse_flash_attn",
    "speculative",
    "utils",
):
    sys.modules[f"sgl_kernel.{_submodule}"] = MagicMock()

from sglang.srt.models.gpt_bigcode import GPTBigCodeForCausalLM  # noqa: E402
from sglang.srt.models.hunyuan import HunYuanMoEV1ForCausalLM  # noqa: E402
from sglang.srt.models.internlm2 import (  # noqa: E402
    InternLM2Attention,
    InternLM2ForCausalLM,
)
from sglang.srt.models.jet_nemotron import JetNemotronForCausalLM  # noqa: E402
from sglang.srt.models.laguna import LagunaForCausalLM  # noqa: E402
from sglang.srt.models import minimax_m3  # noqa: E402
from sglang.srt.models.minimax_m3 import MiniMaxM3SparseForCausalLM  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402


register_cpu_ci(est_time=15, suite="base-b-test-cpu")


def _bare(cls):
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    return obj


def _install_param(root, name, loader):
    module = root
    parts = name.split(".")
    for part in parts[:-1]:
        if part not in module._modules:
            module.add_module(part, nn.Module())
        module = module._modules[part]
    param = nn.Parameter(torch.zeros(1), requires_grad=False)
    param.weight_loader = loader
    module.register_parameter(parts[-1], param)
    return param


def _shard_recorder(calls):
    def load(_param, tensor, shard_id):
        calls.append((shard_id, tensor.clone()))

    return load


def _expert_recorder(calls):
    def load(_param, tensor, name, *, shard_id, expert_id):
        calls.append((name, shard_id, expert_id, tensor.clone()))

    return load


def test_internlm2_v2_reorders_packed_wqkv():
    model = _bare(InternLM2ForCausalLM)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    attention = _bare(InternLM2Attention)
    attention.config = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=8,
    )
    calls = []
    param = nn.Parameter(torch.zeros(1), requires_grad=False)
    param.weight_loader = _shard_recorder(calls)
    attention.register_parameter("wqkv", param)
    model.model.layers[0].attention = attention
    packed = torch.arange(16 * 3).reshape(16, 3)

    loaded = model._load_weights_v2(
        [("model.layers.0.attention.wqkv", packed)]
    )

    assert loaded == {"model.layers.0.attention.wqkv"}
    assert [shard for shard, _ in calls] == ["q", "k", "v"]
    viewed = packed.view(2, 4, 2, 3)
    expected = torch.split(viewed, [2, 1, 1], dim=1)
    for (_, actual), shard in zip(calls, expected):
        torch.testing.assert_close(actual, shard.reshape(-1, 3))


def test_jet_nemotron_v2_fans_six_shards_into_qkvabz():
    model = _bare(JetNemotronForCausalLM)
    calls = []
    target = "model.layers.0.self_attn.qkvabz_proj.weight"
    _install_param(model, target, _shard_recorder(calls))
    sources = ("q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj")
    weights = [
        (target.replace("qkvabz_proj", source), torch.tensor([index]))
        for index, source in enumerate(sources)
    ]

    loaded = model._load_weights_v2(weights)

    assert loaded == {target}
    assert [shard for shard, _ in calls] == list(range(6))
    assert [tensor.item() for _, tensor in calls] == list(range(6))


@pytest.mark.parametrize("scale_name", ["input_scale", "weight_scale"])
def test_gpt_bigcode_v2_fans_attention_scales_to_qkv(scale_name):
    model = _bare(GPTBigCodeForCausalLM)
    calls = []
    name = f"transformer.h.0.attn.c_attn.{scale_name}"
    _install_param(model, name, _shard_recorder(calls))
    tensor = torch.tensor([2.5])

    loaded = model._load_weights_v2(
        [
            ("lm_head.weight", torch.ones(1)),
            ("transformer.h.0.attn.bias", torch.ones(1)),
            (name, tensor),
        ]
    )

    assert loaded == {name}
    assert [shard for shard, _ in calls] == ["q", "k", "v"]
    assert all(value.item() == tensor.item() for _, value in calls)


def test_hunyuan_v2_splits_gate_up_and_packed_qkv():
    model = _bare(HunYuanMoEV1ForCausalLM)
    model.config = SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=1,
        tie_word_embeddings=False,
        use_cla=False,
    )
    model.head_dim = 2
    model.hidden_size = 4
    gate_calls, qkv_calls = [], []
    gate_target = "model.layers.0.mlp.gate_up_proj.weight"
    qkv_target = "model.layers.0.self_attn.qkv_proj.weight"
    _install_param(model, gate_target, _shard_recorder(gate_calls))
    _install_param(model, qkv_target, _shard_recorder(qkv_calls))
    gate_up = torch.arange(16).reshape(4, 4)
    packed_qkv = torch.arange(32).reshape(8, 4)

    loaded = model._load_weights_v2(
        [
            (gate_target.replace("gate_up_proj", "gate_and_up_proj"), gate_up),
            (qkv_target, packed_qkv),
        ]
    )

    assert loaded == {gate_target, qkv_target}
    assert [shard for shard, _ in gate_calls] == [1, 0]
    torch.testing.assert_close(gate_calls[0][1], gate_up[:2])
    torch.testing.assert_close(gate_calls[1][1], gate_up[2:])
    assert [shard for shard, _ in qkv_calls] == ["q", "k", "v"]
    reordered = model._split_qkv_weight(packed_qkv)
    for (_, actual), expected in zip(qkv_calls, reordered.split([4, 2, 2])):
        torch.testing.assert_close(actual, expected)


def _laguna_model():
    model = _bare(LagunaForCausalLM)
    model.config = SimpleNamespace(
        num_experts=1,
        mlp_layer_types=["sparse"],
        tie_word_embeddings=True,
    )
    model.model = nn.Module()
    model.model.start_layer = 0
    model.model.end_layer = 1
    return model


def test_laguna_v2_maps_shared_and_experts_with_pp_and_tied_skips():
    model = _laguna_model()
    shared_calls, expert_calls = [], []
    shared = "model.layers.0.mlp.shared_expert.gate_up_proj.weight"
    w13 = "model.layers.0.mlp.experts.w13_weight"
    w2 = "model.layers.0.mlp.experts.w2_weight"
    _install_param(model, shared, _shard_recorder(shared_calls))
    _install_param(model, w13, _expert_recorder(expert_calls))
    _install_param(model, w2, _expert_recorder(expert_calls))
    prefix = "model.layers.0.mlp.experts.0"
    weights = [
        (shared.replace("gate_up_proj", "gate_proj"), torch.tensor([10])),
        (f"{prefix}.gate_proj.weight", torch.tensor([1])),
        (f"{prefix}.down_proj.weight", torch.tensor([2])),
        (f"{prefix}.up_proj.weight", torch.tensor([3])),
        ("lm_head.weight", torch.tensor([99])),
        ("model.layers.1.mlp.experts.0.gate_proj.weight", torch.tensor([99])),
    ]

    loaded = model._load_weights_v2(weights)

    assert loaded == {shared, w13, w2}
    assert [shard for shard, _ in shared_calls] == [0]
    assert [(shard, expert) for _, shard, expert, _ in expert_calls] == [
        ("w1", 0),
        ("w2", 0),
        ("w3", 0),
    ]


def test_laguna_v2_rejects_incomplete_expert_checkpoint():
    model = _laguna_model()
    calls = []
    _install_param(
        model, "model.layers.0.mlp.experts.w13_weight", _expert_recorder(calls)
    )
    _install_param(
        model, "model.layers.0.mlp.experts.w2_weight", _expert_recorder(calls)
    )

    with pytest.raises(RuntimeError, match="1 routed-expert tensors were not loaded"):
        model._load_weights_v2(
            [
                (
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    torch.ones(1),
                ),
                (
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                    torch.ones(1),
                ),
            ]
        )


def test_minimax_m3_v2_shared_index_qkv_alias_and_postload_once(monkeypatch):
    model = _bare(MiniMaxM3SparseForCausalLM)
    model.config = SimpleNamespace(
        num_local_experts=1,
        sparse_attention_config={},
        num_mtp_modules=0,
    )
    model.num_fused_shared_experts = 1
    model.model = nn.Module()
    model.model.start_layer = 0
    model.model.end_layer = 1
    index_calls, expert_calls, scale_calls, post_calls = [], [], [], []
    index = "model.layers.0.self_attn.index_qkv_proj.weight"
    _install_param(model, index, _shard_recorder(index_calls))
    _install_param(
        model, "model.layers.0.mlp.experts.w13_weight", _expert_recorder(expert_calls)
    )
    _install_param(
        model, "model.layers.0.mlp.experts.w2_weight", _expert_recorder(expert_calls)
    )
    scale = "model.layers.0.self_attn.attn.k_scale"
    _install_param(model, scale, lambda _param, tensor: scale_calls.append(tensor))
    monkeypatch.setattr(
        minimax_m3,
        "build_minimax_fused_qkv_index",
        lambda loaded_model: post_calls.append(loaded_model),
    )
    shared = "model.layers.0.mlp.shared_experts"

    loaded = model._load_weights_v2(
        [
            (
                index.replace("index_qkv_proj", "index_q_proj"),
                torch.tensor([7]),
            ),
            (f"{shared}.gate_proj.weight", torch.tensor([1])),
            (f"{shared}.down_proj.weight", torch.tensor([2])),
            (f"{shared}.up_proj.weight", torch.tensor([3])),
            ("model.layers.0.self_attn.k_scale", torch.tensor([4.0])),
            (
                "model.layers.1.self_attn.index_q_proj.weight",
                torch.tensor([99]),
            ),
        ]
    )

    assert index in loaded
    assert scale in loaded
    assert [shard for shard, _ in index_calls] == ["q"]
    assert [(shard, expert) for _, shard, expert, _ in expert_calls] == [
        ("w1", 1),
        ("w2", 1),
        ("w3", 1),
    ]
    assert scale_calls[0].item() == 4.0
    assert post_calls == [model]
