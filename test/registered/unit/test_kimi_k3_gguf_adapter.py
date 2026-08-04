import json
import sys
import types
from enum import IntEnum
from pathlib import Path

import numpy as np
import pytest
import torch
from sglang.srt.model_loader.kimi_k3_gguf_adapter import (
    KimiK3GGUFAdapter,
    _dense_tensor,
    _find_arch,
    is_kimi_k3_gguf_config,
)


class _Type(IntEnum):
    F32 = 0
    BF16 = 1
    IQ2_XXS = 2


class _Field:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


class _Tensor:
    def __init__(self, name, data, tensor_type=_Type.F32):
        self.name = name
        self.data = np.asarray(data)
        self.tensor_type = tensor_type


class _NameMap:
    def get_name(self, name, try_suffixes=()):
        if name == "model.embed_tokens.weight":
            return "token_embd.weight"
        return None


def _source_names():
    names = ["language_model.model.embed_tokens.weight"]
    layer_tails = [
        "self_attn.A_log",
        "self_attn.dt_bias",
        "self_attn.q_conv1d.weight",
        "self_attn.k_conv1d.weight",
        "self_attn.v_conv1d.weight",
        "self_attn.g_proj.weight",
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
        "block_sparse_moe.routed_expert_down_proj.weight",
        "block_sparse_moe.routed_expert_up_proj.weight",
        "block_sparse_moe.routed_expert_norm.weight",
        "block_sparse_moe.gate.e_score_correction_bias",
    ]
    names.extend(f"language_model.model.layers.0.{tail}" for tail in layer_tails)
    names.extend(
        [
            "language_model.model.output_attn_res_norm.weight",
            "language_model.model.output_attn_res_proj.weight",
        ]
    )
    for expert in range(2):
        for projection in ("w1", "w2", "w3"):
            for suffix in ("packed", "scale"):
                names.append(
                    "language_model.model.layers.0.block_sparse_moe.experts."
                    f"{expert}.{projection}.weight_{suffix}"
                )
    return names


def _tensors():
    dense = {
        "token_embd.weight": np.ones((4, 4), dtype=np.float32),
        "blk.0.ssm_a": -np.exp(np.array([0.25, 0.5], dtype=np.float32)),
        "blk.0.ssm_dt.bias": np.ones(2, dtype=np.float32),
        "blk.0.ssm_conv1d_q.weight": np.ones((1, 2, 1, 2), dtype=np.float32),
        "blk.0.ssm_conv1d_k.weight": np.ones((1, 2, 1, 2), dtype=np.float32),
        "blk.0.ssm_conv1d_v.weight": np.ones((1, 2, 1, 2), dtype=np.float32),
        "blk.0.ssm_g.weight": np.ones((2, 4), dtype=np.float32),
        "blk.0.attn_res_score.weight": np.arange(4, dtype=np.float32),
        "blk.0.ffn_res_score.weight": np.arange(4, dtype=np.float32),
        "output_res_score.weight": np.arange(4, dtype=np.float32),
        "blk.0.ffn_routed_down.weight": np.ones((2, 4), dtype=np.float32),
        "blk.0.ffn_routed_up.weight": np.ones((4, 2), dtype=np.float32),
        "blk.0.ffn_routed_norm.weight": np.ones(2, dtype=np.float32),
        "blk.0.exp_probs_b.bias": np.ones(2, dtype=np.float32),
    }
    tensors = [_Tensor(name, value) for name, value in dense.items()]
    for role in ("gate", "down", "up"):
        tensors.append(
            _Tensor(
                f"blk.0.ffn_{role}_exps.weight",
                np.arange(32, dtype=np.uint8).reshape(2, 4, 4),
                _Type.IQ2_XXS,
            )
        )
    return tensors


def _model_config(path: Path):
    text = types.SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=4,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        linear_attn_config={"full_attn_layers": []},
        num_experts=2,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        num_key_value_heads=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        v_head_dim=2,
    )
    root = types.SimpleNamespace(
        model_type="kimi_k3",
        architectures=["KimiK3ForConditionalGeneration"],
        language_only=True,
    )
    return types.SimpleNamespace(
        model_path=str(path), hf_config=root, hf_text_config=text
    )


def _install_fake_gguf(monkeypatch, tensors):
    fields = {
        "general.architecture": _Field("kimi-k3"),
        "kimi-k3.block_count": _Field(1),
        "kimi-k3.activation.situ_beta": _Field(4.0),
        "kimi-k3.activation.situ_linear_beta": _Field(25.0),
    }

    class Reader:
        def __init__(self, _path):
            self.fields = fields
            self.tensors = tensors

    module = types.SimpleNamespace(
        MODEL_ARCH_NAMES={1: "kimi-linear"},
        GGUFReader=Reader,
        get_tensor_name_map=lambda _arch, _layers: _NameMap(),
    )
    monkeypatch.setitem(sys.modules, "gguf", module)
    return module


def _make_adapter(tmp_path, monkeypatch, tensors=None):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"synthetic")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {name: "model.safetensors" for name in _source_names()}}
        )
    )
    _install_fake_gguf(monkeypatch, tensors or _tensors())
    return KimiK3GGUFAdapter(str(gguf_path), _model_config(gguf_path))


def test_architecture_falls_back_to_complete_kimi_linear_base_map():
    module = types.SimpleNamespace(MODEL_ARCH_NAMES={7: "kimi-linear"})
    assert _find_arch(module) == 7


def test_adapter_requires_language_only(tmp_path, monkeypatch):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"synthetic")
    config = _model_config(gguf_path)
    config.hf_config.language_only = False
    _install_fake_gguf(monkeypatch, _tensors())
    with pytest.raises(ValueError, match="--language-only"):
        KimiK3GGUFAdapter(str(gguf_path), config)


def test_adapter_rejects_unplanned_text_tensor(tmp_path, monkeypatch):
    tensors = _tensors() + [_Tensor("blk.0.silent_corruption", [1.0])]
    with pytest.raises(ValueError, match="extra=.*silent_corruption"):
        _make_adapter(tmp_path, monkeypatch, tensors)


def test_iterator_orders_qtypes_and_inverts_k3_transforms(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch)
    rows = list(adapter.weights_iterator())

    # Three projections x two experts, all before any packed tensor.
    assert all(name.endswith("qweight_type") for name, _ in rows[:6])
    first_qweight = next(
        i for i, (name, _) in enumerate(rows) if name.endswith("qweight")
    )
    assert first_qweight > 6

    by_name = {name: value for name, value in rows if not name.endswith("qweight_type")}
    assert torch.allclose(
        by_name["language_model.model.layers.0.self_attn.A_log"],
        torch.tensor([0.25, 0.5]),
    )
    assert by_name["language_model.model.layers.0.self_attn.q_conv1d.weight"].shape == (
        2,
        1,
        2,
    )
    assert "language_model.model.layers.0.self_attention_res_score_gguf" in by_name
    assert (
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.qweight" in by_name
    )
    assert (
        "language_model.model.layers.0.block_sparse_moe.experts.0.w2.qweight" in by_name
    )
    assert (
        "language_model.model.layers.0.block_sparse_moe.experts.0.w3.qweight" in by_name
    )


def test_kimi_config_detection_is_exact(tmp_path):
    path = tmp_path / "model.gguf"
    config = _model_config(path)
    assert is_kimi_k3_gguf_config(config)
    config.hf_config.model_type = "kimi_linear"
    assert not is_kimi_k3_gguf_config(config)


def test_bf16_reader_bytes_are_reinterpreted_without_dequantization():
    expected = torch.tensor([[1.0, -2.5]], dtype=torch.bfloat16)
    raw = expected.view(torch.uint8).numpy()
    actual = _dense_tensor(_Tensor("protected.weight", raw, _Type.BF16))
    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
