"""Inkling2 adapter layout: target names, the six-slice qkvr, the unembed, and hidden dims."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.utils import get_normalized_target_modules
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Inkling 2 Mini attention geometry.
_D, _HEADS, _KV_HEADS, _HEAD_DIM, _D_REL, _RANK = 2304, 24, 8, 128, 16, 16
_MEMBER_ROWS = {
    "wq_du": _HEADS * _HEAD_DIM,
    "wk_dv": _KV_HEADS * _HEAD_DIM,
    "wv_dv": _KV_HEADS * _HEAD_DIM,
    "wr_du": _HEADS * _D_REL,
    "gate_attn": _HEADS * _HEAD_DIM,
    "attn_sink_proj": _HEADS,
}
# Tinker's Inkling2 schema adapts every attention projection but the sink.
_TRAINED = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn")
_TARGETS = [f"attn.{m}" for m in (*_TRAINED, "wo_ud")] + [
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "model.llm.unembed",
]
_LAYER = "model.llm.layers.{}.attn"


def _text_config(swa_heads=_HEADS, local_layer_ids=(0,)):
    return SimpleNamespace(
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        swa_num_attention_heads=swa_heads,
        swa_num_key_value_heads=_KV_HEADS,
        swa_head_dim=_HEAD_DIM,
        d_rel=_D_REL,
        local_layer_ids=list(local_layer_ids),
    )


def _adapter(
    architecture="Inkling2LoopForConditionalGeneration", num_layers=2, text=None
):
    text = text or _text_config()
    return LoRAAdapter(
        uid="inkling2",
        config=LoRAConfig.from_dict(
            dict(peft_type="LORA", r=_RANK, lora_alpha=32, target_modules=_TARGETS)
        ),
        base_hf_config=SimpleNamespace(
            num_hidden_layers=num_layers,
            architectures=[architecture],
            get_text_config=lambda: text,
        ),
        load_config=None,
        lora_backend=None,
    )


def _member_tensors(layer: int, members=_TRAINED, seed: int = 0):
    generator = torch.Generator().manual_seed(seed + layer)
    tensors = {}
    for member in members:
        prefix = f"{_LAYER.format(layer)}.{member}"
        tensors[f"{prefix}.lora_A.weight"] = torch.randn(_RANK, _D, generator=generator)
        tensors[f"{prefix}.lora_B.weight"] = torch.randn(
            _MEMBER_ROWS[member], _RANK, generator=generator
        )
    return tensors


def test_target_names_normalize_to_served_modules():
    assert get_normalized_target_modules(_TARGETS) == {
        "qkvr",
        "wo_ud",
        "gate_up_proj",
        "down_proj",
        "lm_head",
    }


def test_qkvr_stacks_six_slices_with_zero_sink():
    tensors = {**_member_tensors(0), **_member_tensors(1)}
    adapter = _adapter()
    adapter.initialize_weights_from_tensors(dict(tensors))

    for layer in (0, 1):
        weights = adapter.layers[layer].weights
        prefix = _LAYER.format(layer)
        assert set(weights) == {
            f"{prefix}.qkvr.lora_A.weight",
            f"{prefix}.qkvr.lora_B.weight",
        }
        lora_a = weights[f"{prefix}.qkvr.lora_A.weight"]
        lora_b = weights[f"{prefix}.qkvr.lora_B.weight"]
        assert lora_a.shape == (6 * _RANK, _D)
        assert lora_b.shape == (sum(_MEMBER_ROWS.values()), _RANK)
        a_slices = lora_a.split(_RANK)
        b_slices = lora_b.split(list(_MEMBER_ROWS.values()))
        for slot, member in enumerate(_MEMBER_ROWS):
            if member in _TRAINED:
                torch.testing.assert_close(
                    a_slices[slot], tensors[f"{prefix}.{member}.lora_A.weight"]
                )
                torch.testing.assert_close(
                    b_slices[slot], tensors[f"{prefix}.{member}.lora_B.weight"]
                )
            else:
                assert not a_slices[slot].any() and not b_slices[slot].any()


def test_local_layers_use_swa_geometry():
    """A local layer's members are sized by swa_* heads, not the full-attention ones."""
    swa_heads = 16
    tensors = _member_tensors(0)
    for member in ("wq_du", "wr_du", "gate_attn"):
        rows = _MEMBER_ROWS[member] // _HEADS * swa_heads
        tensors[f"{_LAYER.format(0)}.{member}.lora_B.weight"] = torch.zeros(rows, _RANK)
    adapter = _adapter(num_layers=1, text=_text_config(swa_heads=swa_heads))
    adapter.initialize_weights_from_tensors(tensors)
    lora_b = adapter.layers[0].weights[f"{_LAYER.format(0)}.qkvr.lora_B.weight"]
    q, kv, r = swa_heads * _HEAD_DIM, _KV_HEADS * _HEAD_DIM, swa_heads * _D_REL
    assert lora_b.shape == (q + 2 * kv + r + q + swa_heads, _RANK)


def test_qkvr_rejects_a_member_with_wrong_rows():
    tensors = _member_tensors(0)
    bad = f"{_LAYER.format(0)}.wk_dv.lora_B.weight"
    tensors[bad] = torch.zeros(_MEMBER_ROWS["wq_du"], _RANK)
    with pytest.raises(ValueError, match="wk_dv.lora_B has 3072 rows"):
        _adapter().initialize_weights_from_tensors(tensors)


def test_qkvr_rejects_a_member_with_one_factor():
    tensors = _member_tensors(0)
    tensors.pop(f"{_LAYER.format(0)}.wr_du.lora_B.weight")
    with pytest.raises(ValueError, match="only one factor"):
        _adapter().initialize_weights_from_tensors(tensors)


def test_unembed_loads_as_lm_head():
    vocab = 64
    tensors = {
        "model.llm.unembed.lora_A.weight": torch.ones(_RANK, _D),
        "model.llm.unembed.lora_B.weight": torch.ones(vocab, _RANK),
    }
    adapter = _adapter()
    adapter.initialize_weights_from_tensors(tensors)
    assert set(adapter.embedding_layers) == {
        "model.llm.lm_head.lora_A.weight",
        "model.llm.lm_head.lora_B.weight",
    }


def test_inkling1_keeps_its_four_slices():
    tensors = _member_tensors(0, members=("wq_du", "wk_dv", "wv_dv", "wr_du"))
    adapter = _adapter(architecture="InklingForConditionalGeneration", num_layers=1)
    adapter.initialize_weights_from_tensors(dict(tensors))
    lora_a = adapter.layers[0].weights[f"{_LAYER.format(0)}.qkvr.lora_A.weight"]
    assert lora_a.shape == (4 * _RANK, _D)


class _FakeLinear(nn.Module):
    def __init__(self, input_size, output_sizes):
        super().__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.output_size = sum(output_sizes)


def test_hidden_dims_match_mini():
    from sglang.srt.models.inkling2 import Inkling2ForConditionalGeneration

    attn = SimpleNamespace(
        # A trailing pad slice carries no adapter.
        qkvr=_FakeLinear(_D, [*_MEMBER_ROWS.values(), 8]),
        wo_ud=_FakeLinear(_HEADS * _HEAD_DIM, [_D]),
    )
    model = Inkling2ForConditionalGeneration.__new__(Inkling2ForConditionalGeneration)
    object.__setattr__(
        model, "llm", SimpleNamespace(layers=[SimpleNamespace(attn=attn)])
    )
    object.__setattr__(
        model,
        "text_config",
        SimpleNamespace(
            hidden_size=_D, intermediate_size=1152, padded_vocab_size=201088
        ),
    )
    assert model.get_hidden_dim("qkvr", 0) == (_D, 8600)
    assert model.get_hidden_dim("wo_ud", 0) == (3072, _D)
    assert model.get_hidden_dim("lm_head", 0) == (_D, 201088)
    assert model.get_hidden_dim("gate_up_proj", 0) == (_D, 2304)
    assert model.get_hidden_dim("down_proj", 0) == (1152, _D)
    assert model.get_stacked_multiply("qkvr") == 6
    assert model.get_stacked_multiply("gate_up_proj") == 2
    assert model.get_stacked_multiply("wo_ud") == 1


class _RecordingLoRAQKVR(BaseLayerWithLoRA):
    def __init__(self, width):
        nn.Module.__init__(self)
        self.width, self.rows = width, []

    def forward(self, x):
        self.rows.append(x.shape[0])
        return torch.zeros(x.shape[0], self.width), None


def test_lora_qkvr_projects_each_source_separately():
    """LoRA kernels walk the batch's T token segments, so stacking three sources into
    [3T, D] would leave the k and v rows without their adapter delta."""
    from sglang.srt.models.inkling_common.inkling2_attn import (
        AttentionSources,
        Inkling2Attention,
    )

    sizes = list(_MEMBER_ROWS.values())
    attn = Inkling2Attention.__new__(Inkling2Attention)
    nn.Module.__init__(attn)
    attn.split_sizes = sizes
    attn.qkvr = _RecordingLoRAQKVR(sum(sizes))
    tokens = 5
    sources = AttentionSources(*(torch.randn(tokens, _D) for _ in range(3)))
    attn._project_sources(sources)
    assert attn.qkvr.rows == [tokens, tokens, tokens]
