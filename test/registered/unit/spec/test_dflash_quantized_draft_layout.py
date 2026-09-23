"""A quantized DFlash draft is built from its quantization config by module
name and loaded by tensor name, and both have to see the same names. With an
empty layer prefix every attention and MLP projection was matched as a bare
``qkv_proj`` / ``o_proj`` / ..., so name-based ``ignore`` entries never
applied and the checkpoint's tensors for those layers were dropped without a
word; ``fc`` was a plain ``nn.Linear`` that could not take packed tensors at
all. These cases pin the layer names, the quantizable ``fc``, and the refusal
to drop a tensor whose module the draft does have."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import dflash
from sglang.srt.models.dflash import DFlashDecoderLayer, DFlashDraftModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

HIDDEN = 128  # the INT8 group size, so fc's packed weight needs no padding
TARGET_LAYER_IDS = [1, 3]
K = len(TARGET_LAYER_IDS)


def _draft_config(num_layers=2):
    return SimpleNamespace(
        architectures=["DFlash2DraftModel"],
        hidden_size=HIDDEN,
        num_hidden_layers=num_layers,
        rms_norm_eps=1e-6,
        dflash_config={"block_size": 4, "target_layer_ids": TARGET_LAYER_IDS},
    )


def _int8_config(ignore=()):
    """compressed-tensors pack-quantized INT8 g128 on every Linear, as
    llm-compressor writes it."""
    return CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
            "ignore": list(ignore),
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "group_size": 128,
                        "strategy": "group",
                        "symmetric": True,
                    },
                }
            },
        }
    )


class _RecordingAttention(nn.Module):
    """Stands in for DFlashAttention (whose rotary embedding needs a GPU
    build): records the prefix it was built with and carries the module shapes
    load_weights has to reason about -- a fused qkv_proj holding only a packed
    weight, and a rotary module with a parameter next to its inv_freq buffer."""

    prefixes = []

    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        type(self).prefixes.append(prefix)
        self.qkv_proj = nn.Module()
        self.qkv_proj.weight_packed = nn.Parameter(
            torch.zeros(4, 4, dtype=torch.int32), requires_grad=False
        )
        self.rotary_emb = nn.Module()
        self.rotary_emb.cos_coef = nn.Parameter(torch.zeros(2))
        self.rotary_emb.register_buffer("inv_freq", torch.zeros(2))


class _RecordingMLP(nn.Module):
    prefixes = []

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        type(self).prefixes.append(prefix)


class _Layer(DFlashDecoderLayer):
    attention_cls = _RecordingAttention


class _Draft(DFlashDraftModel):
    decoder_layer_cls = _Layer


@pytest.fixture(autouse=True)
def _cpu_build(monkeypatch):
    _RecordingAttention.prefixes.clear()
    _RecordingMLP.prefixes.clear()
    monkeypatch.setattr(dflash, "DFlashMLP", _RecordingMLP)
    # Scheme resolution asks the GPU for its compute capability.
    monkeypatch.setattr(
        CompressedTensorsConfig,
        "_check_scheme_supported",
        lambda self, min_capability, error=True: True,
    )


def _fc_tensors():
    """Dense fc tensors for a draft whose fc is unquantized."""
    return [("fc.weight", torch.randn(HIDDEN, K * HIDDEN))]


@pytest.mark.parametrize("model_prefix", ["", "draft"])
def test_every_layer_is_built_under_its_checkpoint_name(model_prefix):
    """Names reach the quantization config through this chain; an empty
    prefix at either link turns every projection into a bare name."""
    _Draft(_draft_config(), quant_config=None, prefix=model_prefix)
    root = f"{model_prefix}." if model_prefix else ""
    assert _RecordingAttention.prefixes == [
        f"{root}layers.0.self_attn",
        f"{root}layers.1.self_attn",
    ]
    assert _RecordingMLP.prefixes == [f"{root}layers.0.mlp", f"{root}layers.1.mlp"]


def test_shard_only_ignore_reaches_the_fused_projection():
    """An ignore list that names q_proj/k_proj/v_proj (how llm-compressor
    writes it) applies to the fused qkv_proj only through the mapping the
    loader takes from the model class."""
    ignore = [
        r"re:.*self_attn\.q_proj$",
        r"re:.*self_attn\.k_proj$",
        r"re:.*self_attn\.v_proj$",
    ]
    name = "layers.0.self_attn.qkv_proj"
    assert should_ignore_layer(
        name, ignore=ignore, fused_mapping=DFlashDraftModel.packed_modules_mapping
    )
    assert not should_ignore_layer(name, ignore=ignore)


def test_fc_is_quantized_by_a_config_that_targets_linear():
    model = _Draft(_draft_config(), quant_config=_int8_config())
    assert isinstance(model.fc, ReplicatedLinear)
    assert isinstance(model.fc.quant_method, CompressedTensorsLinearMethod)
    assert model.fc.weight_packed.shape == (HIDDEN, K * HIDDEN // 4)
    assert not hasattr(model.fc, "weight")


def test_fc_stays_dense_when_ignored_or_unquantized():
    ignored = _Draft(_draft_config(), quant_config=_int8_config(ignore=["re:^fc$"]))
    assert isinstance(ignored.fc.quant_method, UnquantizedLinearMethod)
    plain = _Draft(_draft_config(), quant_config=None)
    assert plain.fc.weight.shape == (HIDDEN, K * HIDDEN)


def test_packed_fc_tensors_load():
    model = _Draft(_draft_config(), quant_config=_int8_config())
    packed = torch.randint(
        -(2**31), 2**31 - 1, model.fc.weight_packed.shape, dtype=torch.int32
    )
    scale = torch.randn(HIDDEN, K, dtype=torch.bfloat16)

    model.load_weights(
        [
            ("fc.weight_packed", packed),
            ("fc.weight_scale", scale),
            ("fc.weight_shape", torch.tensor([HIDDEN, K * HIDDEN])),
        ]
    )

    torch.testing.assert_close(model.fc.weight_packed.data, packed)
    torch.testing.assert_close(
        model.fc.weight_scale.data, scale.to(model.fc.weight_scale.dtype)
    )


@pytest.mark.parametrize("name", ["fc.weight", "encoder.fc.weight"])
def test_dense_fc_loads_under_its_own_and_the_vendor_name(name):
    model = _Draft(_draft_config(), quant_config=None)
    weight = torch.randn(HIDDEN, K * HIDDEN)
    model.load_weights([(name, weight)])
    torch.testing.assert_close(model.fc.weight.data, weight)


def test_dense_fc_shape_is_still_checked():
    model = _Draft(_draft_config(), quant_config=None)
    with pytest.raises(ValueError, match="fc.weight shape mismatch"):
        model.load_weights([("fc.weight", torch.randn(HIDDEN, HIDDEN))])


def test_loading_refuses_a_tensor_the_module_stores_differently():
    """A tensor for a module the draft has, under a parameter name that module
    was not built with, must not be dropped: a BF16 q/k/v tensor against a
    fused qkv_proj built packed left the packed weight uninitialized."""
    model = _Draft(_draft_config(), quant_config=None)
    with pytest.raises(ValueError, match="layers.0.self_attn.qkv_proj"):
        model.load_weights(
            [("layers.0.self_attn.q_proj.weight", torch.randn(4, HIDDEN))]
        )
    # The export's "model." prefix resolves to the same module.
    with pytest.raises(ValueError, match="layers.0.self_attn.qkv_proj"):
        model.load_weights(
            [("model.layers.0.self_attn.k_proj.weight", torch.randn(4, HIDDEN))]
        )
    # A packed fc tensor for a draft whose fc is dense, under either name.
    for name in ("fc.weight_packed", "encoder.fc.weight_packed"):
        with pytest.raises(ValueError, match=rf"'{name}'.*\bfc\b"):
            model.load_weights([(name, torch.zeros(4, 4, dtype=torch.int32))])


def test_loading_refuses_decoder_layers_without_an_fc():
    """A checkpoint that fills the layers but never fc would leave fc over
    uninitialized memory; a projector-only load (Domino) stays allowed."""
    model = _Draft(_draft_config(), quant_config=None)
    with pytest.raises(ValueError, match="no fc tensor"):
        model.load_weights([("layers.0.input_layernorm.weight", torch.ones(HIDDEN))])
    model.load_weights(
        _fc_tensors() + [("layers.0.input_layernorm.weight", torch.ones(HIDDEN))]
    )


def test_loading_still_ignores_tensors_the_draft_has_no_parameter_for():
    """Tensors that have no parameter on the draft by design: rotary caches,
    KV-cache and activation scales, biases the layer was built without, and
    modules the draft does not have at all."""
    model = _Draft(_draft_config(), quant_config=None)
    model.load_weights(
        _fc_tensors()
        + [
            ("layers.0.self_attn.rotary_emb.inv_freq", torch.zeros(2)),
            ("layers.0.self_attn.rotary_emb.cos_cached", torch.zeros(2, 2)),
            ("layers.0.self_attn.rotary_emb.sin_cached", torch.zeros(2, 2)),
            ("layers.0.self_attn.k_proj.output_scale", torch.ones(1)),
            ("layers.0.self_attn.k_proj.k_scale", torch.ones(1)),
            ("layers.0.self_attn.v_proj.v_scale", torch.ones(1)),
            ("layers.0.self_attn.q_proj.input_scale", torch.ones(1)),
            ("layers.0.self_attn.q_proj.bias", torch.zeros(4)),
            ("lm_head.weight", torch.zeros(4, HIDDEN)),
            ("embed_tokens.weight", torch.zeros(4, HIDDEN)),
            ("layers.5.self_attn.q_proj.weight", torch.zeros(4, HIDDEN)),
        ]
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
