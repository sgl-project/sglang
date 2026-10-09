from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.linear import MergedColumnParallelLinear, ReplicatedLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import qwen4_exp
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpPLELayer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _EmbeddingStub(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ngram_size = 5


def _config():
    return SimpleNamespace(
        hidden_size=8,
        ple_embed_dim=6,
        ple_conv_kernel_size=3,
        hc_count=4,
        rms_norm_eps=1e-6,
        ple_offload_embedding=False,
    )


def _make_layer(monkeypatch, quant_config=None):
    monkeypatch.setattr(qwen4_exp, "Qwen4ExpNGramEmbedding", _EmbeddingStub)
    return Qwen4ExpPLELayer(
        _config(),
        quant_config=quant_config,
        prefix="model.layers.0.ple",
        layer_id=0,
    )


def _make_model(layer, *, start_layer=0, end_layer=1):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].ple = layer
    model.config = SimpleNamespace(
        num_experts=None,
        tie_word_embeddings=False,
        encoder_only=False,
        text_config=SimpleNamespace(split_ngram_parts=512),
    )
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.language_model_only = True
    model.start_layer = start_layer
    model.end_layer = end_layer
    model._load_qwen4_exp_ple_buffer = lambda *args: False
    return model


def test_unquantized_ple_uses_one_real_merged_replicated_projection(monkeypatch):
    layer = _make_layer(monkeypatch)

    assert isinstance(layer.key_value_proj, MergedColumnParallelLinear)
    assert layer.key_value_proj.parallel_group == "replicated"
    assert not hasattr(layer, "key_proj")
    assert not hasattr(layer, "value_proj")
    assert list(dict(layer.named_parameters()))[:1] == ["key_value_proj.weight"]

    embeddings = torch.randn(3, 6)
    with torch.no_grad():
        layer.key_value_proj.weight.copy_(
            torch.arange(40 * 6, dtype=torch.float32).reshape(40, 6) / 100
        )
    projected, _ = layer.key_value_proj(embeddings)
    key, value = projected.split([32, 8], dim=-1)
    torch.testing.assert_close(
        key, torch.nn.functional.linear(embeddings, layer.key_value_proj.weight[:32])
    )
    torch.testing.assert_close(
        value, torch.nn.functional.linear(embeddings, layer.key_value_proj.weight[32:])
    )


def test_quantized_ple_keeps_separate_projection_parameter_tree(monkeypatch):
    quant_config = SimpleNamespace(
        get_quant_method=lambda layer, prefix: UnquantizedLinearMethod()
    )
    layer = _make_layer(monkeypatch, quant_config=quant_config)

    assert isinstance(layer.key_proj, ReplicatedLinear)
    assert isinstance(layer.value_proj, ReplicatedLinear)
    assert not hasattr(layer, "key_value_proj")
    names = set(dict(layer.named_parameters()))
    assert "key_proj.weight" in names
    assert "value_proj.weight" in names


def test_unquantized_ple_checkpoint_key_and_value_map_into_merged_weight(monkeypatch):
    layer = _make_layer(monkeypatch)
    model = _make_model(layer)

    key = torch.arange(32 * 6, dtype=torch.float32).reshape(32, 6)
    value = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6) + 1000
    loaded = Qwen4ExpForConditionalGeneration.load_weights(
        model,
        [
            ("model.layers.0.ple.key_proj.weight", key),
            ("model.layers.0.ple.value_proj.weight", value),
        ],
    )

    torch.testing.assert_close(layer.key_value_proj.weight, torch.cat([key, value]))
    assert loaded == {"model.layers.0.ple.key_value_proj.weight"}


def test_unquantized_ple_checkpoint_mapping_is_source_order_independent(monkeypatch):
    layer = _make_layer(monkeypatch)
    model = _make_model(layer)
    key = torch.arange(32 * 6, dtype=torch.float32).reshape(32, 6)
    value = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6) + 1000

    loaded = Qwen4ExpForConditionalGeneration.load_weights(
        model,
        [
            ("model.layers.0.ple.value_proj.weight", value),
            ("model.layers.0.ple.key_proj.weight", key),
        ],
    )

    torch.testing.assert_close(layer.key_value_proj.weight, torch.cat([key, value]))
    assert loaded == {"model.layers.0.ple.key_value_proj.weight"}


def test_unquantized_ple_checkpoint_skips_nonowning_pp_stage(monkeypatch):
    layer = _make_layer(monkeypatch)
    model = _make_model(layer, start_layer=1, end_layer=2)
    before = layer.key_value_proj.weight.detach().clone()
    key = torch.randn(32, 6)
    value = torch.randn(8, 6)

    loaded = Qwen4ExpForConditionalGeneration.load_weights(
        model,
        [
            ("model.layers.0.ple.key_proj.weight", key),
            ("model.layers.0.ple.value_proj.weight", value),
        ],
    )

    assert loaded == set()
    torch.testing.assert_close(layer.key_value_proj.weight, before)
