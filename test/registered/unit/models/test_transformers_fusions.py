# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import copy

import pytest
import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
)

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.transformers.fusers import fuse_module, load_fused_weights
from sglang.srt.models.transformers.layers import (
    HFCompatibleGemmaRMSNorm,
    HFCompatibleMergedColumnParallelLinear,
    HFCompatibleQKVParallelLinear,
    HFCompatibleRMSNorm,
    replace_linear_class,
    replace_rms_norm_class,
)
from sglang.srt.models.utils import AutoWeightsLoader
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def single_rank():
    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0):
        torch.manual_seed(73)
        yield


def _config():
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
        attention_bias=True,
    )
    config._attn_implementation = "eager"
    return config


def _load(module, weights, result):
    wrapper = nn.Module()
    wrapper.model = module
    loaded = set()
    remaining = load_fused_weights(
        wrapper,
        ((f"model.{name}", value) for name, value in weights.items()),
        result.stacked_mapping,
        loaded,
    )
    loaded.update(AutoWeightsLoader(wrapper).load_weights(remaining))
    return loaded


@pytest.mark.parametrize("length", [1, 7])
def test_qwen3_attention_fusion_matches_original(length):
    original = Qwen3Attention(_config(), layer_idx=0).eval()
    fused = copy.deepcopy(original)
    result = fuse_module(fused, "model")
    assert result is not None
    assert isinstance(fused.qkv_proj, HFCompatibleQKVParallelLinear)
    assert not hasattr(fused, "q_proj")
    loaded = _load(fused, original.state_dict(), result)
    assert "model.qkv_proj.weight" in loaded
    assert "model.qkv_proj.bias" in loaded
    x = torch.randn(2, length, 32)
    positions = (torch.ones(2, length, 4), torch.zeros(2, length, 4))
    with torch.no_grad():
        expected = original(x, positions, None)[0]
        actual = fused(x, positions, None)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(1, 32), (5, 32), (2, 7, 32)])
def test_qwen3_mlp_fusion_matches_original(shape):
    original = Qwen3MLP(_config()).eval()
    fused = copy.deepcopy(original)
    result = fuse_module(fused, "model")
    assert result is not None
    assert isinstance(fused.gate_up_proj, HFCompatibleMergedColumnParallelLinear)
    assert isinstance(fused.act_fn, SiluAndMul)
    _load(fused, original.state_dict(), result)
    x = torch.randn(shape)
    with torch.no_grad():
        torch.testing.assert_close(fused(x), original(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rank", range(4))
def test_qkv_tensor_parallel_replicates_kv_heads(rank):
    original = Qwen3Attention(_config(), layer_idx=0).eval()
    fused = copy.deepcopy(original)
    with get_parallel().override(
        tp_size=4, tp_rank=rank, attn_tp_size=4, attn_tp_rank=rank
    ):
        result = fuse_module(
            fused, "model", tp_plan={r"model\.(q_proj|k_proj|v_proj)": "colwise"}
        )
        assert result is not None
        _load(fused, original.state_dict(), result)
        x = torch.randn(3, 32)
        actual = fused.qkv_proj(x).split(fused._sglang_qkv_sizes, dim=-1)
    expected = (
        original.q_proj(x).chunk(4, -1)[rank],
        original.k_proj(x).chunk(2, -1)[rank // 2],
        original.v_proj(x).chunk(2, -1)[rank // 2],
    )
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference)


@pytest.mark.parametrize("rank", range(2))
def test_gate_up_tensor_parallel_loads_independent_shards(rank):
    original = Qwen3MLP(_config()).eval()
    fused = copy.deepcopy(original)
    with get_parallel().override(tp_size=2, tp_rank=rank):
        result = fuse_module(
            fused, "model", tp_plan={r"model\.(gate_proj|up_proj)": "colwise"}
        )
        assert result is not None
        _load(fused, original.state_dict(), result)
        x = torch.randn(3, 32)
        actual = fused.act_fn(fused.gate_up_proj(x))
    expected = (
        torch.nn.functional.silu(original.gate_proj(x).chunk(2, -1)[rank])
        * original.up_proj(x).chunk(2, -1)[rank]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["qkv", "mlp"])
def test_fp8_fusion_loads_each_scale_into_its_native_shard(kind):
    module = (
        Qwen3Attention(_config(), layer_idx=0) if kind == "qkv" else Qwen3MLP(_config())
    )
    config = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="static")
    result = fuse_module(module, "model", quant_config=config)
    assert result is not None
    source_names = list(result.stacked_mapping)
    weights = [
        (f"{name}.{suffix}", torch.tensor(float(index + 1)))
        for index, name in enumerate(source_names)
        for suffix in ("weight_scale", "input_scale")
    ]
    wrapper = nn.Module()
    wrapper.model = module
    loaded = set()
    assert (
        list(load_fused_weights(wrapper, weights, result.stacked_mapping, loaded)) == []
    )
    merged = module.qkv_proj if kind == "qkv" else module.gate_up_proj
    expected = torch.arange(1, len(source_names) + 1, dtype=torch.float32)
    torch.testing.assert_close(merged.weight_scale, expected)
    torch.testing.assert_close(merged.input_scale, expected)


def test_mixed_precision_shards_keep_the_original_modules():
    module = Qwen3MLP(_config())
    original = copy.deepcopy(module)
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True, ignored_layers=["model.gate_proj"]
    )
    assert fuse_module(module, "model", quant_config=config) is None
    assert not hasattr(module, "gate_up_proj")
    assert config.packed_modules_mapping == {}
    x = torch.randn(3, 32)
    torch.testing.assert_close(module(x), original(x))


class DifferentInputs(Qwen3MLP):
    def forward(self, x):
        y = x + 1
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(y))


class ReusedGate(Qwen3MLP):
    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x) + self.gate_proj(x)
        )


class MutatedAttention(Qwen3Attention):
    def forward(self, hidden_states):
        q = self.q_proj(hidden_states)
        hidden_states.add_(1)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, k, v


class AliasedMutation(Qwen3Attention):
    def forward(self, hidden_states):
        alias = hidden_states.view_as(hidden_states)
        q = self.q_proj(hidden_states)
        alias.add_(1)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, k, v


class CrossAttention(Qwen3Attention):
    def forward(self, hidden_states, encoder_hidden_states):
        q = self.q_proj(hidden_states)
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)
        return q, k, v


@pytest.mark.parametrize(
    "model_type",
    [DifferentInputs, ReusedGate, MutatedAttention, AliasedMutation, CrossAttention],
)
def test_semantically_incompatible_patterns_are_preserved(model_type):
    module = (
        model_type(_config(), layer_idx=0)
        if issubclass(model_type, Qwen3Attention)
        else model_type(_config())
    )
    names = set(module.state_dict())
    assert fuse_module(module, "model") is None
    assert set(module.state_dict()) == names


def test_gated_mlp_fusion_requires_kernel_aligned_width():
    config = _config()
    config.intermediate_size = 60
    module = Qwen3MLP(config)
    assert fuse_module(module, "model") is None
    assert not hasattr(module, "gate_up_proj")


def test_parallel_fusion_requires_matching_plan():
    module = Qwen3MLP(_config())
    with get_parallel().override(tp_size=2):
        assert fuse_module(module, "model", tp_plan={r"model\..*": "replicate"}) is None


def test_tensor_linear_exposes_native_tuple_for_lora():
    original = nn.Linear(32, 64)
    module = replace_linear_class(original)
    module.weight.weight_loader(module.weight, original.weight)
    module.bias.weight_loader(module.bias, original.bias)
    x = torch.randn(1, 4, 32)
    assert module._hf_returns_tensor
    tensor = module(x)
    output, bias = module.forward_with_bias(x)
    assert isinstance(tensor, torch.Tensor)
    assert bias is None
    torch.testing.assert_close(tensor, output)
    torch.testing.assert_close(tensor, original(x))


class MultiplyThenCastRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = 1e-6

    def forward(self, x):
        value = x.float()
        variance = value.pow(2).mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.eps)
        return (value * self.weight).to(x.dtype)


class WeightlessRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.hidden_size = width
        self.eps = 1e-6

    def forward(self, x):
        value = x.float()
        return (
            value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)


class AdditionalScaleRMSNorm(Qwen3RMSNorm):
    def forward(self, x):
        return super().forward(x) * 2


@pytest.mark.parametrize(
    "model_type", [Qwen3RMSNorm, Gemma3RMSNorm, MultiplyThenCastRMSNorm]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_norm_preserves_weight_and_cast_semantics(model_type, dtype):
    original = model_type(16).to(dtype=dtype)
    with torch.no_grad():
        original.weight.normal_(mean=0.25, std=0.5)
    fused = replace_rms_norm_class(original, 1024).to(dtype=dtype)
    assert isinstance(fused, (HFCompatibleRMSNorm, HFCompatibleGemmaRMSNorm))
    loader = getattr(fused.weight, "weight_loader", None)
    if loader is None:
        with torch.no_grad():
            fused.weight.copy_(original.weight)
    else:
        loader(fused.weight, original.weight)
    x = torch.randn(2, 4, 16, dtype=dtype)
    with torch.no_grad():
        torch.testing.assert_close(fused(x), original(x), rtol=0, atol=0)


def test_weightless_norm_uses_own_head_width():
    original = WeightlessRMSNorm(16)
    fused = replace_rms_norm_class(original, 4096)
    assert isinstance(fused, HFCompatibleRMSNorm)
    assert fused.hidden_size == 16
    assert dict(fused.named_parameters()) == {}
    assert "weight" in dict(fused.named_buffers())
    x = torch.randn(1, 4, 8, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(fused(x), original(x), rtol=0, atol=0)


def test_weightless_norm_without_width_keeps_original():
    original = WeightlessRMSNorm(16)
    del original.hidden_size
    assert replace_rms_norm_class(original, 4096) is original


def test_norm_name_and_zero_initialization_do_not_define_semantics():
    original = Qwen3RMSNorm(16)
    with torch.no_grad():
        original.weight.zero_()
    assert isinstance(replace_rms_norm_class(original, 1024), HFCompatibleRMSNorm)
    unusual = AdditionalScaleRMSNorm(16)
    assert replace_rms_norm_class(unusual, 1024) is unusual


def test_fused_loader_preserves_explicit_unexpected_bias_policy():
    module = Qwen3MLP(_config())
    result = fuse_module(module, "model")
    wrapper = nn.Module()
    wrapper.model = module
    weights = [("model.gate_proj.bias", torch.ones(64))]
    with pytest.raises(ValueError, match="maps to missing"):
        list(load_fused_weights(wrapper, weights, result.stacked_mapping, set()))
    assert (
        list(
            load_fused_weights(
                wrapper,
                weights,
                result.stacked_mapping,
                set(),
                ignore_unexpected_suffixes=(".bias",),
            )
        )
        == []
    )


def test_fusion_preserves_existing_adapter_packing_contract():
    module = Qwen3MLP(_config())
    packing = {"gate_up_proj": ["w1", "w3"]}
    assert fuse_module(module, "model", packed_modules_mapping=packing) is None
    assert hasattr(module, "gate_proj")
    assert packing == {"gate_up_proj": ["w1", "w3"]}


@pytest.mark.parametrize("rank", range(4))
def test_unfused_kv_projection_replicates_whole_heads(rank):
    from sglang.srt.models.transformers.layers import get_attention_projection_shards

    original = Qwen3Attention(_config(), layer_idx=0)
    plan = {r"model\.(q_proj|k_proj|v_proj)": "colwise"}
    with get_parallel().override(
        tp_size=4, tp_rank=rank, attn_tp_size=4, attn_tp_rank=rank
    ):
        shards = get_attention_projection_shards(original, "model", plan)
        for name in ("k_proj", "v_proj"):
            projection = getattr(original, name)
            native = replace_linear_class(
                projection, "colwise", **shards[f"model.{name}"]
            )
            native.weight.weight_loader(native.weight, projection.weight)
            native.bias.weight_loader(native.bias, projection.bias)
            x = torch.randn(3, 32)
            torch.testing.assert_close(native(x), projection(x).chunk(2, -1)[rank // 2])
            assert native.output_size_per_partition == original.head_dim


@pytest.mark.parametrize("missing", ["q_proj", "k_proj", "v_proj"])
def test_fused_qkv_rejects_incomplete_checkpoint_weights(missing):
    original = Qwen3Attention(_config(), layer_idx=0)
    module = copy.deepcopy(original)
    result = fuse_module(module, "model")
    weights = {
        name: value
        for name, value in original.state_dict().items()
        if name != f"{missing}.weight"
    }
    with pytest.raises(
        ValueError, match="Incomplete fused checkpoint parameter.*weight"
    ):
        _load(module, weights, result)


@pytest.mark.parametrize("missing", ["gate_proj", "up_proj"])
def test_fused_glu_rejects_incomplete_checkpoint_weights(missing):
    original = Qwen3MLP(_config())
    module = copy.deepcopy(original)
    result = fuse_module(module, "model")
    weights = {
        name: value
        for name, value in original.state_dict().items()
        if name != f"{missing}.weight"
    }
    with pytest.raises(
        ValueError, match="Incomplete fused checkpoint parameter.*weight"
    ):
        _load(module, weights, result)


def test_fp8_fusion_rejects_partial_per_tensor_scales():
    module = Qwen3MLP(_config())
    result = fuse_module(module, "model", Fp8Config(is_checkpoint_fp8_serialized=True))
    wrapper = nn.Module()
    wrapper.model = module
    with pytest.raises(
        ValueError, match="Incomplete fused checkpoint parameter.*weight_scale"
    ):
        list(
            load_fused_weights(
                wrapper,
                [("model.gate_proj.weight_scale", torch.tensor(1.0))],
                result.stacked_mapping,
                set(),
            )
        )


def test_fused_loader_accepts_native_shared_parameter_contract():
    module = Qwen3MLP(_config())
    result = fuse_module(module, "model")
    merged = module.gate_up_proj
    merged.shared_scale = nn.Parameter(torch.empty(1), requires_grad=False)
    merged.shared_scale.weight_loader = merged.weight_loader
    wrapper = nn.Module()
    wrapper.model = module
    loaded = set()
    assert (
        list(
            load_fused_weights(
                wrapper,
                [("model.gate_proj.shared_scale", torch.tensor([2.0]))],
                result.stacked_mapping,
                loaded,
            )
        )
        == []
    )
    torch.testing.assert_close(merged.shared_scale, torch.tensor([2.0]))
    assert "model.gate_up_proj.shared_scale" in loaded


def test_explicit_partial_reload_preserves_initialized_kv_weights():
    original = Qwen3Attention(_config(), layer_idx=0)
    module = copy.deepcopy(original)
    result = fuse_module(module, "model")
    _load(module, original.state_dict(), result)
    before = module.qkv_proj.weight.detach().clone()
    wrapper = nn.Module()
    wrapper.model = module
    loaded = set()
    updated_query = torch.randn_like(original.q_proj.weight)
    assert (
        list(
            load_fused_weights(
                wrapper,
                [("model.q_proj.weight", updated_query)],
                result.stacked_mapping,
                loaded,
                require_complete=False,
            )
        )
        == []
    )
    torch.testing.assert_close(module.qkv_proj.weight[:32], updated_query)
    torch.testing.assert_close(module.qkv_proj.weight[32:], before[32:])
    assert "model.qkv_proj.weight" in loaded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
