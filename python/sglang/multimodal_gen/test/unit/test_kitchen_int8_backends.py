"""Unit tests for kitchen_int8's backend selection and its sgl-kernel backend."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import wrap_with_lora_layer
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_sgl_kernel import (
    ConvRotInt8SglKernelLinearMethod,
    apply_convrot_int8_gelu_input,
    apply_convrot_int8_shared_input,
    apply_convrot_int8_shared_input_out,
    convrot_int8_fuses_gelu_input,
    convrot_int8_shares_input,
    sgl_kernel_convrot_available,
)
from sglang.multimodal_gen.runtime.layers.quantization.kitchen_int8 import (
    KitchenInt8LinearMethod,
)

_CONFIG_MODULE = (
    "sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config"
)
_LOAD_SGL_KERNEL = (
    "sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_sgl_kernel."
    "_load_sgl_kernel"
)
_LOAD_COMFY_KITCHEN = (
    "sglang.multimodal_gen.runtime.layers.quantization.kitchen_int8._load_comfy_kitchen"
)

QWEN_IMAGE_IGNORED_LAYERS = ["img_mod", "txt_mod", "txt_mlp.net.2"]


requires_kernel = pytest.mark.skipif(
    not sgl_kernel_convrot_available(),
    reason="needs a GPU in sgl-kernel's convrot table and a build with the convrot ops",
)


def _sgl_config(**kwargs) -> KitchenInt8Config:
    return KitchenInt8Config(backend="sgl_kernel", **kwargs)


def _method_for(config, prefix, input_size=3072):
    layer = LinearBase(input_size=input_size, output_size=32)
    return config.get_quant_method(layer, prefix)


@patch(_LOAD_SGL_KERNEL)
def test_ignored_layer_patterns_select_bf16_on_module_path_boundaries(_load):
    """A pattern must match its own module path only: `txt_mlp.net.2` keeps the
    text FFN down-projection in BF16 without also catching `img_mlp.net.2` or
    the text FFN up-projection, and `img_mod` must not catch `img_mlp`."""
    config = _sgl_config(ignored_layers=QWEN_IMAGE_IGNORED_LAYERS)

    kept_bf16 = [
        "transformer_blocks.0.img_mod.1",
        "transformer_blocks.3.txt_mod.1",
        "transformer_blocks.7.txt_mlp.net.2",
    ]
    quantized = [
        "transformer_blocks.0.img_mlp.net.2",
        "transformer_blocks.7.txt_mlp.net.0.proj",
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.add_k_proj",
    ]
    for prefix in kept_bf16:
        assert isinstance(_method_for(config, prefix), UnquantizedLinearMethod)
    for prefix in quantized:
        assert isinstance(_method_for(config, prefix), ConvRotInt8SglKernelLinearMethod)
    assert config.skipped == kept_bf16
    assert config.selected == quantized


@patch(_LOAD_SGL_KERNEL)
def test_input_dim_not_divisible_by_group_stays_bf16(_load):
    config = _sgl_config()
    method = _method_for(config, "blocks.0.mod", input_size=2688)
    assert isinstance(method, UnquantizedLinearMethod)
    assert config.skipped == ["blocks.0.mod(in=2688)"]
    assert config.selected == []
    assert not config.supports_input_partition("blocks.0.attn.to_out", 1536 + 128)
    assert config.supports_input_partition("blocks.0.attn.to_out", 1536)


@patch(_LOAD_SGL_KERNEL)
def test_shared_input_and_gelu_helpers_refuse_deferred_bias_and_lora(_load):
    """The helpers stand in for `layer(x)`; a layer that returns its bias
    separately would silently lose it, and a LoRA-wrapped projection is not a
    `LinearBase` at all, so eligibility must say no to both instead of
    crashing on the wrapper."""
    config = _sgl_config()
    plain = ReplicatedLinear(256, 8, quant_config=config, prefix="q")
    deferred = ReplicatedLinear(
        256, 8, skip_bias_add=True, quant_config=config, prefix="k"
    )
    bf16 = ReplicatedLinear(256, 8, quant_config=None, prefix="v")
    wrapped = wrap_with_lora_layer(plain)

    assert convrot_int8_shares_input([plain, plain])
    assert not convrot_int8_shares_input([plain, deferred])
    assert not convrot_int8_shares_input([plain, bf16])
    assert not convrot_int8_shares_input([wrapped, plain])
    assert not convrot_int8_shares_input([])
    assert convrot_int8_fuses_gelu_input(plain)
    assert not convrot_int8_fuses_gelu_input(deferred)
    assert not convrot_int8_fuses_gelu_input(bf16)
    assert not convrot_int8_fuses_gelu_input(wrapped)


def _quantized_layer(
    config, in_features, out_features, prefix, seed, dtype=torch.bfloat16
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    layer = ReplicatedLinear(
        in_features,
        out_features,
        params_dtype=dtype,
        quant_config=config,
        prefix=prefix,
    ).cuda()
    with torch.no_grad():
        layer.weight.copy_(
            torch.randn(layer.weight.shape, device="cuda", generator=gen) * 0.02
        )
        layer.bias.copy_(torch.randn(layer.bias.shape, device="cuda", generator=gen))
    layer.bias.requires_grad_(False)
    reference_weight = layer.weight.detach().clone()
    layer.quant_method.process_weights_after_loading(layer)
    return layer, reference_weight


@requires_kernel
def test_quantized_layer_tracks_bf16_reference():
    config = _sgl_config()
    layer, weight = _quantized_layer(config, 3072, 3072, "attn.to_q", seed=0)
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == (3072,)

    x = torch.randn(2, 1024, 3072, device="cuda", dtype=torch.bfloat16)
    out, out_bias = layer(x)
    assert out_bias is None
    ref = F.linear(x, weight, layer.bias)
    err = torch.linalg.vector_norm(out.float() - ref.float())
    rel_l2 = (err / torch.linalg.vector_norm(ref.float())).item()
    # Arbitrary loose bound: a wrong scale or rotation lands far above it,
    # W8A8 group-256 quantization noise on Gaussian data far below.
    assert rel_l2 < 2e-2, rel_l2


@requires_kernel
def test_shared_input_helper_is_bitwise_three_layer_calls():
    config = _sgl_config()
    layers = [
        _quantized_layer(config, 3072, 3072, f"attn.to_{n}", seed=i)[0]
        for i, n in enumerate("qkv")
    ]
    x = torch.randn(1, 4096, 3072, device="cuda", dtype=torch.bfloat16)

    assert convrot_int8_shares_input(layers)
    shared = apply_convrot_int8_shared_input(x=x, layers=layers)
    for out, layer in zip(shared, layers, strict=True):
        assert torch.equal(out, layer(x)[0])


@requires_kernel
def test_shared_input_out_helper_is_bitwise_into_joint_buffer_slices():
    """The joint text-image Q/K/V path writes each projection into a row slice
    of one larger buffer; the slice write must equal the allocating helper."""
    config = _sgl_config()
    layers = [
        _quantized_layer(config, 3072, 3072, f"attn.to_{n}", seed=10 + i)[0]
        for i, n in enumerate("qkv")
    ]
    seq_len_txt, seq_len_img = 20, 4096
    x = torch.randn(1, seq_len_img, 3072, device="cuda", dtype=torch.bfloat16)
    bufs = [
        torch.zeros(
            1, seq_len_txt + seq_len_img, 3072, device="cuda", dtype=torch.bfloat16
        )
        for _ in layers
    ]

    apply_convrot_int8_shared_input_out(
        x=x, layers=layers, outs=[buf[:, seq_len_txt:] for buf in bufs]
    )
    for out, buf, layer in zip(
        apply_convrot_int8_shared_input(x=x, layers=layers), bufs, layers, strict=True
    ):
        assert torch.equal(buf[:, seq_len_txt:], out)
        assert torch.equal(buf[:, seq_len_txt:], layer(x)[0])
        assert not buf[:, :seq_len_txt].any()


@requires_kernel
def test_gelu_input_helper_is_bitwise_eager_gelu_then_layer():
    config = _sgl_config()
    down, _ = _quantized_layer(config, 12288, 3072, "img_mlp.net.2", seed=3)
    up = torch.randn(1, 2048, 12288, device="cuda", dtype=torch.bfloat16)

    assert convrot_int8_fuses_gelu_input(down)
    fused = apply_convrot_int8_gelu_input(layer=down, x=up)
    eager, _ = down(F.gelu(up, approximate="tanh"))
    assert torch.equal(fused, eager)


class _FakeTpGroup:
    pass


def _with_tp(monkeypatch, tp_size):
    import sglang.multimodal_gen.runtime.layers.linear as linear

    monkeypatch.setattr(linear, "get_tp_group", lambda: _FakeTpGroup())
    monkeypatch.setattr(linear, "get_group_rank", lambda group: 0)
    monkeypatch.setattr(linear, "get_group_size", lambda group: tp_size)


@patch(_LOAD_SGL_KERNEL)
@pytest.mark.parametrize("tp_size,input_size", [(4, 13824), (8, 3072)])
def test_row_parallel_shard_not_divisible_by_group_is_refused(
    _load, monkeypatch, tp_size, input_size
):
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear

    _with_tp(monkeypatch, tp_size)
    config = _sgl_config()
    # kitchen_int8 semantics: the unsharded input divides, so the layer is
    # selected, and a shard that does not divide is refused at construction.
    with pytest.raises(ValueError, match="divisible by group_size"):
        RowParallelLinear(
            input_size,
            5120,
            bias=True,
            quant_config=config,
            prefix="blocks.0.ffn.fc_out",
        )


@patch(_LOAD_SGL_KERNEL)
def test_row_parallel_shard_divisible_by_group_is_quantized(_load, monkeypatch):
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear

    _with_tp(monkeypatch, 4)
    config = _sgl_config()
    layer = RowParallelLinear(
        12288, 3072, bias=True, quant_config=config, prefix="blocks.0.ffn.net.2"
    )
    assert isinstance(layer.quant_method, ConvRotInt8SglKernelLinearMethod)
    assert config.selected == ["blocks.0.ffn.net.2"]


@patch(_LOAD_SGL_KERNEL)
def test_output_width_not_multiple_of_8_stays_bf16(_load):
    config = _sgl_config()
    layer = LinearBase(input_size=4096, output_size=4)
    method = config.get_quant_method(layer, "blocks.0.attn.to_gate_logits")
    assert isinstance(method, UnquantizedLinearMethod)
    assert config.skipped == ["blocks.0.attn.to_gate_logits(out=4)"]


@patch(_LOAD_SGL_KERNEL)
def test_column_parallel_shard_output_not_multiple_of_8_stays_bf16(_load, monkeypatch):
    from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear

    _with_tp(monkeypatch, 8)
    config = _sgl_config()
    # 32 output rows split eight ways leave 4 per rank; the epilogue stores 8 at
    # a time, so the layer stays in BF16 instead of failing in the first forward.
    layer = ColumnParallelLinear(
        4096, 32, bias=True, quant_config=config, prefix="blocks.0.attn.to_gate_logits"
    )
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    assert config.skipped == ["blocks.0.attn.to_gate_logits(out=4)"]


@requires_kernel
def test_fp16_activations_are_cast_and_returned_in_fp16():
    config = _sgl_config()
    layer, weight = _quantized_layer(config, 3072, 3072, "attn.to_q", seed=3)
    x = torch.randn(2, 64, 3072, device="cuda", dtype=torch.float16)
    out, _ = layer(x)
    assert out.dtype == torch.float16
    ref = F.linear(x.to(torch.bfloat16), weight, layer.bias).to(torch.float16)
    err = torch.linalg.vector_norm(out.float() - ref.float())
    rel_l2 = (err / torch.linalg.vector_norm(ref.float())).item()
    assert rel_l2 < 2e-2, rel_l2


@requires_kernel
def test_fp16_parameters_are_cast_for_the_kernel_in_every_helper():
    """FP16 parameters (--dit-precision fp16) must run through the plain apply
    and every fused helper; the out= helper must refuse an FP16 input."""
    config = _sgl_config()
    layers = []
    weights = []
    for i, n in enumerate("qkv"):
        layer, weight = _quantized_layer(
            config, 3072, 3072, f"attn.to_{n}", seed=20 + i, dtype=torch.float16
        )
        layers.append(layer)
        weights.append(weight)
    assert layers[0].weight.dtype == torch.int8
    assert layers[0].bias.dtype == torch.bfloat16
    x = torch.randn(1, 128, 3072, device="cuda", dtype=torch.float16)

    for layer, weight in zip(layers, weights, strict=True):
        out, _ = layer(x)
        assert out.dtype == torch.float16
        ref = F.linear(x.float(), weight.float(), layer.bias.float())
        err = torch.linalg.vector_norm(out.float() - ref)
        rel_l2 = (err / torch.linalg.vector_norm(ref)).item()
        assert rel_l2 < 2e-2, rel_l2

    shared = apply_convrot_int8_shared_input(x=x, layers=layers)
    for out, layer in zip(shared, layers, strict=True):
        assert out.dtype == torch.float16
        assert torch.equal(out, layer(x)[0])

    down, _ = _quantized_layer(
        config, 12288, 3072, "img_mlp.net.2", seed=23, dtype=torch.float16
    )
    up = torch.randn(1, 128, 12288, device="cuda", dtype=torch.float16)
    fused = apply_convrot_int8_gelu_input(layer=down, x=up)
    assert fused.dtype == torch.float16
    # The GELU runs on the BF16-cast input inside the kernel, so the eager
    # reference is the layer on GELU of that cast, not GELU in FP16.
    eager, _ = down(F.gelu(up.to(torch.bfloat16), approximate="tanh"))
    assert torch.equal(fused, eager.to(torch.float16))

    outs = [
        torch.empty(1, 128, 3072, device="cuda", dtype=torch.bfloat16) for _ in layers
    ]
    with pytest.raises(ValueError, match="must be BF16"):
        apply_convrot_int8_shared_input_out(x=x, layers=layers, outs=outs)
    x_bf16 = x.to(torch.bfloat16)
    apply_convrot_int8_shared_input_out(x=x_bf16, layers=layers, outs=outs)
    for out, layer in zip(outs, layers, strict=True):
        assert torch.equal(out, layer(x_bf16)[0].to(torch.bfloat16))


@requires_kernel
def test_lora_merge_mode_is_redirected_to_dynamic_on_int8_base():
    from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline

    config = _sgl_config()
    layer, weight = _quantized_layer(
        config, 3072, 3072, "transformer_blocks.0.attn.to_q", seed=4
    )
    lora = wrap_with_lora_layer(layer, lora_rank=16, lora_alpha=16, snapshot_base=True)
    layers = {"transformer_blocks.0.attn.to_q": lora}
    # LoRAPipeline is abstract; the decision only needs its two static helpers.
    stand_in = SimpleNamespace(
        _has_quantized_base_weights=LoRAPipeline._has_quantized_base_weights,
        _uses_dtensor_weights=LoRAPipeline._uses_dtensor_weights,
    )

    def decide(module_name, layers, mode):
        return LoRAPipeline._should_merge_lora_for_layers(
            stand_in, module_name, layers, mode
        )

    assert decide("transformer_blocks.0.attn.to_q", layers, "auto") is False
    with pytest.raises(ValueError, match="lora-merge-mode dynamic"):
        decide("transformer_blocks.0.attn.to_q", layers, "merge")

    gen = torch.Generator(device="cuda").manual_seed(4)
    lora_a = (torch.randn(16, 3072, device="cuda", generator=gen) * 0.05).to(
        torch.bfloat16
    )
    lora_b = (torch.randn(3072, 16, device="cuda", generator=gen) * 0.05).to(
        torch.bfloat16
    )
    lora.set_lora_weights(
        lora_a, lora_b, lora_path="test", strength=1.0, merge_weights=False
    )
    x = torch.randn(8, 3072, device="cuda", dtype=torch.bfloat16, generator=gen)
    out, _ = lora(x)
    delta = (lora_b.float() @ lora_a.float()).to(torch.bfloat16)
    ref = F.linear(x, weight + delta, layer.bias)
    err = torch.linalg.vector_norm(out.float() - ref.float())
    rel_l2 = (err / torch.linalg.vector_norm(ref.float())).item()
    assert rel_l2 < 2e-2, rel_l2


_SGL_AVAILABLE = f"{_CONFIG_MODULE}._sgl_kernel_available"
_COMFY_AVAILABLE = f"{_CONFIG_MODULE}._comfy_kitchen_available"


@patch(_LOAD_COMFY_KITCHEN)
@patch(_LOAD_SGL_KERNEL)
def test_auto_backend_prefers_sgl_kernel_and_falls_back_to_comfy(_sgl, _comfy):
    """``auto`` must pick sgl-kernel where its ops run and comfy_kitchen
    otherwise; the choice is per config, not per import."""
    with (
        patch(_SGL_AVAILABLE, return_value=True),
        patch(_COMFY_AVAILABLE, return_value=True),
    ):
        config = KitchenInt8Config()
        assert config.resolve_backend() == "sgl_kernel"
        method = config.get_quant_method(LinearBase(3072, 3072), "blocks.0.attn.to_q")
        assert isinstance(method, ConvRotInt8SglKernelLinearMethod)
        assert config.selected_by_backend["sgl_kernel"] == ["blocks.0.attn.to_q"]
    with (
        patch(_SGL_AVAILABLE, return_value=False),
        patch(_COMFY_AVAILABLE, return_value=True),
    ):
        config = KitchenInt8Config()
        assert config.resolve_backend() == "comfy_kitchen"
        method = config.get_quant_method(LinearBase(3072, 3072), "blocks.0.attn.to_q")
        assert isinstance(method, KitchenInt8LinearMethod)
        assert config.selected_by_backend["comfy_kitchen"] == ["blocks.0.attn.to_q"]


@patch(_LOAD_COMFY_KITCHEN)
@patch(_LOAD_SGL_KERNEL)
def test_auto_backend_serves_narrow_outputs_with_comfy_or_leaves_bf16(_sgl, _comfy):
    """The sgl-kernel epilogue stores 8 outputs at a time; a narrower layer goes
    to comfy_kitchen when it is installed and stays BF16 otherwise."""
    with patch(_SGL_AVAILABLE, return_value=True):
        with patch(_COMFY_AVAILABLE, return_value=True):
            config = KitchenInt8Config()
            method = config.get_quant_method(LinearBase(4096, 4), "blocks.0.gate")
            assert isinstance(method, KitchenInt8LinearMethod)
            assert config.selected_by_backend["comfy_kitchen"] == ["blocks.0.gate"]
        with patch(_COMFY_AVAILABLE, return_value=False):
            config = KitchenInt8Config()
            method = config.get_quant_method(LinearBase(4096, 4), "blocks.0.gate")
            assert isinstance(method, UnquantizedLinearMethod)
            assert config.skipped == ["blocks.0.gate(out=4)"]


@patch(_LOAD_COMFY_KITCHEN)
@patch(_LOAD_SGL_KERNEL)
def test_serialized_layers_pick_the_backend_per_marker_group_size(_sgl, _comfy):
    """A Comfy checkpoint may mix group sizes; 16 exists only in comfy_kitchen
    while 256 runs on the sgl ops with the stored [N, 1] scale."""
    markers = {
        "visual.proj": {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 16,
        },
        "blocks.0.fc1": {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        },
    }
    with (
        patch(_SGL_AVAILABLE, return_value=True),
        patch(_COMFY_AVAILABLE, return_value=True),
    ):
        config = KitchenInt8Config(layer_markers=markers)
        small = ReplicatedLinear(
            64, 16, bias=False, quant_config=config, prefix="visual.proj"
        )
        fc1 = ReplicatedLinear(
            256, 16, bias=False, quant_config=config, prefix="blocks.0.fc1"
        )
    assert isinstance(small.quant_method, KitchenInt8LinearMethod)
    assert isinstance(fc1.quant_method, ConvRotInt8SglKernelLinearMethod)
    assert fc1.quant_method.is_checkpoint_serialized
    assert fc1.weight.dtype == torch.int8 and fc1.weight.shape == (16, 256)
    assert fc1.weight_scale.dtype == torch.float32
    assert fc1.weight_scale.shape == (16, 1)
    assert config.selected == ["visual.proj", "blocks.0.fc1"]


def test_explicit_backend_rejects_group_sizes_it_has_no_kernel_for():
    with pytest.raises(ValueError, match="group sizes"):
        KitchenInt8Config(group_size=16, backend="sgl_kernel")
    with pytest.raises(ValueError, match="group sizes"):
        KitchenInt8Config(group_size=128, backend="comfy_kitchen")
    with pytest.raises(ValueError, match="backend must be one of"):
        KitchenInt8Config(backend="triton")


def test_require_comfy_kitchen_pins_auto_and_refuses_explicit_sgl_kernel():
    with (
        patch(_SGL_AVAILABLE, return_value=True),
        patch(_COMFY_AVAILABLE, return_value=True),
    ):
        config = KitchenInt8Config()
        config.require_comfy_kitchen("is not validated here")
        assert config.resolve_backend() == "comfy_kitchen"
        with pytest.raises(ValueError, match="sgl_kernel is not validated here"):
            KitchenInt8Config(backend="sgl_kernel").require_comfy_kitchen(
                "is not validated here"
            )


@requires_kernel
def test_serialized_layer_on_sgl_kernel_matches_online_quantization_bitwise():
    """Loading the online path's INT8 weight and fp32 [N, 1] row scale through
    the serialized method must reproduce the online layer bit for bit."""
    online, _ = _quantized_layer(_sgl_config(), 3072, 3072, "attn.to_q", seed=30)
    markers = {
        "attn.to_q": {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        }
    }
    config = KitchenInt8Config(layer_markers=markers, backend="sgl_kernel")
    serialized = ReplicatedLinear(
        3072, 3072, params_dtype=torch.bfloat16, quant_config=config, prefix="attn.to_q"
    ).cuda()
    assert isinstance(serialized.quant_method, ConvRotInt8SglKernelLinearMethod)
    with torch.no_grad():
        serialized.weight.copy_(online.weight)
        serialized.weight_scale.copy_(online.weight_scale.view(-1, 1))
        serialized.bias.copy_(online.bias)
    serialized.quant_method.process_weights_after_loading(serialized)

    x = torch.randn(2, 512, 3072, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(serialized(x)[0], online(x)[0])
    assert convrot_int8_shares_input([serialized, online])


def test_auto_logs_why_the_sgl_kernel_backend_is_unavailable(caplog):
    """A refused GPU (CC 10.3) must still say why once the method is kitchen_int8
    with auto backend, not only when sgl_kernel is requested explicitly."""
    import logging

    from sglang.multimodal_gen.runtime.layers.quantization.configs import (
        kitchen_int8_config,
    )

    kitchen_int8_config._log_sgl_kernel_fallback.cache_clear()
    reason = (
        "CC 10.3 is not supported: Blackwell Ultra cuts INT8 tensor-core throughput"
    )
    with (
        patch(
            "sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_sgl_kernel."
            "sgl_kernel_convrot_unavailable_reason",
            return_value=reason,
        ),
        patch(_COMFY_AVAILABLE, return_value=True),
        caplog.at_level(logging.INFO),
    ):
        config = KitchenInt8Config()
        assert config.resolve_backend() == "comfy_kitchen"
        assert config.resolve_backend() == "comfy_kitchen"
    messages = [
        r.getMessage()
        for r in caplog.records
        if "sgl-kernel backend unavailable" in r.getMessage()
    ]
    assert len(messages) == 1 and reason in messages[0]


@patch(_LOAD_SGL_KERNEL)
def test_shared_input_requires_one_group_size(_sgl):
    """One rotated activation serves several layers only at one group width; a
    serialized checkpoint may mix 64- and 256-group layers."""
    markers = {
        "a": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
        "b": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64},
        "c": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    }
    config = KitchenInt8Config(layer_markers=markers, backend="sgl_kernel")
    a, b, c = (
        ReplicatedLinear(256, 16, bias=False, quant_config=config, prefix=name)
        for name in "abc"
    )
    assert convrot_int8_shares_input([a, c])
    assert not convrot_int8_shares_input([a, b])


def test_every_no_kernel_branch_raises_instead_of_falling_back():
    """A serialized INT8 layer that no backend serves must not land in a BF16
    parameter, and an impossible auto config must fail at construction."""
    with (
        patch(_SGL_AVAILABLE, return_value=False),
        patch(_COMFY_AVAILABLE, return_value=True),
    ):
        with pytest.raises(ValueError, match="no backend on this machine"):
            KitchenInt8Config(group_size=128)
    with (
        patch(_SGL_AVAILABLE, return_value=True),
        patch(_COMFY_AVAILABLE, return_value=False),
    ):
        with pytest.raises(ValueError, match="no kernel for group size 128"):
            KitchenInt8Config(group_size=128).require_comfy_kitchen("is pinned")
        markers = {
            "gate": {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            }
        }
        config = KitchenInt8Config(layer_markers=markers)
        with pytest.raises(ValueError, match="pip install comfy-kitchen"):
            config.get_quant_method(LinearBase(256, 4), "gate")


@patch(_LOAD_COMFY_KITCHEN)
@patch(_LOAD_SGL_KERNEL)
def test_serialized_layers_log_the_backend_split_once_loaded(_sgl, _comfy, caplog):
    """Serialized layers skip online quantization, so the backend split must be
    reported when their weights are in place instead."""
    import logging

    markers = {
        "fc1": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
        "small": {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 16,
        },
    }
    with (
        patch(_SGL_AVAILABLE, return_value=True),
        patch(_COMFY_AVAILABLE, return_value=True),
        caplog.at_level(logging.INFO),
    ):
        config = KitchenInt8Config(layer_markers=markers)
        fc1 = ReplicatedLinear(256, 16, bias=False, quant_config=config, prefix="fc1")
        small = ReplicatedLinear(
            64, 16, bias=False, quant_config=config, prefix="small"
        )
        fc1.quant_method.process_weights_after_loading(fc1)
        assert not [r for r in caplog.records if "serialized INT8" in r.getMessage()]
        small.quant_method.process_weights_after_loading(small)
    lines = [
        r.getMessage() for r in caplog.records if "serialized INT8" in r.getMessage()
    ]
    assert lines == [
        "kitchen_int8: loaded 2 serialized INT8 linear layers (sgl_kernel 1, comfy_kitchen 1)"
    ]
