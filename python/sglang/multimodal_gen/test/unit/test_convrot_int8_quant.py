"""Unit tests for the convrot_int8_customkernel online quantization method."""

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
from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_customkernel_config import (
    ConvRotInt8CustomKernelConfig,
    convrot_int8_supported_capabilities,
)
from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_customkernel import (
    _REQUIRED_OPS,
    ConvRotInt8CustomKernelLinearMethod,
    apply_convrot_int8_gelu_input,
    apply_convrot_int8_shared_input,
    apply_convrot_int8_shared_input_out,
    convrot_int8_fuses_gelu_input,
    convrot_int8_shares_input,
)

_LOAD_SGL_KERNEL = (
    "sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_customkernel."
    "_load_sgl_kernel"
)

QWEN_IMAGE_IGNORED_LAYERS = ["img_mod", "txt_mod", "txt_mlp.net.2"]


def _kernel_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import sgl_kernel  # noqa: F401
    except ImportError:
        return False
    if not all(hasattr(torch.ops.sgl_kernel, op) for op in _REQUIRED_OPS):
        return False
    return torch.cuda.get_device_capability() in convrot_int8_supported_capabilities()


requires_kernel = pytest.mark.skipif(
    not _kernel_available(),
    reason="needs a GPU in sgl-kernel's convrot table and a build with the convrot ops",
)


def _method_for(config, prefix, input_size=3072):
    layer = LinearBase(input_size=input_size, output_size=32)
    return config.get_quant_method(layer, prefix)


@patch(_LOAD_SGL_KERNEL)
def test_ignored_layer_patterns_select_bf16_on_module_path_boundaries(_load):
    """A pattern must match its own module path only: `txt_mlp.net.2` keeps the
    text FFN down-projection in BF16 without also catching `img_mlp.net.2` or
    the text FFN up-projection, and `img_mod` must not catch `img_mlp`."""
    config = ConvRotInt8CustomKernelConfig(ignored_layers=QWEN_IMAGE_IGNORED_LAYERS)

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
        assert isinstance(
            _method_for(config, prefix), ConvRotInt8CustomKernelLinearMethod
        )
    assert config.skipped == kept_bf16
    assert config.selected == quantized


@patch(_LOAD_SGL_KERNEL)
def test_input_dim_not_divisible_by_group_stays_bf16(_load):
    config = ConvRotInt8CustomKernelConfig()
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
    config = ConvRotInt8CustomKernelConfig()
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


def _quantized_layer(config, in_features, out_features, prefix, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    layer = ReplicatedLinear(
        in_features,
        out_features,
        params_dtype=torch.bfloat16,
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
    config = ConvRotInt8CustomKernelConfig()
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
    config = ConvRotInt8CustomKernelConfig()
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
    config = ConvRotInt8CustomKernelConfig()
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
    config = ConvRotInt8CustomKernelConfig()
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
def test_row_parallel_shard_not_divisible_by_group_stays_bf16(
    _load, monkeypatch, tp_size, input_size
):
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear

    _with_tp(monkeypatch, tp_size)
    config = ConvRotInt8CustomKernelConfig()
    layer = RowParallelLinear(
        input_size, 5120, bias=True, quant_config=config, prefix="blocks.0.ffn.fc_out"
    )
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    assert config.skipped == [f"blocks.0.ffn.fc_out(in={input_size // tp_size})"]
    assert config.selected == []


@patch(_LOAD_SGL_KERNEL)
def test_row_parallel_shard_divisible_by_group_is_quantized(_load, monkeypatch):
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear

    _with_tp(monkeypatch, 4)
    config = ConvRotInt8CustomKernelConfig()
    layer = RowParallelLinear(
        12288, 3072, bias=True, quant_config=config, prefix="blocks.0.ffn.net.2"
    )
    assert isinstance(layer.quant_method, ConvRotInt8CustomKernelLinearMethod)
    assert config.selected == ["blocks.0.ffn.net.2"]


@patch(_LOAD_SGL_KERNEL)
def test_output_width_not_multiple_of_8_stays_bf16(_load):
    config = ConvRotInt8CustomKernelConfig()
    layer = LinearBase(input_size=4096, output_size=4)
    method = config.get_quant_method(layer, "blocks.0.attn.to_gate_logits")
    assert isinstance(method, UnquantizedLinearMethod)
    assert config.skipped == ["blocks.0.attn.to_gate_logits(out=4)"]


@patch(_LOAD_SGL_KERNEL)
def test_column_parallel_shard_output_not_multiple_of_8_stays_bf16(_load, monkeypatch):
    from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear

    _with_tp(monkeypatch, 8)
    config = ConvRotInt8CustomKernelConfig()
    # 32 output rows split eight ways leave 4 per rank; the epilogue stores 8 at
    # a time, so the layer stays in BF16 instead of failing in the first forward.
    layer = ColumnParallelLinear(
        4096, 32, bias=True, quant_config=config, prefix="blocks.0.attn.to_gate_logits"
    )
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    assert config.skipped == ["blocks.0.attn.to_gate_logits(out=4)"]


@requires_kernel
def test_fp16_activations_are_cast_and_returned_in_fp16():
    config = ConvRotInt8CustomKernelConfig()
    layer, weight = _quantized_layer(config, 3072, 3072, "attn.to_q", seed=3)
    x = torch.randn(2, 64, 3072, device="cuda", dtype=torch.float16)
    out, _ = layer(x)
    assert out.dtype == torch.float16
    ref = F.linear(x.to(torch.bfloat16), weight, layer.bias).to(torch.float16)
    err = torch.linalg.vector_norm(out.float() - ref.float())
    rel_l2 = (err / torch.linalg.vector_norm(ref.float())).item()
    assert rel_l2 < 2e-2, rel_l2


@requires_kernel
def test_lora_merge_mode_is_redirected_to_dynamic_on_int8_base():
    from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline

    config = ConvRotInt8CustomKernelConfig()
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
