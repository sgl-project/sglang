# SPDX-License-Identifier: Apache-2.0
"""XPU dispatch, loading contracts and native ConvRot integration tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
    ConvRotInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_jit import (
    convrot_int8_fuses_gelu_input,
    convrot_int8_shares_input,
)
from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_xpu import (
    ConvRotInt8XpuLinearMethod,
    _load_xpu_kernels,
)

_CONFIG = (
    "sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config"
)
_XPU = "sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_xpu"

requires_xpu = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires BMG XPU and ConvRot sgl-kernel-xpu"
)


@pytest.fixture(autouse=True)
def xpu_platform():
    with patch(f"{_CONFIG}.current_platform", SimpleNamespace(is_xpu=lambda: True)):
        yield


def _fake_kernels():
    return patch(f"{_XPU}._load_xpu_kernels", return_value=(None, None))


def _markers(group_size=256):
    return {
        "proj": {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": group_size,
        }
    }


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("backend", ["auto", "xpu"])
def test_xpu_selection_never_probes_cuda_or_comfy(serialized, backend):
    with (
        _fake_kernels(),
        patch(f"{_CONFIG}._jit_available", side_effect=AssertionError("CUDA probe")),
        patch(
            f"{_CONFIG}._comfy_kitchen_available",
            side_effect=AssertionError("comfy probe"),
        ),
    ):
        config = ConvRotInt8Config(
            backend=backend, layer_markers=_markers() if serialized else None
        )
        method = config.get_quant_method(LinearBase(5376, 5376), "proj")
    assert isinstance(method, ConvRotInt8XpuLinearMethod)
    assert method.is_checkpoint_serialized == serialized
    assert config.resolve_backend() == "xpu"
    assert config.selected_by_backend["xpu"] == ["proj"]
    assert config.selected_by_backend["jit"] == []
    assert config.selected_by_backend["comfy_kitchen"] == []
    assert config.get_supported_act_dtypes() == [torch.bfloat16]


@pytest.mark.parametrize("backend", ["jit", "comfy_kitchen"])
def test_xpu_refuses_cuda_backends(backend):
    with pytest.raises(ValueError, match="requires backend auto or xpu"):
        ConvRotInt8Config(backend=backend)


@pytest.mark.parametrize("group_size", [16, 64, 128, 512])
def test_xpu_auto_rejects_unsupported_online_group(group_size):
    with pytest.raises(ValueError, match="only group size 256"):
        ConvRotInt8Config(group_size=group_size)


@pytest.mark.parametrize("group_size", [16, 64])
def test_xpu_serialized_group_has_no_eager_fallback(group_size):
    config = ConvRotInt8Config(layer_markers=_markers(group_size))
    with pytest.raises(ValueError, match="No comfy_kitchen/eager fallback"):
        config.get_quant_method(LinearBase(256, 32), "proj")


def test_explicit_xpu_rejects_unsupported_group():
    with pytest.raises(ValueError, match="group sizes"):
        ConvRotInt8Config(backend="xpu", group_size=64)


@pytest.mark.parametrize("backend", ["auto", "xpu"])
def test_xpu_cannot_be_pinned_to_comfy_for_fsdp(backend):
    config = ConvRotInt8Config(backend=backend)
    with pytest.raises(ValueError, match="No comfy_kitchen fallback"):
        config.require_comfy_kitchen("has not been validated with FSDP")


def test_old_kernel_package_fails_with_rebuild_hint():
    _load_xpu_kernels.cache_clear()
    with patch.dict("sys.modules", {"sgl_kernel": SimpleNamespace()}):
        with pytest.raises(RuntimeError, match="rebuild and install"):
            _load_xpu_kernels()


def test_kernel_and_device_probe_runs_once_per_process():
    ops = SimpleNamespace(
        convrot_rotate_quantize=SimpleNamespace(default="rotate"),
        convrot_int8_linear_prequant=SimpleNamespace(default="gemm"),
        query_device=lambda device: (2, 0),
    )
    package = SimpleNamespace(
        convrot_rotate_quantize=None, convrot_int8_linear_prequant=None
    )
    _load_xpu_kernels.cache_clear()
    try:
        with (
            patch.dict("sys.modules", {"sgl_kernel": package}),
            patch.object(torch.ops, "sgl_kernel", ops, create=True),
            patch.object(torch.xpu, "is_available", return_value=True) as probe,
            patch.object(torch.xpu, "current_device", return_value=0),
        ):
            config = ConvRotInt8Config(layer_markers=_markers())
            methods = [
                config.get_quant_method(LinearBase(256, 32), "proj") for _ in range(3)
            ]
        assert probe.call_count == 1
        assert all(
            (m._rotate_quantize, m._linear_prequant) == ("rotate", "gemm")
            for m in methods
        )
    finally:
        _load_xpu_kernels.cache_clear()


def test_xpu_load_error_is_not_substituted():
    with patch(f"{_XPU}._load_xpu_kernels", side_effect=RuntimeError("missing op")):
        with pytest.raises(RuntimeError, match="missing op"):
            ConvRotInt8Config().get_quant_method(LinearBase(256, 32), "proj")


def test_intended_online_bf16_exclusions_are_preserved():
    config = ConvRotInt8Config(ignored_layers=["ignored"])
    for prefix, k in (("ignored.proj", 256), ("adaln", 2688)):
        assert isinstance(
            config.get_quant_method(LinearBase(k, 32), prefix),
            UnquantizedLinearMethod,
        )
    assert config.skipped == ["ignored.proj", "adaln(in=2688)"]


@pytest.mark.parametrize("n", [0, 4, 33])
def test_serialized_invalid_output_width_raises(n):
    config = ConvRotInt8Config(layer_markers=_markers())
    with pytest.raises(ValueError, match="divisible by 32"):
        config.get_quant_method(LinearBase(256, n), "proj")


def test_serialized_input_partition_above_native_limit_raises():
    config = ConvRotInt8Config(layer_markers=_markers())
    with pytest.raises(ValueError, match=r"\[256, 8192\].*8448 inputs"):
        config.get_quant_method(LinearBase(8448, 32), "proj")


@pytest.mark.parametrize("k,n", [(256, 4), (256, 33), (8448, 32)])
def test_online_unsupported_shape_stays_bf16(k, n):
    config = ConvRotInt8Config()
    method = config.get_quant_method(LinearBase(k, n), "proj")
    assert isinstance(method, UnquantizedLinearMethod)
    assert config.skipped == [f"proj(in={k}, out={n})"]
    assert config.selected_by_backend["xpu"] == []


def test_online_row_parallel_shard_is_screened_per_partition(monkeypatch):
    import sglang.multimodal_gen.runtime.layers.linear as linear

    monkeypatch.setattr(linear, "get_tp_group", lambda: object())
    monkeypatch.setattr(linear, "get_group_rank", lambda group: 0)
    monkeypatch.setattr(linear, "get_group_size", lambda group: 4)
    config = ConvRotInt8Config()
    # 2560 divides by the group, but its 640-wide shard does not.
    narrow = RowParallelLinear(
        2560, 32, params_dtype=torch.bfloat16, quant_config=config, prefix="narrow"
    )
    with _fake_kernels():
        # 16384 exceeds the native K limit, but its 4096-wide shard does not.
        wide = RowParallelLinear(
            16384, 32, params_dtype=torch.bfloat16, quant_config=config, prefix="wide"
        )
    assert isinstance(narrow.quant_method, UnquantizedLinearMethod)
    assert isinstance(wide.quant_method, ConvRotInt8XpuLinearMethod)
    assert config.skipped == ["narrow(in=640, out=32)"]
    assert config.selected_by_backend["xpu"] == ["wide"]


@pytest.mark.parametrize("k", [128, 384, 8448])
def test_weight_creation_validates_actual_input_partition(k):
    with _fake_kernels():
        method = ConvRotInt8XpuLinearMethod(
            ConvRotInt8Config(), group_size=256, is_checkpoint_serialized=True
        )
    with pytest.raises(ValueError, match="input_size_per_partition"):
        method.create_weights(torch.nn.Module(), k, [32], k, 32, torch.bfloat16)


def test_fp16_parameters_are_rejected_without_implicit_torch_casts():
    with _fake_kernels():
        with pytest.raises(ValueError, match="BF16 parameters"):
            ReplicatedLinear(
                256,
                32,
                params_dtype=torch.float16,
                quant_config=ConvRotInt8Config(),
                prefix="proj",
            )


@pytest.mark.parametrize("backend", ["auto", "xpu"])
def test_partition_eligibility_includes_native_k_limit(backend):
    config = ConvRotInt8Config(backend=backend, layer_markers=_markers())
    assert config.supports_input_partition("proj", 1792)
    assert config.supports_input_partition("proj", 8192)
    assert not config.supports_input_partition("proj", 8448)
    assert not config.supports_input_partition("proj", 128)
    assert config.supports_input_partition("unquantized", 8448)


@pytest.mark.parametrize(
    "layer_type,k,n,expected_k,expected_n",
    [
        (ColumnParallelLinear, 5376, 21504, 5376, 5376),
        (RowParallelLinear, 7168, 5376, 1792, 5376),
    ],
)
def test_tp4_native_weight_and_scale_loading(
    monkeypatch, layer_type, k, n, expected_k, expected_n
):
    import sglang.multimodal_gen.runtime.layers.linear as linear

    monkeypatch.setattr(linear, "get_tp_group", lambda: object())
    monkeypatch.setattr(linear, "get_group_rank", lambda group: 2)
    monkeypatch.setattr(linear, "get_group_size", lambda group: 4)
    with _fake_kernels():
        layer = layer_type(
            k,
            n,
            params_dtype=torch.bfloat16,
            quant_config=ConvRotInt8Config(layer_markers=_markers()),
            prefix="proj",
        )
    assert layer.weight.shape == (expected_n, expected_k)
    assert layer.weight_scale.shape == (expected_n, 1)
    # Use expanded CPU sources so the test need not allocate the full H3 weight.
    weight = torch.arange(n, dtype=torch.int32).to(torch.int8)[:, None].expand(n, k)
    scale = torch.arange(n, dtype=torch.float32)[:, None]
    layer.weight_loader(layer.weight, weight)
    layer.weight_loader(layer.weight_scale, scale)
    start = expected_n * 2 if layer_type is ColumnParallelLinear else 0
    torch.testing.assert_close(
        layer.weight[:, 0], weight[start : start + expected_n, 0]
    )
    torch.testing.assert_close(layer.weight_scale, scale[start : start + expected_n])


@pytest.mark.parametrize("widths", [[7168] * 3, [14336] * 2])
def test_tp4_merged_qkv_and_gate_up_scales_follow_each_logical_shard(
    monkeypatch, widths
):
    import sglang.multimodal_gen.runtime.layers.linear as linear

    monkeypatch.setattr(linear, "get_tp_group", lambda: object())
    monkeypatch.setattr(linear, "get_group_rank", lambda group: 2)
    monkeypatch.setattr(linear, "get_group_size", lambda group: 4)
    with _fake_kernels():
        layer = MergedColumnParallelLinear(
            256,
            widths,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=ConvRotInt8Config(layer_markers=_markers()),
            prefix="proj",
        )
    scale = torch.arange(sum(widths), dtype=torch.float32)[:, None]
    weight = scale.to(torch.int8).expand(-1, 256)
    layer.weight_loader(layer.weight, weight)
    layer.weight_loader(layer.weight_scale, scale)
    expected = torch.cat(
        [
            part[2 * (width // 4) : 3 * (width // 4)]
            for part, width in zip(scale.split(widths), widths, strict=True)
        ]
    )
    torch.testing.assert_close(layer.weight_scale, expected)
    torch.testing.assert_close(layer.weight[:, :1], expected.to(torch.int8))


@requires_xpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_apply_rejects_non_bf16_without_fallback(dtype):
    layer = ReplicatedLinear(
        256,
        32,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=ConvRotInt8Config(layer_markers=_markers()),
        prefix="proj",
    ).xpu()
    with pytest.raises(ValueError, match="BF16 activations on XPU"):
        layer(torch.zeros(1, 256, device="xpu", dtype=dtype))


@requires_xpu
@pytest.mark.parametrize("device_index", [0, 1, 2, 3])
def test_modelopt_mixed_checkpoint_dispatch_runs_native_on_each_card(device_index):
    if device_index >= torch.xpu.device_count():
        pytest.skip("requires four visible XPU devices")
    from sgl_kernel import convrot_int8_linear_prequant, convrot_rotate_quantize

    from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
        ModelOptFp4Config,
    )

    device = torch.device("xpu", device_index)
    with torch.xpu.device(device):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            checkpoint_uses_comfy_quantization=True,
        )
        config.set_comfy_layer_markers(_markers())
        layer = ReplicatedLinear(
            256, 32, params_dtype=torch.bfloat16, quant_config=config, prefix="proj"
        ).to(device)
        assert isinstance(layer.quant_method, ConvRotInt8XpuLinearMethod)
        weight = torch.randn(32, 256, device=device, dtype=torch.bfloat16)
        qweight, wscale = convrot_rotate_quantize(weight)
        layer.weight.data.copy_(qweight)
        layer.weight_scale.data.copy_(wscale[:, None])
        layer.bias.data.zero_()
        layer.quant_method.process_weights_after_loading(layer)
        x = torch.randn(2, 17, 256, device=device, dtype=torch.bfloat16)
        q, scale = convrot_rotate_quantize(x.reshape(-1, 256))
        expected = convrot_int8_linear_prequant(
            q, qweight, scale, wscale, layer.bias
        ).reshape(2, 17, 32)
        out, bias = layer(x)
        assert bias is None
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert not convrot_int8_shares_input([layer])
        assert not convrot_int8_fuses_gelu_input(layer)


@requires_xpu
@pytest.mark.parametrize("offloaded", [False, True])
def test_online_quantization_matches_serialized_layer(offloaded):
    device = torch.device("xpu", torch.xpu.current_device())
    online_config = ConvRotInt8Config()
    layer = ReplicatedLinear(
        512,
        96,
        params_dtype=torch.bfloat16,
        quant_config=online_config,
        prefix="proj",
    )
    layer.weight.data.normal_(std=0.02)
    layer.bias.data.zero_()
    weight = layer.weight.detach().to(device)
    if not offloaded:
        layer = layer.to(device)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight.device.type == ("cpu" if offloaded else "xpu")
    layer = layer.to(device)
    serialized = ReplicatedLinear(
        512,
        96,
        params_dtype=torch.bfloat16,
        quant_config=ConvRotInt8Config(layer_markers=_markers()),
        prefix="proj",
    ).to(device)
    serialized.weight.data.copy_(layer.weight)
    serialized.weight_scale.data.copy_(layer.weight_scale[:, None])
    serialized.bias.data.copy_(layer.bias)
    serialized.quant_method.process_weights_after_loading(serialized)
    x = torch.randn(2, 17, 512, device=device, dtype=torch.bfloat16)
    actual = layer(x)[0]
    torch.testing.assert_close(serialized(x)[0], actual, rtol=0, atol=0)
    ref = torch.nn.functional.linear(x, weight, layer.bias)
    relative_l2 = (
        torch.linalg.vector_norm(actual.float() - ref.float())
        / torch.linalg.vector_norm(ref.float())
    ).item()
    assert relative_l2 < 0.02
    assert online_config._processed == 1


@requires_xpu
@pytest.mark.parametrize("shape", [(256,), (0, 256), (2, 0, 256), (2, 17, 256)])
@pytest.mark.parametrize("deferred_bias", [False, True])
def test_forward_shapes_noncontiguous_and_bias_contract(shape, deferred_bias):
    layer = ReplicatedLinear(
        256,
        32,
        skip_bias_add=deferred_bias,
        params_dtype=torch.bfloat16,
        quant_config=ConvRotInt8Config(layer_markers=_markers()),
        prefix="proj",
    ).xpu()
    layer.weight.data.fill_(1)
    layer.weight_scale.data.fill_(0.01)
    layer.bias.data.fill_(1)
    x = torch.randn(*shape[:-1], 512, device="xpu", dtype=torch.bfloat16)[..., ::2]
    out, bias = layer(x)
    assert out.shape == (*shape[:-1], 32)
    assert out.dtype == torch.bfloat16
    if deferred_bias:
        assert bias is layer.bias
    else:
        assert bias is None
    if x.numel():
        expected = layer.quant_method.apply(
            layer, x.contiguous(), None if deferred_bias else layer.bias
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@requires_xpu
def test_layer_trace_contains_only_two_native_kernels():
    layer = ReplicatedLinear(
        256,
        32,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=ConvRotInt8Config(layer_markers=_markers()),
        prefix="proj",
    ).xpu()
    layer.weight.data.fill_(1)
    layer.weight_scale.data.fill_(0.01)
    x = torch.randn(17, 256, device="xpu", dtype=torch.bfloat16)
    layer(x)
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as prof:
        layer(x)
        torch.xpu.synchronize()
    names = [event.name for event in prof.events()]
    assert names.count("sgl_kernel::convrot_rotate_quantize") == 1
    assert names.count("sgl_kernel::convrot_int8_linear_prequant") == 1
    assert not any("comfy_kitchen::" in name or "_int_mm" in name for name in names)
    kernels = [
        event
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.XPU
    ]
    assert len(kernels) == 2, [event.name for event in kernels]
