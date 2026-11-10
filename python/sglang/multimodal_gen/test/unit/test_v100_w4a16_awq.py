import torch
from sglang.multimodal_gen.runtime.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.v100_w4a16_awq import (
    V100W4A16AWQConfig,
    V100W4A16AWQLinearMethod,
    _pack_awq_columns,
    _unpack_awq_columns,
)


def test_v100_w4a16_awq_pack_round_trip():
    values = torch.arange(32, dtype=torch.int32).reshape(2, 16) % 16
    torch.testing.assert_close(_unpack_awq_columns(_pack_awq_columns(values)), values)


def test_v100_w4a16_awq_rank_local_storage_and_cpu_forward():
    torch.manual_seed(23)
    layer = ReplicatedLinear(
        128,
        64,
        bias=True,
        params_dtype=torch.float16,
        quant_config=V100W4A16AWQConfig(),
        prefix="probe",
    )
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.05)
        layer.bias.normal_(mean=0.0, std=0.01)
    inputs = torch.randn(7, 128, dtype=torch.float16)
    reference = torch.nn.functional.linear(inputs, layer.weight, layer.bias)

    assert isinstance(layer.quant_method, V100W4A16AWQLinearMethod)
    fp16_weight_bytes = layer.weight.numel() * layer.weight.element_size()
    layer.quant_method.process_weights_after_loading(layer)
    actual, output_bias = layer(inputs)
    stored_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (layer.weight, layer.weight_scale, layer.weight_zero)
    )

    assert layer.weight.dtype == torch.int32
    assert layer.weight.shape == (128, 8)
    assert layer.weight_scale.shape == (1, 64)
    assert layer.weight_zero.shape == (1, 8)
    assert stored_bytes < fp16_weight_bytes * 0.28
    assert output_bias is None
    error = actual.float() - reference.float()
    assert error.square().mean().sqrt() < 0.07
    assert (
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(), reference.float().flatten(), dim=0
        )
        > 0.99
    )


def test_v100_w4a16_awq_ignored_layer_and_shape_contracts():
    ignored = ReplicatedLinear(
        16,
        8,
        params_dtype=torch.float16,
        quant_config=V100W4A16AWQConfig(ignored_layers=["final_layer"]),
        prefix="final_layer.video_out",
    )
    assert isinstance(ignored.quant_method, UnquantizedLinearMethod)

    for input_size, output_size, message in (
        (96, 64, "divisible by group_size"),
        (128, 62, "divisible by 8"),
    ):
        try:
            ReplicatedLinear(
                input_size,
                output_size,
                params_dtype=torch.float16,
                quant_config=V100W4A16AWQConfig(),
                prefix="bad_shape",
            )
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("An incompatible AWQ shape must be rejected")

    try:
        ReplicatedLinear(
            128,
            64,
            params_dtype=torch.bfloat16,
            quant_config=V100W4A16AWQConfig(),
            prefix="bad_dtype",
        )
    except ValueError as exc:
        assert "requires FP16" in str(exc)
    else:
        raise AssertionError("BF16 source weights must be rejected on V100 W4A16")
