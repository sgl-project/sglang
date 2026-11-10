import torch
from sglang.multimodal_gen.runtime.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.v100_w8a16 import (
    V100W8A16Config,
    V100W8A16LinearMethod,
)


def test_v100_w8a16_rank_local_storage_and_forward():
    torch.manual_seed(17)
    config = V100W8A16Config()
    layer = ReplicatedLinear(
        64,
        32,
        bias=True,
        params_dtype=torch.float16,
        quant_config=config,
        prefix="probe",
    )
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.05)
        layer.bias.normal_(mean=0.0, std=0.01)
    inputs = torch.randn(5, 64, dtype=torch.float16)
    reference = torch.nn.functional.linear(inputs, layer.weight, layer.bias)

    assert isinstance(layer.quant_method, V100W8A16LinearMethod)
    fp16_weight_bytes = layer.weight.numel() * layer.weight.element_size()
    layer.quant_method.process_weights_after_loading(layer)

    actual, output_bias = layer(inputs)
    stored_bytes = (
        layer.weight.numel() * layer.weight.element_size()
        + layer.weight_scale.numel() * layer.weight_scale.element_size()
    )
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.dtype == torch.float16
    assert layer.weight_scale.shape == (32,)
    assert stored_bytes < fp16_weight_bytes * 0.52
    assert output_bias is None
    torch.testing.assert_close(actual, reference, rtol=0.025, atol=0.025)


def test_v100_w8a16_ignored_layer_and_fp16_contract():
    config = V100W8A16Config(ignored_layers=["final_layer"])
    ignored = ReplicatedLinear(
        8,
        4,
        params_dtype=torch.float16,
        quant_config=config,
        prefix="final_layer.video_out",
    )
    assert isinstance(ignored.quant_method, UnquantizedLinearMethod)

    try:
        ReplicatedLinear(
            8,
            4,
            params_dtype=torch.bfloat16,
            quant_config=V100W8A16Config(),
            prefix="bad_dtype",
        )
    except ValueError as exc:
        assert "requires FP16" in str(exc)
    else:
        raise AssertionError("BF16 source weights must be rejected on V100 W8A16")
