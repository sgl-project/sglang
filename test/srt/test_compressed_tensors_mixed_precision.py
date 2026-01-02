import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
    CompressedTensorsWNA16,
)


def _build_mixed_precision_config():
    return {
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "targets": ["attn"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": True,
                },
            },
            "group_1": {
                "format": "pack-quantized",
                "targets": ["mlp"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 128,
                    "symmetric": True,
                    "dynamic": False,
                },
            },
        },
    }


def test_mixed_precision_parses_activation_by_group_format():
    target_scheme_map = CompressedTensorsConfig._quantization_scheme_map_from_config(
        _build_mixed_precision_config()
    )

    attn_scheme = target_scheme_map["attn"]
    mlp_scheme = target_scheme_map["mlp"]

    assert attn_scheme["format"] == "float-quantized"
    assert attn_scheme["input_activations"] is not None
    assert mlp_scheme["format"] == "pack-quantized"
    assert mlp_scheme["input_activations"] is None


def test_get_scheme_uses_group_format_for_mixed_precision():
    quant_config = CompressedTensorsConfig.from_config(_build_mixed_precision_config())
    quant_config._check_scheme_supported = (  # type: ignore[method-assign]
        lambda *args, **kwargs: True
    )

    dummy_layer = torch.nn.Linear(4, 4)

    attn_scheme = quant_config.get_scheme(dummy_layer, layer_name="attn")
    assert isinstance(attn_scheme, CompressedTensorsW8A8Fp8)

    mlp_scheme = quant_config.get_scheme(dummy_layer, layer_name="mlp")
    assert isinstance(mlp_scheme, CompressedTensorsWNA16)
