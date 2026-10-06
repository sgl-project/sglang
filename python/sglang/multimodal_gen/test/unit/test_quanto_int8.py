import base64
import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization.configs.quanto_int8_config import (
    QuantoInt8Config,
    inspect_quanto_int8_checkpoint,
)
from sglang.multimodal_gen.runtime.layers.quantization.quanto_int8 import (
    normalize_quanto_int8_weights,
)


def _save_quanto_checkpoint(path, *, activations="none"):
    prefix = "language_model.layers.0.mlp.up_proj"
    quantization_map = {prefix: {"weights": "qint8", "activations": activations}}
    save_file(
        {
            f"{prefix}.weight._data": torch.tensor(
                [[1, -2], [3, 4], [-5, 6]], dtype=torch.int8
            ),
            f"{prefix}.weight._scale": torch.tensor(
                [[0.5], [0.25], [0.125]], dtype=torch.bfloat16
            ),
            f"{prefix}.input_scale": torch.tensor(1, dtype=torch.bfloat16),
            f"{prefix}.output_scale": torch.tensor(1, dtype=torch.bfloat16),
        },
        path,
        metadata={
            "quantization_format": "quanto",
            "quantization_map_base64": base64.b64encode(
                json.dumps(quantization_map).encode()
            ).decode(),
        },
    )


def test_quanto_checkpoint_drives_native_linear_end_to_end(tmp_path):
    checkpoint = tmp_path / "encoder.safetensors"
    _save_quanto_checkpoint(checkpoint)
    config = inspect_quanto_int8_checkpoint(
        str(checkpoint), param_name_mapper=lambda name: f"model.{name}"
    )

    assert isinstance(config, QuantoInt8Config)
    prefix = "model.language_model.layers.0.mlp.up_proj"
    layer = ReplicatedLinear(
        2,
        3,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix=prefix,
    )
    tensors = dict(normalize_quanto_int8_weights(load_file(str(checkpoint)).items()))
    raw_prefix = prefix.removeprefix("model.")
    for suffix, parameter in (
        ("weight", layer.weight),
        ("weight_scale", layer.weight_scale),
    ):
        parameter.weight_loader(parameter, tensors.pop(f"{raw_prefix}.{suffix}"))

    x = torch.tensor([[2.0, -1.0]], dtype=torch.bfloat16)
    expected_weight = layer.weight.to(torch.bfloat16) * layer.weight_scale
    output, _ = layer(x)
    torch.testing.assert_close(output, F.linear(x, expected_weight))
    assert not tensors
    assert config.selected == {prefix}


def test_quanto_checkpoint_rejects_activation_quantization(tmp_path):
    checkpoint = tmp_path / "encoder.safetensors"
    _save_quanto_checkpoint(checkpoint, activations="qint8")
    with pytest.raises(ValueError, match="activation quantization"):
        inspect_quanto_int8_checkpoint(str(checkpoint))


def test_quanto_weight_only_auxiliary_scales_must_be_identity():
    with pytest.raises(ValueError, match="must equal 1"):
        list(normalize_quanto_int8_weights([("layer.input_scale", torch.tensor(0.5))]))
