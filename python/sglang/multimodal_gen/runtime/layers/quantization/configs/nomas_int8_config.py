# SPDX-License-Identifier: Apache-2.0
"""Admission and linear-layer selection for official Nomas INT8 checkpoints."""

import json

import torch
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.nomas_int8 import (
    CHECKPOINT_FORMAT,
    GROUP_SIZE,
    NomasInt8ColumnParallelLinear,
    NomasInt8Linear,
    NomasInt8LinearMethod,
    NomasInt8RowParallelLinear,
)


class NomasInt8Config(QuantizationConfig):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    @classmethod
    def get_name(cls):
        return "nomas_int8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        raise ValueError("Nomas INT8 config must be read from checkpoint metadata")

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return (
                NomasInt8LinearMethod()
                if prefix in self.layers
                else UnquantizedLinearMethod()
            )
        return None

    def remap_checkpoint_prefixes(self, param_names_mapping):
        from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

        mapper = get_param_names_mapping(param_names_mapping)
        mapped = {}
        for name, shape in self.layers.items():
            target = mapper(name + ".weight")[0].removesuffix(".weight")
            if target in mapped:
                raise ValueError(f"Duplicate mapped Nomas INT8 layer: {target}")
            mapped[target] = shape
        self.layers = mapped

    def configure_model(self, model):
        """Replace only declared linears before checkpoint materialization.

        The model is normally constructed on meta. Reconstructing its parallel
        linears retains their existing collectives and weight-loader contracts.
        """
        for name, shape in self.layers.items():
            try:
                original = model.get_submodule(name)
            except AttributeError as exc:
                raise ValueError(f"Unknown Nomas INT8 layer: {name}") from exc
            if isinstance(original, LinearBase):
                expected = [original.output_size, original.input_size]
            elif type(original) is nn.Linear:
                expected = [original.out_features, original.in_features]
            else:
                raise ValueError(f"Unsupported Nomas INT8 layer: {name}")
            if list(shape) != expected:
                raise ValueError(f"Nomas INT8 layer shape mismatch: {name}")
            kwargs = dict(
                input_size=shape[1],
                output_size=shape[0],
                bias=original.bias is not None,
                params_dtype=torch.bfloat16,
                quant_config=self,
                prefix=name,
            )
            if type(original) in (ColumnParallelLinear, RowParallelLinear):
                kwargs.update(
                    skip_bias_add=original.skip_bias_add, tp_group=original.tp_group
                )
                if isinstance(original, ColumnParallelLinear):
                    kwargs["gather_output"] = original.gather_output
                else:
                    kwargs.update(
                        input_is_parallel=original.input_is_parallel,
                        reduce_results=original.reduce_results,
                        input_sizes=original.input_sizes,
                    )
                cls = (
                    NomasInt8ColumnParallelLinear
                    if isinstance(original, ColumnParallelLinear)
                    else NomasInt8RowParallelLinear
                )
            elif type(original) is nn.Linear:
                cls = NomasInt8Linear
            else:
                raise ValueError(f"Unsupported Nomas INT8 parallel layer: {name}")
            with torch.device(original.weight.device):
                replacement = cls(**kwargs)
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, replacement)


def inspect_nomas_int8_checkpoint(path):
    """Validate all declared tensor headers without reading large weight payloads."""
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        if metadata.get("format") != CHECKPOINT_FORMAT:
            return None
        for key, expected in (
            ("group_size", str(GROUP_SIZE)),
            ("quantization", "symmetric_int8_absmax"),
            ("compute_dtype", "bfloat16"),
            ("rounding", "half_away_from_zero"),
            ("model_type", "HY-Nomas-G/2-32324-MMDiT"),
        ):
            if metadata.get(key) != expected:
                raise ValueError(f"Nomas INT8 checkpoint requires {key}={expected}")
        try:
            layers = json.loads(metadata.get("layers", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Nomas INT8 layers metadata must be a JSON object"
            ) from exc
        if not isinstance(layers, dict) or not layers:
            raise ValueError("Nomas INT8 requires a nonempty layer manifest")
        keys = set(reader.keys())
        for suffix in (".weight_q", ".weight_scale"):
            if {
                key.removesuffix(suffix) for key in keys if key.endswith(suffix)
            } != set(layers):
                raise ValueError(f"Nomas INT8 manifest does not match {suffix} tensors")
        for name, shape in layers.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(shape, list)
                or len(shape) != 2
                or any(type(dim) is not int or dim <= 0 for dim in shape)
            ):
                raise ValueError(f"Invalid Nomas INT8 layer shape: {name}")
            for suffix, expected_shape, dtype in (
                (".weight_q", shape, "I8"),
                (
                    ".weight_scale",
                    [shape[0], (shape[1] + GROUP_SIZE - 1) // GROUP_SIZE],
                    "F32",
                ),
            ):
                tensor = reader.get_slice(name + suffix)
                if tensor.get_shape() != expected_shape or tensor.get_dtype() != dtype:
                    raise ValueError(
                        f"Nomas INT8 tensor shape/dtype mismatch: {name + suffix}"
                    )
        for key in keys:
            if (
                not key.endswith((".weight_q", ".weight_scale"))
                and reader.get_slice(key).get_dtype() != "BF16"
            ):
                raise ValueError(f"Nomas INT8 unquantized tensor must be BF16: {key}")
    return NomasInt8Config(layers)
