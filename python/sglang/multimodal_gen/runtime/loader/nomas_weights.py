# SPDX-License-Identifier: Apache-2.0
"""Checkpoint inspection and model configuration for Nomas INT8 DiT and FP8 VAE."""

import json
from functools import partial

import torch
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.quanto_int8_config import (
    QuantoInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8Linear,
    validate_fp8_block_device,
)
from sglang.multimodal_gen.runtime.layers.quantization.quanto_int8 import (
    QuantoInt8ColumnParallelLinear,
    QuantoInt8Linear,
    QuantoInt8RowParallelLinear,
)

GROUP_SIZE = 128
INT8_CHECKPOINT_FORMAT = "nomas-int8-bf16-single-v1"
FP8_CHECKPOINT_FORMAT = "pixelumm_fp8_bf16_single_v1"


def inspect_nomas_fp8_checkpoint(weight_files):
    """Admit the complete official single-file format before allocating weights."""
    for path in weight_files:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            metadata = reader.metadata() or {}
            if metadata.get("format") != FP8_CHECKPOINT_FORMAT:
                if any(
                    key.endswith((".fp8_weight", ".fp8_scale")) for key in reader.keys()
                ):
                    raise ValueError("PixelUMM FP8 checkpoint requires format metadata")
                continue
            if len(weight_files) != 1:
                raise ValueError(
                    "PixelUMM FP8 requires a complete single-file checkpoint"
                )
            for key, value in {
                "block_size": "128",
                "weight_dtype": "float8_e4m3fn",
                "scale_dtype": "float32",
                "unquantized_float_dtype": "bfloat16",
            }.items():
                if metadata.get(key) != value:
                    raise ValueError(f"Unsupported PixelUMM FP8 metadata: {key}")
            sources = json.loads(metadata.get("quantized_source_keys", "[]"))
            if not isinstance(sources, list) or not all(
                isinstance(key, str) and key.endswith(".weight") for key in sources
            ):
                raise ValueError("Invalid PixelUMM FP8 layer manifest")
            layers = {key.removesuffix(".weight") for key in sources}
            if len(layers) != len(sources) or int(
                metadata.get("quantized_layers", -1)
            ) != len(layers):
                raise ValueError("Invalid PixelUMM FP8 layer count")
            expected = {
                f"{name}.{suffix}"
                for name in layers
                for suffix in ("fp8_weight", "fp8_scale")
            }
            actual = {
                key
                for key in reader.keys()
                if key.endswith((".fp8_weight", ".fp8_scale"))
            }
            if actual != expected:
                raise ValueError("Incomplete PixelUMM FP8 weights/scales")
            shapes = {}
            for name in layers:
                weight, scale = (
                    reader.get_slice(name + ".fp8_weight"),
                    reader.get_slice(name + ".fp8_scale"),
                )
                shape = weight.get_shape()
                if (
                    len(shape) != 2
                    or any(d <= 0 or d % 128 for d in shape)
                    or weight.get_dtype() != "F8_E4M3"
                    or scale.get_dtype() != "F32"
                    or scale.get_shape() != [d // 128 for d in shape]
                ):
                    raise ValueError(f"Invalid PixelUMM FP8 weight/scale: {name}")
                scales = reader.get_tensor(name + ".fp8_scale")
                if not torch.isfinite(scales).all() or not (scales > 0).all():
                    raise ValueError(
                        f"PixelUMM FP8 scales must be finite and positive: {name}"
                    )
                shapes[name] = tuple(shape)
            for name in set(reader.keys()) - expected:
                if reader.get_slice(name).get_dtype() != "BF16":
                    raise ValueError(
                        f"Expected BF16 unquantized PixelUMM tensor: {name}"
                    )
            return shapes
    return None


def configure_nomas_fp8(model, shapes):
    expected = set(model.fp8_quantized_layer_names())
    if set(shapes) != expected:
        raise ValueError("Incomplete or incompatible PixelUMM FP8 layer manifest")
    for name, shape in shapes.items():
        linear = model.get_submodule(name)
        if (
            not isinstance(linear, nn.Linear)
            or linear.bias is not None
            or tuple(linear.weight.shape) != shape
        ):
            raise ValueError(f"PixelUMM FP8 model/checkpoint mismatch: {name}")
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        activation_block_size=(128, 128),
    )
    for name in shapes:
        linear = model.get_submodule(name)
        parent, _, leaf = name.rpartition(".")
        with torch.device(linear.weight.device):
            replacement = Fp8Linear(
                linear.in_features,
                linear.out_features,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=quant_config,
                prefix=name,
            )
        model.get_submodule(parent).add_module(leaf, replacement)


def inspect_nomas_int8_checkpoint(path):
    """Validate all declared tensor headers without reading large weight payloads."""
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        if metadata.get("format") != INT8_CHECKPOINT_FORMAT:
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
    return build_nomas_int8_config(layers)


def resolve_nomas_checkpoint_quantization(weight_files):
    """Detect and validate the official single-file INT8 DiT format."""
    for path in weight_files:
        quant_config = inspect_nomas_int8_checkpoint(path)
        if quant_config is not None:
            if len(weight_files) != 1:
                raise ValueError("Nomas INT8 requires one complete safetensors file")
            return quant_config
    return None


def resolve_nomas_vae_checkpoint_configurator(weight_files, dtype):
    """Validate FP8 VAE execution and return its meta-model configurator."""
    shapes = inspect_nomas_fp8_checkpoint(weight_files)
    if shapes is None:
        return None
    if dtype != torch.bfloat16:
        raise ValueError("PixelUMM FP8 requires BF16 VAE compute precision")
    # Residency may start on CPU; validate the device that will run inference.
    validate_fp8_block_device(get_local_torch_device())
    return partial(configure_nomas_fp8, shapes=shapes)


def build_nomas_int8_config(layers):
    """Describe official Nomas storage using the shared INT8 configuration."""
    return QuantoInt8Config(
        set(layers),
        group_size=GROUP_SIZE,
        scale_dtype=torch.float32,
        compute_dtype=torch.bfloat16,
        layer_shapes=layers,
        supports_fsdp=False,
        supports_quantized_cpu_load=True,
    )


def validate_nomas_int8_config(quant_config):
    if not (
        isinstance(quant_config, QuantoInt8Config)
        and quant_config.group_size == GROUP_SIZE
        and quant_config.scale_dtype == torch.float32
        and quant_config.compute_dtype == torch.bfloat16
        and quant_config.layer_shapes is not None
    ):
        raise ValueError("Nomas supports only its official INT8 checkpoint format")


def configure_nomas_int8(model, quant_config):
    """Replace only declared linears before checkpoint materialization.

    The model is normally constructed on meta. Reconstructing its parallel
    linears retains their existing collectives and weight-loader contracts.
    """
    for name, shape in quant_config.layer_shapes.items():
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
            quant_config=quant_config,
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
                QuantoInt8ColumnParallelLinear
                if isinstance(original, ColumnParallelLinear)
                else QuantoInt8RowParallelLinear
            )
        elif type(original) is nn.Linear:
            cls = QuantoInt8Linear
        else:
            raise ValueError(f"Unsupported Nomas INT8 parallel layer: {name}")
        with torch.device(original.weight.device):
            replacement = cls(**kwargs)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, replacement)


def map_nomas_vae_checkpoint_key(name):
    """Map official PixelUMM FP8 names to the shared linear parameters."""
    for source, target in (
        (".fp8_weight", ".weight"),
        (".fp8_scale", ".weight_scale_inv"),
    ):
        if name.endswith(source):
            return name.removesuffix(source) + target
    return name
