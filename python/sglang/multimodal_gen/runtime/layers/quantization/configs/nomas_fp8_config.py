# SPDX-License-Identifier: Apache-2.0
"""Checkpoint admission and layer configuration for PixelUMM FP8 weights."""

import json

import torch
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.runtime.layers.quantization.nomas_fp8 import NomasFP8Linear

CHECKPOINT_FORMAT = "pixelumm_fp8_bf16_single_v1"


def inspect_nomas_fp8_checkpoint(weight_files):
    """Admit the complete official single-file format before allocating weights."""
    for path in weight_files:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            metadata = reader.metadata() or {}
            if metadata.get("format") != CHECKPOINT_FORMAT:
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
    for name in shapes:
        linear = model.get_submodule(name)
        parent, _, leaf = name.rpartition(".")
        model.get_submodule(parent).add_module(
            leaf,
            NomasFP8Linear(
                linear.in_features, linear.out_features, device=linear.weight.device
            ),
        )
