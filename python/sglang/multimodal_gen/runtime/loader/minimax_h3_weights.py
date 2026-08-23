# SPDX-License-Identifier: Apache-2.0
"""Checkpoint inspection for MiniMax-H3 transformer overrides."""

import json
from typing import Any

from safetensors import safe_open

from sglang.multimodal_gen.runtime.layers.quantization.comfy_fp8 import ComfyFp8Config
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)


def comfy_quant_key_filter(name: str) -> bool:
    return not name.endswith(".comfy_quant")


def inspect_minimax_h3_safetensors(
    safetensors_list: list[str],
) -> tuple[tuple[int, int] | None, dict[str, dict[str, Any]]]:
    """Read H3 architecture metadata and Comfy per-layer format markers."""
    adaln_curve_shape = None
    layer_markers: dict[str, dict[str, Any]] = {}
    checkpoint_keys: set[str] = set()
    checkpoint_meta: dict[str, tuple[str, tuple[int, ...]]] = {}
    fp8_weight_prefixes: set[str] = set()

    for path in safetensors_list:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = checkpoint.keys()
            checkpoint_keys.update(keys)
            if "adaln_t_table" in keys:
                shape = tuple(checkpoint.get_slice("adaln_t_table").get_shape())
                if len(shape) != 2 or shape[0] < 2:
                    raise ValueError(
                        "MiniMax-H3 adaln_t_table must have shape [N, D] with "
                        f"N >= 2, got {shape} in {path}"
                    )
                if adaln_curve_shape is not None and adaln_curve_shape != shape:
                    raise ValueError(
                        "MiniMax-H3 checkpoint shards disagree on adaln_t_table "
                        f"shape: {adaln_curve_shape} vs {shape}"
                    )
                adaln_curve_shape = shape

            for key in keys:
                if key.endswith((".weight", ".weight_scale")):
                    tensor_slice = checkpoint.get_slice(key)
                    dtype = tensor_slice.get_dtype()
                    checkpoint_meta[key] = (
                        dtype,
                        tuple(tensor_slice.get_shape()),
                    )
                    if key.endswith(".weight") and dtype == "F8_E4M3":
                        fp8_weight_prefixes.add(key.removesuffix(".weight"))
                if not key.endswith(".comfy_quant"):
                    continue
                try:
                    marker = json.loads(checkpoint.get_tensor(key).numpy().tobytes())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"Invalid Comfy quantization marker {key!r} in {path}"
                    ) from exc
                if not isinstance(marker, dict):
                    raise ValueError(
                        f"Comfy quantization marker {key!r} must contain a JSON object"
                    )
                prefix = key.removesuffix(".comfy_quant")
                previous = layer_markers.get(prefix)
                if previous is not None and previous != marker:
                    raise ValueError(
                        f"Conflicting Comfy quantization markers for {prefix!r}"
                    )
                layer_markers[prefix] = marker

    if layer_markers:
        missing_markers = fp8_weight_prefixes - layer_markers.keys()
        if missing_markers:
            raise ValueError(
                "MiniMax-H3 FP8 weights are missing comfy_quant metadata: "
                f"{sorted(missing_markers)[:5]}"
            )

    for prefix, marker in layer_markers.items():
        marker_format = marker.get("format")
        required = {f"{prefix}.weight", f"{prefix}.weight_scale"}
        if marker_format == "float8_e4m3fn" and not marker.get(
            "full_precision_matrix_mult", False
        ):
            required.add(f"{prefix}.input_scale")
        if marker_format not in ("float8_e4m3fn", "int8_tensorwise"):
            continue
        missing = required - checkpoint_keys
        if missing:
            raise ValueError(
                f"MiniMax-H3 Comfy layer {prefix!r} is missing checkpoint "
                f"tensors: {sorted(missing)}"
            )
        if marker_format == "int8_tensorwise":
            weight_dtype, weight_shape = checkpoint_meta[f"{prefix}.weight"]
            scale_dtype, scale_shape = checkpoint_meta[f"{prefix}.weight_scale"]
            if weight_dtype != "I8" or scale_dtype != "F32":
                raise ValueError(
                    f"MiniMax-H3 Comfy INT8 layer {prefix!r} needs I8 weights "
                    f"and F32 scales, got {weight_dtype} and {scale_dtype}"
                )
            if len(weight_shape) != 2:
                raise ValueError(
                    f"MiniMax-H3 Comfy INT8 layer {prefix!r} needs a 2D weight, "
                    f"got {weight_shape}"
                )
            expected_scale_shape = (weight_shape[0], 1)
            if scale_shape != expected_scale_shape:
                raise ValueError(
                    f"MiniMax-H3 Comfy INT8 layer {prefix!r} needs scale shape "
                    f"{expected_scale_shape}, got {scale_shape}"
                )

    return adaln_curve_shape, layer_markers


def resolve_minimax_h3_checkpoint_quantization(
    layer_markers: dict[str, dict[str, Any]],
) -> QuantizationConfig | None:
    if not layer_markers:
        return None

    formats = sorted({str(marker.get("format")) for marker in layer_markers.values()})
    if formats == ["int8_tensorwise"]:
        return KitchenInt8Config(layer_markers=layer_markers)
    if formats == ["float8_e4m3fn"]:
        return ComfyFp8Config(layer_markers)
    raise NotImplementedError(
        "Unsupported MiniMax-H3 Comfy quantization format(s): " + ", ".join(formats)
    )


def validate_minimax_h3_checkpoint_variant(
    checkpoint_paths: list[str], selected_variant: str
) -> None:
    names = " ".join(path.lower() for path in checkpoint_paths)
    checkpoint_variant = next(
        (variant for variant in ("fl2va", "ref2va") if variant in names), None
    )
    if (
        checkpoint_variant is not None
        and checkpoint_variant != selected_variant.lower()
    ):
        raise ValueError(
            f"MiniMax-H3 checkpoint variant {checkpoint_variant!r} does not match "
            f"--model-variant {selected_variant!r}"
        )


__all__ = [
    "comfy_quant_key_filter",
    "inspect_minimax_h3_safetensors",
    "resolve_minimax_h3_checkpoint_quantization",
    "validate_minimax_h3_checkpoint_variant",
]
