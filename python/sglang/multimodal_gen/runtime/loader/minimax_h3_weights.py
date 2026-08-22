# SPDX-License-Identifier: Apache-2.0
"""Checkpoint inspection for MiniMax-H3 transformer overrides."""

from typing import Any

from safetensors import safe_open

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    inspect_comfy_quant_markers,
    resolve_comfy_checkpoint_quantization,
)


def comfy_quant_key_filter(name: str) -> bool:
    return not name.endswith(".comfy_quant")


def inspect_minimax_h3_safetensors(
    safetensors_list: list[str],
) -> tuple[tuple[int, int] | None, dict[str, dict[str, Any]]]:
    """Read H3 architecture metadata and Comfy per-layer format markers."""
    adaln_curve_shape = None
    layer_markers = inspect_comfy_quant_markers(safetensors_list)

    for path in safetensors_list:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = checkpoint.keys()
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

    return adaln_curve_shape, layer_markers


def resolve_minimax_h3_checkpoint_quantization(
    layer_markers: dict[str, dict[str, Any]],
) -> QuantizationConfig | None:
    return resolve_comfy_checkpoint_quantization(layer_markers)


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
