# SPDX-License-Identifier: Apache-2.0
"""Checkpoint inspection for MiniMax-H3 transformer overrides."""

from typing import Any

import torch
from safetensors import safe_open

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    build_nvfp4_config_from_safetensors_list,
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
    mapping = get_param_names_mapping(MiniMaxH3DiTArchConfig().param_names_mapping)
    layer_markers = inspect_comfy_quant_markers(
        safetensors_list,
        param_name_mapper=lambda prefix: mapping(f"{prefix}.weight")[0].removesuffix(
            ".weight"
        ),
    )

    for path in safetensors_list:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if mapping(key)[0] != "adaln_t_table":
                    continue
                shape = tuple(checkpoint.get_slice(key).get_shape())
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
    safetensors_list: list[str] | None = None,
    param_names_mapping: dict | None = None,
    reverse_param_names_mapping: dict | None = None,
) -> QuantizationConfig | None:
    formats = {str(marker.get("format")) for marker in layer_markers.values()}
    if "nvfp4" in formats:
        unsupported = formats - {"nvfp4", "int8_tensorwise", "float8_e4m3fn"}
        if unsupported:
            raise NotImplementedError(
                "Unsupported Comfy NVFP4 companion format(s): "
                + ", ".join(sorted(unsupported))
            )
        if safetensors_list is None:
            raise ValueError("MiniMax-H3 NVFP4 metadata requires checkpoint files")
        config = build_nvfp4_config_from_safetensors_list(
            safetensors_list,
            param_names_mapping,
            reverse_param_names_mapping,
        )
        if not isinstance(config, ModelOptFp4Config):
            raise ValueError("Could not resolve MiniMax-H3 NVFP4 checkpoint layout")
        config.set_comfy_layer_markers(layer_markers)
        config.checkpoint_uses_comfy_quantization = True
        config.checkpoint_uses_native_qkv_layout = True
        config.checkpoint_weight_scale_layout = "swizzled"
        config.swap_weight_nibbles = True
        return config
    return resolve_comfy_checkpoint_quantization(layer_markers)


def validate_minimax_h3_checkpoint_variant(
    checkpoint_paths: list[str], selected_variant: str
) -> None:
    # hybrid is an explicit cross-partition override, not a filename heuristic
    if selected_variant.strip().lower() == "hybrid":
        return
    names = " ".join(path.lower() for path in checkpoint_paths)
    checkpoint_variants = {
        variant for variant in ("fl2va", "ref2va") if variant in names
    }
    if (
        len(checkpoint_variants) == 1
        and selected_variant.lower() not in checkpoint_variants
    ):
        (checkpoint_variant,) = checkpoint_variants
        raise ValueError(
            f"MiniMax-H3 checkpoint variant {checkpoint_variant!r} does not match "
            f"--model-variant {selected_variant!r}"
        )


def _qkv_class_separation(block_stats: torch.Tensor, labels: torch.Tensor) -> float:
    """Between-class / within-class variance of per-block statistics."""
    mean = block_stats.mean()
    between = within = 0.0
    for cls in range(3):
        values = block_stats[labels == cls]
        between += values.numel() * float((values.mean() - mean) ** 2)
        within += float(((values - values.mean()) ** 2).sum())
    return between / max(within, 1e-12)


def detect_minimax_h3_qkv_layout(
    path: str,
    num_heads: int,
    head_dim: int,
    min_margin: float = 10.0,
    min_separation: float = 0.02,
) -> str | None:
    """Tell the fused-QKV row layout of an unquantized single-file checkpoint.

    The official checkpoint interleaves rows per head (``[q_h, k_h, v_h]`` for
    each head), while Comfy-Org exports store ``[q_all, k_all, v_all]``; both
    layouts exist among community single files, and the tensor names and
    shapes are the same. Q, K and V rows differ in scale, so the mean row norm
    of each ``head_dim`` block separates cleanly into three classes under the
    true layout only (measured on block 0: about 0.2 vs below 0.02 for both the
    Comfy-Org BF16 export and the official transformer).

    Returns ``"native"``, ``"interleaved"``, or ``None`` when the checkpoint
    has no BF16/FP16/FP32 ``blocks.0`` QKV weight or the margin is too small.
    """
    mapping = get_param_names_mapping(MiniMaxH3DiTArchConfig().param_names_mapping)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        key = next(
            (
                k
                for k in checkpoint.keys()
                if mapping(k)[0] == "blocks.0.attn.qkv_proj.weight"
            ),
            None,
        )
        if key is None:
            return None
        weight = checkpoint.get_tensor(key)
    rows = 3 * num_heads * head_dim
    if (
        weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or weight.dim() != 2
        or weight.shape[0] != rows
    ):
        return None
    row_norms = torch.cat(
        [chunk.float().norm(dim=1) for chunk in weight.split(4096, dim=0)]
    )
    block_stats = row_norms.reshape(3 * num_heads, head_dim).mean(dim=1)
    index = torch.arange(3 * num_heads)
    native = _qkv_class_separation(block_stats, index // num_heads)
    interleaved = _qkv_class_separation(block_stats, index % 3)
    if native > min_separation and native > min_margin * interleaved:
        return "native"
    if interleaved > min_separation and interleaved > min_margin * native:
        return "interleaved"
    return None


__all__ = [
    "comfy_quant_key_filter",
    "detect_minimax_h3_qkv_layout",
    "inspect_minimax_h3_safetensors",
    "resolve_minimax_h3_checkpoint_quantization",
    "validate_minimax_h3_checkpoint_variant",
]
