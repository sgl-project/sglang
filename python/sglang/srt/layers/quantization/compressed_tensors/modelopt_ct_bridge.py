# SPDX-License-Identifier: Apache-2.0
"""
ModelOpt-to-CompressedTensors bridge for unified quantization support.

Maps ModelOpt quantization recipes to existing CompressedTensors schemes so that
ModelOpt checkpoints can be loaded and run through the same kernels as
compressed-tensors (e.g. FP8 per-channel per-token -> CompressedTensorsW8A8Fp8).

See: Unified ModelOpt Quantization Support in vLLM/SGLang design doc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy, QuantizationType

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.modelopt_scheme import (
    ModelOptQuantizationScheme,
    detect_modelopt_quantization_scheme,
)

logger = logging.getLogger(__name__)

# Backward compatibility: string constant for FP8 per-channel per-token recipe
FP8_PER_CHANNEL_PER_TOKEN_CFG = (
    ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG.value
)


def is_modelopt_fp8_per_channel_per_token_config(config: Dict[str, Any]) -> bool:
    """
    Return True if the ModelOpt config describes FP8 per-channel per-token recipe.

    Delegates to detect_modelopt_quantization_scheme for consistency.
    """
    return (
        detect_modelopt_quantization_scheme(config)
        == ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG
    )


def _get_quant_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get quantization section from flat or nested config."""
    if "quantization" in config:
        return config["quantization"]
    return config


def modelopt_config_to_compressed_tensors_config(
    config: Dict[str, Any],
    packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
) -> CompressedTensorsConfig:
    """
    Build a CompressedTensorsConfig that matches the ModelOpt FP8 per-channel per-token recipe.

    The resulting config will use CompressedTensorsW8A8Fp8 with:
    - weight strategy: channel (per-channel scales)
    - input activation: dynamic per-token (is_static_input_scheme=False)

    :param config: Full ModelOpt config dict (from hf_quant_config.json or config.json).
    :param packed_modules_mapping: Optional mapping of fused layer names to shard names.
    :return: CompressedTensorsConfig instance suitable for get_quant_method / create_weights.
    """
    if (
        detect_modelopt_quantization_scheme(config)
        != ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG
    ):
        raise ValueError(
            "Config is not a ModelOpt FP8 per-channel per-token config. "
            f"Expected {ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG.value}."
        )

    quant = _get_quant_section(config)
    ignore: List[str] = list(quant.get("ignore") or quant.get("exclude_modules") or [])

    # CompressedTensors format that supports activation quantization (W8A8).
    quant_format = CompressionFormat.float_quantized.value

    # Single config group: all linear layers get the same scheme.
    # Weights: FP8, per-channel, static. Activations: FP8, per-token, dynamic.
    weights_args = {
        "type": QuantizationType.FLOAT.value,
        "num_bits": 8,
        "strategy": QuantizationStrategy.CHANNEL.value,
        "symmetric": True,
        "dynamic": False,
    }
    input_activations_args = {
        "type": QuantizationType.FLOAT.value,
        "num_bits": 8,
        "strategy": QuantizationStrategy.TOKEN.value,
        "symmetric": True,
        "dynamic": True,
    }

    ct_config_dict: Dict[str, Any] = {
        "format": quant_format,
        "ignore": ignore,
        "config_groups": {
            "default": {
                "weights": weights_args,
                "input_activations": input_activations_args,
                "targets": ["re:.*"],
            }
        },
        "packed_modules_mapping": packed_modules_mapping or {},
        "_modelopt_bridge": True,
    }

    logger.info(
        "ModelOpt->CT bridge: using CompressedTensorsW8A8Fp8 (channel, dynamic per-token) for %s",
        ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG.value,
    )
    return CompressedTensorsConfig.from_config(ct_config_dict)


# --- E2: Weight loader adapter (name mapping) ----------------------------------
# ModelOpt export may use the same param names as CT (weight, weight_scale, input_scale).
# If a checkpoint uses different names, remap them before loading.

# Optional: mapping from ModelOpt checkpoint key suffix -> SGLang/CT param suffix.
# Example: {"scale": "weight_scale"} if ModelOpt used "scale" for weight scale.
MODELOPT_TO_CT_WEIGHT_KEY_SUFFIXES: Dict[str, str] = {
    # Add entries if ModelOpt uses different suffixes, e.g. "scale" -> "weight_scale"
}


def remap_modelopt_state_dict_keys_for_ct(
    state_dict: Dict[str, Any],
    key_suffix_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Remap state dict keys from ModelOpt naming to CompressedTensors/SGLang naming.

    Used when loading a ModelOpt FP8 per-channel checkpoint via the CT bridge so that
    parameter names match what CompressedTensorsW8A8Fp8 layers expect (weight,
    weight_scale, input_scale).

    :param state_dict: Checkpoint state dict (name -> tensor).
    :param key_suffix_map: Optional override; defaults to MODELOPT_TO_CT_WEIGHT_KEY_SUFFIXES.
    :return: New dict with keys remapped (no copy of tensors).
    """
    if key_suffix_map is None:
        key_suffix_map = MODELOPT_TO_CT_WEIGHT_KEY_SUFFIXES
    if not key_suffix_map:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for suffix, new_suffix in key_suffix_map.items():
            if k.endswith(suffix):
                new_k = k[: -len(suffix)] + new_suffix
                break
        out[new_k] = v
    return out
