# SPDX-License-Identifier: Apache-2.0
"""
ModelOpt quantization scheme identifiers and detection.

Defines the set of ModelOpt recipes (FP8_DEFAULT_CFG, FP8_PER_CHANNEL_PER_TOKEN_CFG,
INT4_AWQ_CFG, W4A8_AWQ_BETA_CFG, NVFP4_DEFAULT_CFG, etc.) and logic to detect
which scheme a given ModelOpt config describes. Used by config parsing and
the ModelOpt->CompressedTensors bridge.

See: Unified ModelOpt Quantization Support in vLLM/SGLang design doc.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ModelOptQuantizationScheme(str, Enum):
    """
    ModelOpt quantization recipe identifiers.

    Names align with ModelOpt export config (quant_cfg / recipe) and
    QUANT_CFG_CHOICES in modelopt_utils. Schemes marked with uses_ct_bridge
    are loaded via CompressedTensorsConfig; others use native ModelOpt configs.
    """

    # FP8: existing native (per-tensor) and CT-bridge (per-channel per-token)
    FP8_DEFAULT_CFG = "FP8_DEFAULT_CFG"
    FP8_PER_CHANNEL_PER_TOKEN_CFG = "FP8_PER_CHANNEL_PER_TOKEN_CFG"

    # INT8 (future CT bridge)
    INT8_DEFAULT_CFG = "INT8_DEFAULT_CFG"
    INT8_SMOOTHQUANT_CFG = "INT8_SMOOTHQUANT_CFG"

    # INT4 / W4A8 (future CT bridge; you will implement)
    INT4_AWQ_CFG = "INT4_AWQ_CFG"
    W4A8_AWQ_BETA_CFG = "W4A8_AWQ_BETA_CFG"

    # FP4: existing native
    NVFP4_DEFAULT_CFG = "NVFP4_DEFAULT_CFG"
    NVFP4_AWQ_LITE_CFG = "NVFP4_AWQ_LITE_CFG"

    # Legacy / other
    MIXED_PRECISION = "MIXED_PRECISION"
    UNKNOWN = "UNKNOWN"

    def uses_ct_bridge(self) -> bool:
        """True if this scheme is served via CompressedTensors (CT) bridge."""
        return self in (
            ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG,
            ModelOptQuantizationScheme.INT8_DEFAULT_CFG,
            ModelOptQuantizationScheme.INT8_SMOOTHQUANT_CFG,
            ModelOptQuantizationScheme.INT4_AWQ_CFG,
            ModelOptQuantizationScheme.W4A8_AWQ_BETA_CFG,
        )

    def quant_method(self) -> str:
        """SGLang quantization method string (modelopt_fp8, modelopt_fp4, etc.)."""
        if self in (
            ModelOptQuantizationScheme.FP8_DEFAULT_CFG,
            ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG,
            ModelOptQuantizationScheme.INT8_DEFAULT_CFG,
            ModelOptQuantizationScheme.INT8_SMOOTHQUANT_CFG,
        ):
            return "modelopt_fp8"
        if self in (
            ModelOptQuantizationScheme.NVFP4_DEFAULT_CFG,
            ModelOptQuantizationScheme.NVFP4_AWQ_LITE_CFG,
            ModelOptQuantizationScheme.INT4_AWQ_CFG,
        ):
            return "modelopt_fp4"
        if self == ModelOptQuantizationScheme.W4A8_AWQ_BETA_CFG:
            return "modelopt_fp8"  # or dedicated w4a8 method when added
        if self == ModelOptQuantizationScheme.MIXED_PRECISION:
            return "w4afp8"
        return "modelopt_fp8"


# Keys in ModelOpt config for recipe/cfg name
QUANT_CFG_KEY = "quant_cfg"
QUANT_ALGO_KEY = "quant_algo"
RECIPE_KEY = "recipe"


def _get_quant_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get quantization section from flat or nested config."""
    if "quantization" in config:
        return config["quantization"]
    return config


def _is_modelopt_producer(config: Dict[str, Any]) -> bool:
    """Return True if producer.name indicates modelopt."""
    producer = config.get("producer") or {}
    return str(producer.get("name", "")).lower().startswith("modelopt")


def detect_modelopt_quantization_scheme(
    config: Dict[str, Any],
    *,
    require_modelopt_producer: bool = False,
) -> Optional[ModelOptQuantizationScheme]:
    """
    Detect which ModelOpt quantization scheme a config describes.

    :param config: Full ModelOpt config (e.g. from hf_quant_config.json or config.json).
    :param require_modelopt_producer: If True, return None unless producer.name is modelopt.
    :return: Detected scheme, or None if not a recognized ModelOpt config.
    """
    quant = _get_quant_section(config)

    # Explicit recipe / quant_cfg takes precedence
    quant_cfg = quant.get(QUANT_CFG_KEY) or quant.get(RECIPE_KEY)
    if quant_cfg is not None:
        quant_cfg_str = str(quant_cfg).strip().upper()
        if require_modelopt_producer and not _is_modelopt_producer(config):
            return None
        for scheme in ModelOptQuantizationScheme:
            if scheme == ModelOptQuantizationScheme.UNKNOWN:
                continue
            if scheme.value == quant_cfg_str:
                return scheme
        return None

    # Fall back to quant_algo
    if require_modelopt_producer and not _is_modelopt_producer(config):
        return None

    quant_algo = (quant.get(QUANT_ALGO_KEY) or "").upper()

    if quant_algo == "MIXED_PRECISION":
        return ModelOptQuantizationScheme.MIXED_PRECISION
    if "NVFP4" in quant_algo or "FP4" in quant_algo:
        if "AWQ" in quant_algo:
            return ModelOptQuantizationScheme.NVFP4_AWQ_LITE_CFG
        return ModelOptQuantizationScheme.NVFP4_DEFAULT_CFG
    if "FP8" in quant_algo:
        if "PER_CHANNEL" in quant_algo and "PER_TOKEN" in quant_algo:
            return ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG
        return ModelOptQuantizationScheme.FP8_DEFAULT_CFG
    if "INT4" in quant_algo and "AWQ" in quant_algo:
        return ModelOptQuantizationScheme.INT4_AWQ_CFG
    if "W4A8" in quant_algo and "AWQ" in quant_algo:
        return ModelOptQuantizationScheme.W4A8_AWQ_BETA_CFG
    if "INT8" in quant_algo:
        if "SMOOTHQUANT" in quant_algo or "SMOOTH_QUANT" in quant_algo:
            return ModelOptQuantizationScheme.INT8_SMOOTHQUANT_CFG
        return ModelOptQuantizationScheme.INT8_DEFAULT_CFG

    return None
