# SPDX-License-Identifier: Apache-2.0
"""
Nunchaku quantization configuration for SVDQuant integration.

This module provides AWQ-style quantization config that integrates with
the LinearMethodBase pattern used in SGLang.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from .base_config import QuantizationConfig, QuantizeMethodBase


# Layer prefixes that use different quantization methods
# These patterns are based on Nunchaku's transformer implementation
SVDQ_W4A4_LAYER_PATTERNS = [
    "attn.to_qkv",
    "attn.to_out",
    "attn.add_qkv_proj",
    "attn.to_add_out",
    "img_mlp",
    "txt_mlp",
]

AWQ_W4A16_LAYER_PATTERNS = [
    "img_mod",
    "txt_mod",
]

# Layers to skip quantization
SKIP_QUANTIZATION_PATTERNS = [
    "norm",
    "embed",
    "rotary",
    "pos_embed",
]


@dataclass
class NunchakuConfig(QuantizationConfig):
    """
    Configuration for Nunchaku (SVDQuant) W4A4-style quantization.

    SVDQuant uses 4-bit weights and 4-bit activations with low-rank decomposition
    to maintain quality while achieving 3-4x speedup and ~3.6x memory reduction.

    Attributes:
        precision: Quantization precision type. Options:
            - "int4": Standard INT4 quantization (compatible with most NVIDIA GPUs)
            - "nvfp4": FP4 quantization for RTX 50 series GPUs
        rank: SVD low-rank dimension for absorbing outliers (default: 32)
        group_size: Quantization group size (automatically set based on precision)
        act_unsigned: Use unsigned activation quantization (int4 only)
        quantized_model_path: Path to pre-quantized model weights (.safetensors)
        enable_offloading: Enable CPU offloading for low memory
    """

    precision: str = "int4"  # "int4" or "nvfp4"
    rank: int = 32
    group_size: Optional[int] = None
    act_unsigned: bool = False
    quantized_model_path: Optional[str] = None
    enable_offloading: bool = False

    # ---- QuantizationConfig interface -------------------------------------------------
    @classmethod
    def get_name(cls) -> str:
        # Name used in quantization config files, e.g. `"quant_method": "svdquant"`.
        return "svdquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # Nunchaku models are designed for BF16/FP16 activations.
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # SVDQuant kernels target modern NVIDIA GPUs (Volta / Turing / Ampere+).
        # Using 70 keeps this consistent with other weight-quant backends.
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        """Get possible quantization config filenames."""
        return ["quantization_config.json", "quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NunchakuConfig":
        """
        Create configuration from a generic quantization config dict.

        This mirrors how SGLang LLM backends construct QuantizationConfig objects
        from serialized metadata.
        """
        return cls(
            precision=config.get("precision", "int4"),
            rank=int(config.get("rank", 32)),
            group_size=config.get("group_size"),
            act_unsigned=bool(config.get("act_unsigned", False)),
            quantized_model_path=config.get("quantized_model_path"),
            enable_offloading=bool(config.get("enable_offloading", False)),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """
        Return the per-layer quantization method based on layer type and prefix.

        This follows the SGLang AWQ pattern where different layers can use
        different quantization methods.

        Args:
            layer: The layer to get quantization method for.
            prefix: The full name of the layer in the state dict.

        Returns:
            - NunchakuSVDQLinearMethod for attention and MLP layers (W4A4)
            - NunchakuAWQLinearMethod for modulation layers (W4A16)
            - None for layers that should not be quantized
        """
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        # Only quantize linear layers
        if not isinstance(layer, LinearBase):
            return None

        # Skip layers that shouldn't be quantized
        for pattern in SKIP_QUANTIZATION_PATTERNS:
            if pattern in prefix.lower():
                return None

        # Check if this layer should use SVDQ W4A4
        for pattern in SVDQ_W4A4_LAYER_PATTERNS:
            if pattern in prefix:
                from .nunchaku_linear import NunchakuSVDQLinearMethod
                return NunchakuSVDQLinearMethod(
                    precision=self.precision,
                    rank=self.rank,
                    act_unsigned=self.act_unsigned,
                )

        # Check if this layer should use AWQ W4A16
        for pattern in AWQ_W4A16_LAYER_PATTERNS:
            if pattern in prefix:
                from .nunchaku_linear import NunchakuAWQLinearMethod
                return NunchakuAWQLinearMethod(
                    # AWQ W4A16 in Nunchaku uses group_size=64 regardless of the
                    # SVDQ precision/group_size used for W4A4. The checkpoints for
                    # Qwen-Image are produced with group_size=64, so we hardcode
                    # this here to ensure shapes match (e.g. wzeros.shape[0] =
                    # in_features // 64).
                    group_size=64,
                )

        # Default: use SVDQ W4A4 for other linear layers in the transformer
        from .nunchaku_linear import NunchakuSVDQLinearMethod
        return NunchakuSVDQLinearMethod(
            precision=self.precision,
            rank=self.rank,
            act_unsigned=self.act_unsigned,
        )

    # ---- Nunchaku-specific helpers ---------------------------------------------------
    def __post_init__(self):
        """Set group_size based on precision if not specified and validate fields."""
        if self.group_size is None:
            if self.precision == "nvfp4":
                self.group_size = 16
            elif self.precision == "int4":
                self.group_size = 64
            else:
                raise ValueError(
                    f"Invalid precision: {self.precision}. Must be 'int4' or 'nvfp4'"
                )

        # Validate precision
        if self.precision not in ["int4", "nvfp4"]:
            raise ValueError(
                f"Invalid precision: {self.precision}. Must be 'int4' or 'nvfp4'"
            )

        # Validate rank
        if self.rank <= 0:
            raise ValueError(f"Rank must be positive, got {self.rank}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "NunchakuConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "precision": self.precision,
            "rank": self.rank,
            "group_size": self.group_size,
            "act_unsigned": self.act_unsigned,
            "quantized_model_path": self.quantized_model_path,
            "enable_offloading": self.enable_offloading,
        }

    @classmethod
    def from_pretrained(cls, model_path: str) -> Optional["NunchakuConfig"]:
        """
        Load quantization config from model directory.

        Args:
            model_path: Path to model directory

        Returns:
            NunchakuConfig if found, None otherwise
        """
        import json
        import os

        for filename in cls.get_config_filenames():
            config_path = os.path.join(model_path, filename)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                # Only construct if the config explicitly marks SVDQuant.
                if config_dict.get("quant_method") == cls.get_name():
                    return cls.from_config(config_dict)
        return None

