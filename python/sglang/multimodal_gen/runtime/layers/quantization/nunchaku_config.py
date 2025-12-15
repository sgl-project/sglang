# SPDX-License-Identifier: Apache-2.0
"""
Nunchaku quantization configuration for SVDQuant integration.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .base_config import QuantizationConfig, QuantizeMethodBase


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
        processor: Attention processor type ("flashattn2" or "nunchaku-fp16")
        enable_offloading: Enable CPU offloading for low memory
    """

    precision: str = "int4"  # "int4" or "nvfp4"
    rank: int = 32
    group_size: Optional[int] = None
    act_unsigned: bool = False
    quantized_model_path: Optional[str] = None
    processor: str = "flashattn2"  # "flashattn2" or "nunchaku-fp16"
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
            processor=config.get("processor", "flashattn2"),
            enable_offloading=bool(config.get("enable_offloading", False)),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """
        Return the per-layer quantization method.

        For Nunchaku SVDQuant we delegate quantization to the Nunchaku models
        themselves (via `nunchaku.*` Transformer2DModel classes), so we do not
        apply any additional vLLM-style per-layer quantization here.
        """
        return None

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

        # Validate processor
        if self.processor not in ["flashattn2", "nunchaku-fp16"]:
            raise ValueError(
                f"Invalid processor: {self.processor}. "
                "Must be 'flashattn2' or 'nunchaku-fp16'"
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
            "processor": self.processor,
            "enable_offloading": self.enable_offloading,
        }

    def get_nunchaku_kwargs(self) -> dict:
        """Get keyword arguments for Nunchaku model initialization."""
        return {
            "precision": self.precision,
            "rank": self.rank,
            "processor": self.processor,
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

