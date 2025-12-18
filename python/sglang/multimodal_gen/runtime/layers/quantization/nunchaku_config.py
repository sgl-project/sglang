# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Optional

import torch
import json
import os

from .base_config import QuantizationConfig, QuantizeMethodBase

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

    Attributes:
        precision: Quantization precision type. Options:
            - "int4": Standard INT4 quantization
            - "nvfp4": FP4 quantization 
        rank: SVD low-rank dimension for absorbing outliers
        group_size: Quantization group size (automatically set based on precision)
        act_unsigned: Use unsigned activation quantization 
        quantized_model_path: Path to pre-quantized model weights (.safetensors)
        enable_offloading: Enable CPU offloading for low memory
    """

    precision: str = "int4"  # "int4" or "nvfp4"
    rank: int = 32
    group_size: Optional[int] = None
    act_unsigned: bool = False
    quantized_model_path: Optional[str] = None
    enable_offloading: bool = False

    @classmethod
    def get_name(cls) -> str:
        return "svdquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json", "quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NunchakuConfig":

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
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None

        for pattern in SKIP_QUANTIZATION_PATTERNS:
            if pattern in prefix.lower():
                return None

        for pattern in SVDQ_W4A4_LAYER_PATTERNS:
            if pattern in prefix:
                from .nunchaku_linear import NunchakuSVDQLinearMethod
                return NunchakuSVDQLinearMethod(
                    precision=self.precision,
                    rank=self.rank,
                    act_unsigned=self.act_unsigned,
                )

        for pattern in AWQ_W4A16_LAYER_PATTERNS:
            if pattern in prefix:
                from .nunchaku_linear import NunchakuAWQLinearMethod
                return NunchakuAWQLinearMethod(
                    group_size=64,
                )

        from .nunchaku_linear import NunchakuSVDQLinearMethod
        return NunchakuSVDQLinearMethod(
            precision=self.precision,
            rank=self.rank,
            act_unsigned=self.act_unsigned,
        )

    def __post_init__(self):
        if self.group_size is None:
            if self.precision == "nvfp4":
                self.group_size = 16
            elif self.precision == "int4":
                self.group_size = 64
            else:
                raise ValueError(
                    f"Invalid precision: {self.precision}. Must be 'int4' or 'nvfp4'"
                )

        if self.precision not in ["int4", "nvfp4"]:
            raise ValueError(
                f"Invalid precision: {self.precision}. Must be 'int4' or 'nvfp4'"
            )

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
        for filename in cls.get_config_filenames():
            config_path = os.path.join(model_path, filename)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                if config_dict.get("quant_method") == cls.get_name():
                    return cls.from_config(config_dict)
        return None

