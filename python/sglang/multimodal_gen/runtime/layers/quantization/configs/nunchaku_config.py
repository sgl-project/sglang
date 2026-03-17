# SPDX-License-Identifier: Apache-2.0
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import torch
from safetensors.torch import load_file as safetensors_load_file
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .base_config import QuantizationConfig, QuantizeMethodBase

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def is_nunchaku_available() -> bool:
    try:
        import nunchaku  # noqa

        logger.debug("Nunchaku package detected")
        return True
    except Exception:
        return False


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
        transformer_weights_path: Path to pre-quantized transformer weights (.safetensors)
        model_cls: DiT model class that provides quantization rules via get_nunchaku_quant_rules()
    """

    precision: str = "int4"
    rank: int = 32
    group_size: Optional[int] = None
    act_unsigned: bool = False
    transformer_weights_path: Optional[str] = None
    model_cls: Optional[type] = None

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
            transformer_weights_path=config.get("transformer_weights_path"),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if not isinstance(layer, LinearBase):
            return None

        # get quantization rules from model class
        quant_rules = self._get_quant_rules()

        # priority: skip > awq_w4a16 > svdq_w4a4 > default
        skip_patterns = quant_rules.get("skip", [])
        for pattern in skip_patterns:
            if pattern in prefix.lower():
                return None

        awq_patterns = quant_rules.get("awq_w4a16", [])
        for pattern in awq_patterns:
            if pattern in prefix:
                from ..nunchaku_linear import NunchakuAWQLinearMethod

                return NunchakuAWQLinearMethod(group_size=64)

        svdq_patterns = quant_rules.get("svdq_w4a4", [])
        for pattern in svdq_patterns:
            if pattern in prefix:
                from ..nunchaku_linear import NunchakuSVDQLinearMethod

                return NunchakuSVDQLinearMethod(
                    precision=self.precision,
                    rank=self.rank,
                    act_unsigned=self.act_unsigned,
                )

        # default: apply svdq_w4a4 to all remaining linear layers
        from ..nunchaku_linear import NunchakuSVDQLinearMethod

        return NunchakuSVDQLinearMethod(
            precision=self.precision,
            rank=self.rank,
            act_unsigned=self.act_unsigned,
        )

    def _get_quant_rules(self) -> dict[str, list[str]]:
        if self.model_cls is not None and hasattr(
            self.model_cls, "get_nunchaku_quant_rules"
        ):
            return self.model_cls.get_nunchaku_quant_rules()
        return {}

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
            "transformer_weights_path": self.transformer_weights_path,
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


def _patch_native_svdq_linear(
    module: nn.Module, tensor: Any, svdq_linear_cls: type
) -> bool:
    if (
        isinstance(module, svdq_linear_cls)
        and getattr(module, "wtscale", None) is not None
    ):
        module.wtscale = tensor
        return True
    return False


def _patch_sglang_svdq_linear(
    module: nn.Module, tensor: Any, svdq_method_cls: type
) -> bool:
    quant_method = getattr(module, "quant_method", None)
    if not isinstance(quant_method, svdq_method_cls):
        return False

    existing = getattr(module, "wtscale", None)
    if isinstance(existing, nn.Parameter):
        with torch.no_grad():
            existing.data.copy_(tensor.to(existing.data.dtype))
    else:
        module.wtscale = tensor

    # Keep alpha in sync (kernel reads `layer._nunchaku_alpha`)
    try:
        module._nunchaku_alpha = float(tensor.detach().cpu().item())
    except Exception:
        module._nunchaku_alpha = None
    return True


def _patch_sglang_svdq_wcscales(
    module: nn.Module, tensor: Any, svdq_method_cls: type
) -> bool:
    quant_method = getattr(module, "quant_method", None)
    if not isinstance(quant_method, svdq_method_cls):
        return False

    existing = getattr(module, "wcscales", None)
    if isinstance(existing, nn.Parameter):
        with torch.no_grad():
            existing.data.copy_(tensor.to(existing.data.dtype))
    else:
        module.wcscales = tensor
    return True


def _patch_nunchaku_scales(
    model: nn.Module,
    safetensors_list: list[str],
) -> None:
    """Patch transformer module with Nunchaku scale tensors from safetensors weights.

    For NVFP4 checkpoints, correctness depends on `wtscale` and attention
    `wcscales`. The FSDP loader may skip some of these metadata tensors.
    """

    if not safetensors_list:
        return

    if len(safetensors_list) != 1:
        logger.warning(
            "Nunchaku scale patch expects a single safetensors file, "
            "but got %d files. Skipping.",
            len(safetensors_list),
        )
        return

    from nunchaku.models.linear import SVDQW4A4Linear  # type: ignore[import]

    state_dict = safetensors_load_file(safetensors_list[0])
    if state_dict is None:
        return

    num_wtscale = 0
    num_wcscales = 0

    from ..nunchaku_linear import NunchakuSVDQLinearMethod

    for name, module in model.named_modules():
        wt = state_dict.get(f"{name}.wtscale")
        if wt is not None:
            if _patch_native_svdq_linear(module, wt, SVDQW4A4Linear):
                num_wtscale += 1
            elif _patch_sglang_svdq_linear(module, wt, NunchakuSVDQLinearMethod):
                num_wtscale += 1

        wc = state_dict.get(f"{name}.wcscales")
        if wc is not None:
            # Some modules may have wcscales as a direct attribute/Parameter.
            existing = getattr(module, "wcscales", None)
            if isinstance(existing, nn.Parameter):
                with torch.no_grad():
                    existing.data.copy_(wc.to(existing.data.dtype))
                num_wcscales += 1
            elif existing is not None:
                setattr(module, "wcscales", wc)
                num_wcscales += 1
            elif _patch_sglang_svdq_wcscales(module, wc, NunchakuSVDQLinearMethod):
                num_wcscales += 1

    if num_wtscale > 0:
        logger.info("Patched wtscale for %d layers", num_wtscale)
    if num_wcscales > 0:
        logger.info("Patched wcscales for %d layers", num_wcscales)
