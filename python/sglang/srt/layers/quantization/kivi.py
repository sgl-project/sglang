import logging
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)


class KIVIConfig(QuantizationConfig):
    def __init__(
        self,
        k_bits: int = 2,
        v_bits: int = 2,
        k_group_size: int = 32,
        v_group_size: int = 32,
        residual_length: int = 128,
    ) -> None:
        super().__init__()
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.k_group_size = k_group_size
        self.v_group_size = v_group_size
        self.residual_length = residual_length

    @classmethod
    def get_name(cls) -> str:
        return "kivi"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        # KIVI can be enabled directly by --quantization kivi.
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KIVIConfig":
        return cls(
            k_bits=config.get("k_bits", 2),
            v_bits=config.get("v_bits", 2),
            k_group_size=config.get("k_group_size", 32),
            v_group_size=config.get("v_group_size", 32),
            residual_length=config.get("residual_length", 128),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        del prefix
        if isinstance(layer, LinearBase):
            # Keep model weights unquantized; KIVI targets KV cache path.
            return UnquantizedLinearMethod()
        if isinstance(layer, RadixAttention):
            return KIVIKVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class KIVIKVCacheMethod(BaseKVCacheMethod):
    def __init__(self, quant_config: KIVIConfig):
        super().__init__(quant_config)
        logger.info("Using KIVI quantization integration for KV-cache hooks.")

    def create_weights(self, layer: torch.nn.Module):
        super().create_weights(layer)
        layer.kivi_runtime_enabled = True
        layer.kivi_k_bits = self.quant_config.k_bits
        layer.kivi_v_bits = self.quant_config.v_bits
        layer.kivi_k_group_size = self.quant_config.k_group_size
        layer.kivi_v_group_size = self.quant_config.v_group_size
        layer.kivi_residual_length = self.quant_config.residual_length
