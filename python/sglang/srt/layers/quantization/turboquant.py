"""
TurboQuant KV cache quantization config for SGLang.

Implements Google's TurboQuant (ICLR 2026) for KV cache compression.
TurboQuant compresses KV cache to 3-4 bits per coordinate with near-zero
accuracy loss using a data-oblivious approach requiring no calibration.

Usage:
    python -m sglang.launch_server --model-path <model> --kv-cache-dtype turboquant
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod

logger = logging.getLogger(__name__)

# Default TurboQuant settings
TURBOQUANT_DEFAULT_BITS = 4
TURBOQUANT_DEFAULT_MODE = "mse"  # "mse" or "prod"


class TurboQuantConfig(QuantizationConfig):
    """Config for TurboQuant KV cache quantization.

    TurboQuant is a KV-cache-only quantization method. It does not quantize
    model weights or activations — only the stored key/value cache entries.

    Args:
        bits: Number of bits per coordinate (1-4). Default 4.
        mode: "mse" for MSE-optimal quantization, "prod" for inner-product-optimal
              (adds 1-bit QJL on residual). Default "mse".
    """

    def __init__(
        self,
        bits: float = TURBOQUANT_DEFAULT_BITS,
        mode: str = TURBOQUANT_DEFAULT_MODE,
    ) -> None:
        super().__init__()
        allowed_bits = {1, 2, 2.5, 3, 3.5, 4}
        if bits not in allowed_bits:
            raise ValueError(
                f"TurboQuant bits must be one of {sorted(allowed_bits)}, got {bits}"
            )
        if mode not in ("mse", "prod"):
            raise ValueError(f"TurboQuant mode must be 'mse' or 'prod', got {mode}")
        self.bits = bits
        self.mode = mode

    def get_name(self) -> str:
        return "turboquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    def get_min_capability(self) -> int:
        # Requires SM80+ for Triton kernel support
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TurboQuantConfig":
        bits = cls.get_from_keys_or(config, ["bits", "turboquant_bits"], TURBOQUANT_DEFAULT_BITS)
        mode = cls.get_from_keys_or(config, ["mode", "turboquant_mode"], TURBOQUANT_DEFAULT_MODE)
        return cls(bits=bits, mode=mode)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.radix_attention import RadixAttention

        if isinstance(layer, RadixAttention):
            return TurboQuantKVCacheMethod(self)
        # TurboQuant only quantizes KV cache, not weights
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class TurboQuantKVCacheMethod(BaseKVCacheMethod):
    """KV cache method for TurboQuant.

    Manages the Hadamard transform state and quantization parameters
    for TurboQuant KV cache compression.
    """

    def __init__(self, quant_config: TurboQuantConfig):
        super().__init__(quant_config)
        self.bits = quant_config.bits
        self.mode = quant_config.mode

    def create_weights(self, layer: torch.nn.Module):
        """Create scale parameters (for compatibility with the base class)."""
        # TurboQuant doesn't use traditional k/v scales from checkpoints.
        # We still create them for interface compatibility but they default to 1.0.
        layer.k_scale = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32), requires_grad=False
        )
        layer.v_scale = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32), requires_grad=False
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        # TurboQuant uses fixed scales (1.0) since it handles normalization
        # internally via the Hadamard transform.
        layer.k_scale_float = 1.0
        layer.v_scale_float = 1.0
