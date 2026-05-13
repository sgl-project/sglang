"""Marker config for MLX backend on-the-fly quantization (mlx_q4 / mlx_q8).

The MLX backend (``python/sglang/srt/hardware_backend/mlx/``) performs its own
quantization at model-load time via :func:`mlx_lm.utils.quantize_model`. The
standard PyTorch ``QuantizationConfig`` machinery is **never** invoked on that
path.

This module exists purely so that the names ``mlx_q4`` and ``mlx_q8`` are
recognized by ``QUANTIZATION_METHODS`` — that way
:meth:`ModelConfig._verify_quantization` and downstream registry lookups treat
them as known methods without any backend-specific carve-outs in the generic
config code.

If a user passes ``--quantization mlx_q4`` without ``SGLANG_USE_MLX=1`` they
will eventually reach a code path that tries to instantiate this Config class,
at which point we raise a clear error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


class MlxQuantizationConfig(QuantizationConfig):
    """Marker config for MLX backend on-the-fly quantization presets.

    Not a real quantization config — the MLX backend handles quantization
    itself. Any standard-PyTorch-path method that touches this class raises
    a helpful error pointing the user at ``SGLANG_USE_MLX=1``.
    """

    _ERR = (
        "MLX on-the-fly quantization (--quantization mlx_q4 / mlx_q8) is "
        "handled by the MLX backend at model-load time via mlx_lm.utils."
        "quantize_model, not by this QuantizationConfig class. If you "
        "reached this error, SGLANG_USE_MLX=1 is likely not set."
    )

    def __init__(self, preset: str):
        super().__init__()
        self.preset = preset

    @classmethod
    def get_name(cls) -> str:
        return "mlx"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return []

    @classmethod
    def get_min_capability(cls) -> int:
        # Capability check is for NVIDIA SM versions; not meaningful for MLX.
        return 0

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MlxQuantizationConfig":
        raise NotImplementedError(cls._ERR)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        raise NotImplementedError(self._ERR)
