"""Marker config and auto-detect hook for MLX backend quantization presets.

The MLX backend (``python/sglang/srt/hardware_backend/mlx/``) performs its own
quantization at model load time via :func:`mlx_lm.utils.quantize_model`. The
standard PyTorch ``QuantizationConfig`` machinery is never invoked on that
path.

This module serves two purposes:

1. Registry registration. Listing ``mlx_q4`` and ``mlx_q8`` in
   ``QUANTIZATION_METHODS`` lets :meth:`ModelConfig._verify_quantization`
   recognize them as known methods without backend-specific exceptions in
   the generic config code.

2. Auto-detection for mlx-community HF repos.
   :meth:`override_quantization_method` claims ``config.json`` blocks of
   the form ``{"group_size": <int>, "bits": <int>}`` with no
   ``quant_method`` key and resolves them to the matching preset.
   Already-quantized mlx-community repos load on Apple Silicon without the
   user passing ``--quantization`` on the CLI. Resolves #25119.

The PyTorch path constructors (``from_config``, ``get_quant_method``) raise
``NotImplementedError`` with a clear pointer to ``SGLANG_USE_MLX=1``, since
this class is not a real PyTorch quantization implementation.
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

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        """Auto-detect mlx-community-shape quantization configs.

        mlx-community models ship ``config.json`` with::

            "quantization_config": {"group_size": <int>, "bits": <int>}

        No ``quant_method`` key, no other identifying field. Without this
        override, :meth:`ModelConfig._verify_quantization` cannot match the
        shape to any registered method and raises ``Unknown quantization
        method`` (see #25119). Match it here and return the preset whose
        bit-width agrees, so pre-quantized HF repos load on Apple Silicon
        without the user having to pass ``--quantization`` on the CLI.

        Returns ``None`` for any input that does not look like a bare MLX
        preset: non-dict, dict with an explicit ``quant_method``, missing
        keys, non-integer values, or unsupported bit-width. Also defers to
        any explicit ``--quantization`` CLI choice (``user_quant``) per the
        registry contract: CLI selection takes priority over auto-detect.
        """
        if user_quant is not None:
            # User passed --quantization explicitly; respect that choice
            # regardless of the HF config shape. Matches the moe_wna16 /
            # modelopt convention.
            return None
        if not isinstance(hf_quant_cfg, dict):
            return None
        if "quant_method" in hf_quant_cfg:
            # Configs that declare a quant_method belong to whichever method
            # registers under that name; do not hijack them.
            return None
        bits = hf_quant_cfg.get("bits")
        group_size = hf_quant_cfg.get("group_size")
        if not isinstance(bits, int) or not isinstance(group_size, int):
            return None
        if bits == 4:
            return "mlx_q4"
        if bits == 8:
            return "mlx_q8"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        raise NotImplementedError(self._ERR)
