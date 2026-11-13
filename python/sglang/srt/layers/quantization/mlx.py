"""Marker config and auto-detect hook for MLX backend quantization presets.

The MLX backend (``python/sglang/srt/hardware_backend/mlx/``) performs its own
quantization at model load time via :func:`mlx_lm.utils.quantize_model`. The
standard PyTorch ``QuantizationConfig`` machinery is never invoked on that
path.

This module serves two purposes:

1. Registry registration. Listing ``mlx``, ``mlx_q4``, and ``mlx_q8`` in
   ``QUANTIZATION_METHODS`` lets :meth:`ModelConfig._verify_quantization`
   recognize them as known methods without backend-specific exceptions in
   the generic config code. ``mlx_q4`` / ``mlx_q8`` are on-the-fly load
   presets. ``mlx`` is a passthrough marker for already-quantized MLX
   checkpoints whose bit-width is not one of those presets.

2. Auto-detection for mlx-community HF repos.
   :meth:`override_quantization_method` claims ``config.json`` blocks of
   the form ``{"group_size": <int>, "bits": <int>}`` with no
   ``quant_method`` key. ``bits=4`` / ``bits=8`` map to the on-the-fly
   preset names; any other positive integer bit-width maps to ``mlx``.
   ``mlx_lm.load`` already instantiated the quantized modules, so the
   runner does not requantize.

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
    """Marker config for MLX backend quantization names.

    Not a real quantization config — the MLX backend handles quantization
    itself. Any standard-PyTorch-path method that touches this class raises
    a helpful error pointing the user at ``SGLANG_USE_MLX=1``.
    """

    _ERR = (
        "MLX quantization (auto-detected mlx, or --quantization mlx_q4 / "
        "mlx_q8) is handled by the MLX backend at model-load time, not by "
        "this QuantizationConfig class. If you reached this error, "
        "SGLANG_USE_MLX=1 is likely not set."
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
    def from_config(cls, config: Dict[str, Any]) -> MlxQuantizationConfig:
        raise NotImplementedError(cls._ERR)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        """Auto-detect mlx-community-shape quantization configs.

        mlx-community models ship ``config.json`` with::

            "quantization_config": {"group_size": <int>, "bits": <int>}

        No ``quant_method`` key. Map 4-bit / 8-bit to ``mlx_q4`` / ``mlx_q8``
        and every other positive bit-width to the ``mlx`` passthrough
        marker so pre-quantized HF repos load without ``--quantization``.

        Returns ``None`` for any input that does not look like a bare MLX
        config: non-dict, dict with an explicit ``quant_method``, missing
        keys, non-integer values, or (on the passthrough branch) bool /
        non-positive ``bits``. ``group_size`` is only a shape check.
        Extra keys (e.g. ``mode: affine``) are ignored. Also defers to
        any explicit ``--quantization`` CLI choice (``user_quant``) per
        the registry contract: CLI selection takes priority over auto-detect.
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
        if isinstance(bits, bool) or bits <= 0:
            return None
        # Pre-quantized Hub dumps that are not mlx_q4/mlx_q8. The MLX
        # runner loads via mlx_lm and does not treat this name as an
        # on-the-fly preset.
        return "mlx"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        raise NotImplementedError(self._ERR)
