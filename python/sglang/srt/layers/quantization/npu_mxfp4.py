"""Ascend NPU MXFP4 W4A8 online quantization config.

Triggered by ``--quantization mxfp4_w4a8_npu``.

Online mode: loads FP16/BF16 weights, quantises to MXFP4 (dual-level) in
``process_weights_after_loading``.  During inference, activations are
dynamically quantised to MXFP4 and ``npu_dual_level_quant_matmul`` is used
for the matrix multiply.

Hardware requirement: Ascend 950 (DualLevelQuantBatchMatmul is NOT supported
on Atlas A2/A3 – check your hardware before enabling).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped

logger = logging.getLogger(__name__)


class NPUMxfp4Config(QuantizationConfig):
    """Quantization config for Ascend NPU MXFP4 W4A8 online quantization.

    Weights are quantised online to MXFP4 dual-level format during model
    loading.  Activations are quantised dynamically to MXFP4 at inference
    time.  The matmul is executed via ``npu_dual_level_quant_matmul``.
    """

    def __init__(
        self,
        ignored_layers: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4_w4a8_npu"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # NPU bypasses CUDA capability checks

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict) -> NPUMxfp4Config:
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        if ignored_layers:
            normalized: List[str] = []
            for layer in ignored_layers:
                base = layer.removeprefix("model.")
                normalized.append(base)
                normalized.append(f"model.{base}")
            ignored_layers = normalized
        packed_modules_mapping = (
            cls.get_from_keys_or(config, ["packed_modules_mapping"], {}) or {}
        )
        return cls(
            ignored_layers=ignored_layers,
            packed_modules_mapping=packed_modules_mapping,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
                NPUMXFP4W4A8LinearMethod,
            )

            return NPUMXFP4W4A8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # MoE MXFP4 not yet implemented; fall back to unquantised
            logger.warning(
                "MXFP4 W4A8 quantization is not yet supported for FusedMoE layers "
                "(prefix=%s). Falling back to unquantized MoE — MoE weights will "
                "run in full precision (BF16/FP16).",
                prefix,
            )
            return UnquantizedFusedMoEMethod(
                layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
            )
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []
