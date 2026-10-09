"""MXFP4 W4A8 online quantization config (MXFP4 weights + MXFP8 activations).

Triggered by ``--quantization mxfp_w4a8``.

Online mode: FP16/BF16 weights are quantised to MXFP4 in
``process_weights_after_loading``; activations are dynamically quantised to
MXFP8 (``float8_e4m3fn`` + UE8M0 block scale) at inference time and the matmul
runs via ``npu_quant_matmul`` with FP4 weights.

The config is device-agnostic and dispatches per device in
``get_quant_method``; only the Ascend NPU backend (Ascend 950 / A5) is
implemented today.
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
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)


class Mxfp4W4A8Config(QuantizationConfig):
    """MXFP4 W4A8 online quantization config; dispatches per device.

    True W4(weight) A8(activation): weights are quantised online to MXFP4 and
    activations to MXFP8 at inference time. The device-specific linear method
    is selected in ``get_quant_method``; only Ascend NPU is wired up today.
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
        return "mxfp_w4a8"

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
    def from_config(cls, config: Dict) -> Mxfp4W4A8Config:
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
            if is_npu():
                from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
                    NPUMXFP4W4A8LinearMethod,
                )

                return NPUMXFP4W4A8LinearMethod(self)
            raise NotImplementedError(
                "mxfp_w4a8 (MXFP4 weights + MXFP8 activations, W4A8) is currently "
                "only implemented for the Ascend NPU backend; no CUDA/other-device "
                "kernel exists yet. Add a device branch here when one lands."
            )
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
