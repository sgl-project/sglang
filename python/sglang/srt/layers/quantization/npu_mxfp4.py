"""Online W4A8 for Ascend, selected by ``--quantization mxfp_w4a8``."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.npu_mxfp4_utils import (
    MXFP4_BLOCK_SIZE,
    local_input_size,
)
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)


class Mxfp4W4A8Config(QuantizationConfig):
    """Online MXFP4 weights and dynamic MXFP8 activations for Linear and MoE."""

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
            # FP4 matmul cannot handle a partial K block (e.g. visual K=4304).
            local_k = local_input_size(layer)
            if local_k % MXFP4_BLOCK_SIZE != 0:
                logger.warning(
                    "mxfp_w4a8: skipping %s (local_input_size=%d not a multiple of %d); "
                    "falling back to unquantized precision.",
                    prefix,
                    local_k,
                    MXFP4_BLOCK_SIZE,
                )
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
            if is_layer_skipped(
                prefix,
                self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(
                    layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
                )
            if is_npu():
                from sglang.srt.hardware_backend.npu.quantization.online_moe_methods import (
                    NPUW4A8MXFP4OnlineMoEMethod,
                )

                return NPUW4A8MXFP4OnlineMoEMethod(self)
            raise NotImplementedError(
                "mxfp_w4a8 (MXFP4 weights + MXFP8 activations, W4A8) FusedMoE is "
                "currently only implemented for the Ascend NPU backend; no "
                "CUDA/other-device kernel exists yet."
            )
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []
