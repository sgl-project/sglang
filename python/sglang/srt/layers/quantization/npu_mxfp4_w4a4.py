"""MXFP4 W4A4 online quantization config (single-level MXFP4 weights + activations).

Triggered by ``--quantization mxfp4`` on the Ascend NPU backend. On CUDA / AMD /
CPU the ``mxfp4`` key resolves to the upstream :class:`Mxfp4Config` (OCP MXFP4
MoE) instead; the per-device split is done at registration time in
``sglang.srt.layers.quantization.__init__`` (this config is only registered
inside the ``is_npu()`` block, mirroring ``GPTQAscendConfig``).

Online mode: FP16/BF16 weights are quantised to single-level MXFP4 in
``process_weights_after_loading``; activations are dynamically quantised to
MXFP4 at inference time and the matmul runs via ``npu_quant_matmul`` with
``group_sizes=[1, 1, MXFP4_BLOCK_SIZE]`` (``x1_dtype = x2_dtype = float4_e2m1fn_x2``).

Offline (msmodelslim ``W4A4_MXFP4``) checkpoints are handled separately by the
``modelslim`` config (``ModelSlimMXFP4Scheme``), not this class.
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


class Mxfp4W4A4Config(QuantizationConfig):
    """Single-level MXFP4 W4A4 online quantization config for Ascend NPU.

    True W4(weight) A4(activation): both weights and activations are quantised
    to single-level MXFP4 (``float4_e2m1fn_x2``). The device-specific linear
    method is selected in ``get_quant_method``; only Ascend NPU is wired up
    today (on other devices ``mxfp4`` maps to the upstream ``Mxfp4Config``).
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
        return "mxfp4"

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
    def from_config(cls, config: Dict) -> Mxfp4W4A4Config:
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
                    NPUSingleLevelMXFP4LinearMethod,
                )

                return NPUSingleLevelMXFP4LinearMethod(self)
            raise NotImplementedError(
                "mxfp4 W4A4 (single-level MXFP4 weights + activations) is currently "
                "only implemented for the Ascend NPU backend; no CUDA/other-device "
                "kernel exists in this config. Add a device branch here when one lands."
            )
        elif isinstance(layer, FusedMoE):
            # MoE single-level MXFP4 W4A4 not yet implemented; fall back to unquantised
            logger.warning(
                "MXFP4 W4A4 quantization is not yet supported for FusedMoE layers "
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
