"""MXFP4 W4A4 online config: dual-level Linear and single-level MoE.

Triggered by ``--quantization mxfp4`` on the Ascend NPU backend. On CUDA / AMD /
CPU the ``mxfp4`` key resolves to the upstream :class:`Mxfp4Config` (OCP MXFP4
MoE) instead; the per-device split is done at registration time in
``sglang.srt.layers.quantization.__init__`` (this config is only registered
inside the ``is_npu()`` block, mirroring ``GPTQAscendConfig``).

Online Linear layers quantize FP16/BF16 weights and activations to dual-level
MXFP4 and run ``npu_dual_level_quant_matmul``. MoE experts use single-level
MXFP4 because the grouped kernels do not support dual-level scales. Ignored or
group-unaligned Linear layers remain unquantized. This per-layer scale policy
does not establish model-level RTN accuracy. Requires Ascend 950.

Offline (msmodelslim ``W4A4_MXFP4``) checkpoints are single-level (the checkpoint
stores UE8M0 scales) and are handled separately by the ``modelslim`` config
(``ModelSlimMXFP4Scheme`` → ``NPUSingleLevelMXFP4OfflineLinearMethod``), not this
class.
"""

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


class Mxfp4W4A4Config(QuantizationConfig):
    """Online MXFP4 on Ascend: dual-level Linear and single-level MoE experts.

    Ignored layers retain the unquantized method. On other devices, ``mxfp4``
    resolves to the upstream ``Mxfp4Config`` instead of this NPU-only config.
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

    def is_static_cfg(self) -> bool:
        return False

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
            # Preserve the existing fallback for non-aligned visual Linear
            # layers, such as Qwen3.5 vision FC2 with K=4304.
            local_k = local_input_size(layer)
            if local_k % MXFP4_BLOCK_SIZE != 0:
                logger.warning(
                    "mxfp4 W4A4: skipping %s (local_input_size=%d not a multiple of "
                    "%d); falling back to unquantized precision.",
                    prefix,
                    local_k,
                    MXFP4_BLOCK_SIZE,
                )
                return UnquantizedLinearMethod()
            if is_npu():
                from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
                    NPUDualLevelMXFP4LinearMethod,
                )

                # Online Linear uses dual-level scales; the single-level
                # Linear kernel remains reserved for offline ModelSlim.
                return NPUDualLevelMXFP4LinearMethod(self)
            raise NotImplementedError(
                "mxfp4 W4A4 Linear (dual-level MXFP4 weights + activations) is currently "
                "only implemented for the Ascend NPU backend; no CUDA/other-device "
                "kernel exists in this config. Add a device branch here when one lands."
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
                    NPUW4A4MXFP4OnlineMoEMethod,
                )

                return NPUW4A4MXFP4OnlineMoEMethod(self)
            logger.warning(
                "MXFP4 W4A4 MoE is only implemented for the Ascend NPU backend "
                "(prefix=%s); falling back to unquantized MoE (full precision).",
                prefix,
            )
            return UnquantizedFusedMoEMethod(
                layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
            )
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []
