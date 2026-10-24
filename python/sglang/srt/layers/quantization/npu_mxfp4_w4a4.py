"""MXFP4 W4A4 online quantization config (dual-level MXFP4 weights + activations).

Triggered by ``--quantization mxfp4`` on the Ascend NPU backend. On CUDA / AMD /
CPU the ``mxfp4`` key resolves to the upstream :class:`Mxfp4Config` (OCP MXFP4
MoE) instead; the per-device split is done at registration time in
``sglang.srt.layers.quantization.__init__`` (this config is only registered
inside the ``is_npu()`` block, mirroring ``GPTQAscendConfig``).

Online mode: FP16/BF16 weights are quantised to **dual-level** MXFP4 in
``process_weights_after_loading`` (L0 FLOAT32 scales for 512-element blocks and
L1 UE8M0 scales for 32-element blocks); activations are dynamically quantised the same way and the matmul runs
via ``npu_dual_level_quant_matmul`` (see :class:`NPUDualLevelMXFP4LinearMethod`).
Dual-level is the sole online path — it captures per-block dynamic range far more
accurately than a single-level UE8M0 scale, avoiding the RTN degradation that made
single-level online decoding loop under greedy sampling. Requires Ascend 950 (A5).

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
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)


class Mxfp4W4A4Config(QuantizationConfig):
    """Dual-level MXFP4 W4A4 online quantization config for Ascend NPU.

    True W4(weight) A4(activation): both weights and activations are quantised
    to dual-level MXFP4 (``float4_e2m1fn_x2``). The device-specific linear
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
                    NPUDualLevelMXFP4LinearMethod,
                )

                # Online W4A4 always uses dual-level MXFP4 (L0/L1 scales):
                # single-level RTN was too lossy and degenerated under greedy
                # decoding. Requires Ascend 950 (A5). The single-level kernel is
                # retained only for the offline msmodelslim path.
                return NPUDualLevelMXFP4LinearMethod(self)
            raise NotImplementedError(
                "mxfp4 W4A4 (dual-level MXFP4 weights + activations) is currently "
                "only implemented for the Ascend NPU backend; no CUDA/other-device "
                "kernel exists in this config. Add a device branch here when one lands."
            )
        elif isinstance(layer, FusedMoE):
            from sglang.srt.hardware_backend.npu.quantization.mxfp_capabilities import (
                W4A4,
                get_mxfp_capability,
                validate_mxfp_configuration,
            )

            cap = get_mxfp_capability("srt", "moe", W4A4, "online")
            reason = validate_mxfp_configuration(cap, prefix=prefix)
            logger.warning("Ascend MXFP (prefix=%s): %s", prefix, reason)
            return UnquantizedFusedMoEMethod(
                layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
            )
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []
