"""MXFP4 W4A4 online quantization config (dual-level MXFP4 weights + activations).

Triggered by ``--quantization mxfp4`` on the Ascend NPU backend. On CUDA / AMD /
CPU the ``mxfp4`` key resolves to the upstream :class:`Mxfp4Config` (OCP MXFP4
MoE) instead; the per-device split is done at registration time in
``sglang.srt.layers.quantization.__init__`` (this config is only registered
inside the ``is_npu()`` block, mirroring ``GPTQAscendConfig``).

Online mode: FP16/BF16 weights are quantised to **dual-level** MXFP4 in
``process_weights_after_loading`` (a finer FP8 E4M3 L0 block scale plus a coarser
L1 scale); activations are dynamically quantised the same way and the matmul runs
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

# MXFP4 block (group) size. The reduction dim K must be a multiple of this for the
# block scales to tile evenly; non-aligned layers fall back to BF16.
MXFP4_W4A4_GROUP_SIZE = 32


class Mxfp4W4A4Config(QuantizationConfig):
    """MXFP4 W4A4 online quantization config for Ascend NPU.

    True W4(weight) A4(activation): both weights and activations are MXFP4
    (``float4_e2m1fn_x2``). ``get_quant_method`` dispatches per layer type, and
    only Ascend NPU is wired up today (on other devices ``mxfp4`` maps to the
    upstream ``Mxfp4Config``):

    * ``LinearBase`` → dual-level MXFP4 (``NPUDualLevelMXFP4LinearMethod``);
      single-level RTN degenerated under greedy decoding, so the finer FP8 L0
      scale is used for the non-expert layers.
    * ``FusedMoE`` → single-level MXFP4 experts
      (``NPUMXFP4W4A4FusedMoEMethod``). MoE has no grouped dual-level kernel, so
      the experts stay single-level; the online-RTN risk is contained by the
      mixed-precision layout (only the experts drop to W4A4 — the surrounding
      Linear layers keep the more accurate dual-level MXFP4).
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
            # MXFP4 block scales use group_size=32, so a reduction dim K that is
            # not a multiple of 32 does not tile evenly — the same constraint that
            # forces the W4A8 path to skip such layers. Qwen3.5 vision MLP
            # linear_fc2 (K=4304, 4304/32=134.5) hits this; fall back to BF16,
            # mirroring how the offline msmodelslim yaml leaves linear_fc2
            # unquantized. LLM text layers and the tp=1 vision QKV are 32-aligned.
            if layer.input_size % MXFP4_W4A4_GROUP_SIZE != 0:
                logger.warning(
                    "mxfp4 W4A4: skipping %s (input_size=%d not a multiple of "
                    "%d); falling back to unquantized BF16.",
                    prefix,
                    layer.input_size,
                    MXFP4_W4A4_GROUP_SIZE,
                )
                return UnquantizedLinearMethod()
            if is_npu():
                from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
                    NPUDualLevelMXFP4LinearMethod,
                )

                # Online W4A4 always uses dual-level MXFP4 (finer FP8 L0 scales):
                # single-level RTN was too lossy and degenerated under greedy
                # decoding. Requires Ascend 950 (A5). The single-level kernel is
                # retained only for the offline msmodelslim path.
                return NPUDualLevelMXFP4LinearMethod(self)
            raise NotImplementedError(
                "mxfp4 W4A4 (single-level MXFP4 weights + activations) is currently "
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
                    NPUMXFP4W4A4FusedMoEMethod,
                )

                # Experts run single-level MXFP4 (packed fp4 weights + fp4
                # activations). Single-level, not dual-level: there is no
                # grouped dual-level matmul kernel, so the online-RTN accuracy
                # risk is mitigated by keeping this to the experts only (the
                # mixed-precision layout — non-expert layers stay MXFP8/BF16).
                # Requires Ascend 950 (A5).
                return NPUMXFP4W4A4FusedMoEMethod(self)
            # MoE single-level MXFP4 W4A4 has no CUDA/other-device kernel; fall
            # back to unquantised rather than fail load on non-NPU backends.
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
