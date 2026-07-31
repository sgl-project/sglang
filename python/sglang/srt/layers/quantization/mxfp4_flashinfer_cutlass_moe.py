"""DeepSeek-V4 MXFP4 expert backend backed by FlashInfer CUTLASS MoE.

``Fp8Config`` selects this backend for SM90 and SM120; SM100 uses the
TRT-LLM implementation.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.utils import is_flashinfer_available, log_info_on_rank0
from sglang.srt.utils.common import is_sm120_supported

# Suppress TRT-LLM CUTLASS trace logs without overriding user configuration.
os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

# MXFP4 group/block size (E8M0 scale per 32 fp4 weights).
_GROUP_SIZE = 32


class Mxfp4FlashinferCutlassMoEMethod:
    """FlashInfer MXFP4 MoE: W4A16 on SM90 and W4A8 on SM120."""

    def __init__(self, fp8_method, prefix: str):
        if not is_flashinfer_available():
            raise RuntimeError("Mxfp4FlashinferCutlassMoEMethod requires FlashInfer.")
        self._use_mxfp8_act_scaling = is_sm120_supported()
        self._fp8 = fp8_method
        self.prefix = prefix
        self._swiglu_limit_tensor: torch.Tensor | None = None
        self._mxfp4_weight_global_scale_tensor: torch.Tensor | None = None

    @property
    def load_up_proj_weight_first(self) -> bool:
        """Load W13 directly as ``[up; gate]`` for FlashInfer CUTLASS."""
        return True

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        # Both CUTLASS paths require dimensions aligned to 128.
        if hidden_size % 128 != 0 or intermediate_size_per_partition % 128 != 0:
            raise ValueError(
                "Mxfp4FlashinferCutlassMoEMethod requires hidden_size and "
                "intermediate_size_per_partition to be multiples of 128 "
                f"(got hidden={hidden_size}, "
                f"intermediate={intermediate_size_per_partition})."
            )
        # Keep checkpoint scales in native E8M0 instead of staging them as FP32.
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            fp4_scale_dtype=torch.float8_e8m0fnu,
            **extra_weight_attrs,
        )

    def create_moe_runner(self, layer: Module, moe_runner_config) -> None:
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        self.moe_runner_config = moe_runner_config

        E = layer.num_local_experts
        device = layer.w13_weight.device
        if self._use_mxfp8_act_scaling:
            # FlashInfer's MXFP4 ABI requires a neutral per-expert global scale.
            self._mxfp4_weight_global_scale_tensor = torch.ones(
                E, dtype=torch.float32, device=device
            )

        # FlashInfer defaults alpha/beta to 1/0, so DSv4 only supplies its clamp.
        swiglu_limit = getattr(moe_runner_config, "swiglu_limit", None)
        if swiglu_limit is not None:
            self._swiglu_limit_tensor = torch.full(
                (E,), float(swiglu_limit), dtype=torch.float32, device=device
            )
        else:
            self._swiglu_limit_tensor = None

        # Register the fused func at runner construction so the FusedOpPool
        # lookup at `MoeRunner.__init__` finds it.
        import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401

        self.runner = MoeRunner(MoeRunnerBackend.FLASHINFER_MXFP4, moe_runner_config)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Preserve the base FP4 post-load handling.
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        arch = "SM120" if self._use_mxfp8_act_scaling else "SM90"
        log_info_on_rank0(
            logger,
            f"Preparing DSv4 MXFP4 experts for FlashInfer {arch} CUTLASS "
            f"(layer: {self.prefix})...",
        )

        # FlashInfer consumes the raw bytes of the checkpoint's E8M0 scales.
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            scale = getattr(layer, name)
            if scale.dtype != torch.float8_e8m0fnu:
                raise TypeError(
                    f"{name} must remain native E8M0 for FlashInfer MXFP4, "
                    f"got {scale.dtype}."
                )
        w13_scale_u8 = layer.w13_weight_scale_inv.data.view(torch.uint8)
        w2_scale_u8 = layer.w2_weight_scale_inv.data.view(torch.uint8)

        if self._use_mxfp8_act_scaling:
            from flashinfer import block_scale_interleave

            if (
                not layer.w13_weight.is_contiguous()
                or not layer.w2_weight.is_contiguous()
            ):
                raise ValueError("SM120 FlashInfer MXFP4 weights must be contiguous.")
            for scale_u8 in (w13_scale_u8, w2_scale_u8):
                scale_u8.copy_(block_scale_interleave(scale_u8).reshape_as(scale_u8))
        else:
            from flashinfer.fused_moe import (
                interleave_moe_scales_for_sm90_mixed_gemm,
                interleave_moe_weights_for_sm90_mixed_gemm,
            )

            w13_il = interleave_moe_weights_for_sm90_mixed_gemm(
                layer.w13_weight.data.view(torch.uint8).contiguous(), "fp4"
            )
            w2_il = interleave_moe_weights_for_sm90_mixed_gemm(
                layer.w2_weight.data.view(torch.uint8).contiguous(), "fp4"
            )
            w13_s_il = interleave_moe_scales_for_sm90_mixed_gemm(
                w13_scale_u8, group_size=_GROUP_SIZE
            )
            w2_s_il = interleave_moe_scales_for_sm90_mixed_gemm(
                w2_scale_u8, group_size=_GROUP_SIZE
            )
            layer.w13_weight = Parameter(w13_il, requires_grad=False)
            layer.w2_weight = Parameter(w2_il, requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(w13_s_il, requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(w2_s_il, requires_grad=False)

        layer._dsv4_mxfp4_backend = (
            "flashinfer_cutlass_sm120"
            if self._use_mxfp8_act_scaling
            else "flashinfer_cutlass_sm90"
        )
        # SM90 creates full-size interleaved copies; release old layouts per layer.
        if not self._use_mxfp8_act_scaling:
            torch.cuda.empty_cache()

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMxfp4MoeQuantInfo,
        )

        quant_info = FlashInferCutlassMxfp4MoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_weight_scale=layer.w13_weight_scale_inv,
            w2_weight_scale=layer.w2_weight_scale_inv,
            mxfp4_weight_global_scale=self._mxfp4_weight_global_scale_tensor,
            w13_bias=None,
            w2_bias=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=self._swiglu_limit_tensor,
            moe_tp_size=layer.moe_tp_size,
            moe_tp_rank=layer.moe_tp_rank,
            moe_ep_size=layer.moe_ep_size,
            moe_ep_rank=layer.moe_ep_rank,
            padded_hidden=None,
        )
        return self.runner.run(dispatch_output, quant_info)
