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

from sglang.srt.runtime_context import get_exec, get_platform
from sglang.srt.utils import is_flashinfer_available, log_info_on_rank0

# Suppress TRT-LLM CUTLASS trace logs without overriding user configuration.
os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.utils import RoutingMethodType

# MXFP4 group/block size (E8M0 scale per 32 fp4 weights).
_GROUP_SIZE = 32


def _prepare_mxfp4_weights_inplace(
    weight: torch.Tensor, scale: torch.Tensor, *, use_humming: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from flashinfer.fused_moe import (
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
    )

    if use_humming:
        from flashinfer.fused_moe import (
            preprocess_moe_weights_for_sm90_mixed_gemm_humming,
        )

    # Loader inputs are contiguous; otherwise these make full-size copies.
    weight = weight.view(torch.uint8).contiguous()
    scale = scale.contiguous()
    residuals = []
    # FlashInfer processes weights and clamps Humming ranges per expert; scale
    # folding preserves dim 0. Fresh outputs can safely replace each input chunk.
    for weight_block, scale_block in zip(weight.split(8), scale.split(8)):
        if use_humming:
            weight_out, scale_out, residual = (
                preprocess_moe_weights_for_sm90_mixed_gemm_humming(
                    weight_block, scale_block
                )
            )
            residuals.append(residual)
        else:
            weight_out = interleave_moe_weights_for_sm90_mixed_gemm(weight_block, "fp4")
            scale_out = interleave_moe_scales_for_sm90_mixed_gemm(
                scale_block, group_size=_GROUP_SIZE
            )
        weight_block.copy_(weight_out)
        scale_block.copy_(scale_out.reshape_as(scale_block))
        folded_shape = scale_out.shape[1:]
        del weight_out, scale_out
    return (
        weight,
        scale.view(scale.shape[0], *folded_shape),
        torch.cat(residuals) if use_humming else None,
    )


class Mxfp4FlashinferCutlassMoEMethod:
    """FlashInfer MXFP4 MoE: W4A16/W4A8 on SM90 and W4A8 on SM120."""

    fuse_routed_scaling_factor_in_topk = True

    def __init__(self, fp8_method, prefix: str):
        if not is_flashinfer_available():
            raise RuntimeError("Mxfp4FlashinferCutlassMoEMethod requires FlashInfer.")
        self._use_mxfp8_act_scaling = get_platform().is_sm120
        precision = get_exec().moe.flashinfer_mxfp4_moe_precision
        # precision=fp8 is an SM90 knob (Humming W4A8); on SM120 the MXFP8
        # activation path already computes in FP8, so the flag is simply inert
        # there rather than an error -- one config can move across hardware.
        self._use_sm90_humming = not self._use_mxfp8_act_scaling and precision == "fp8"
        self._fp8 = fp8_method
        self.prefix = prefix
        self._swiglu_limit_tensor: torch.Tensor | None = None
        self._use_swiglu_step = False
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
        # The fused func reads the layer's MoE placement off the config.
        if moe_runner_config.layer is None:
            moe_runner_config.layer = layer

        E = layer.num_local_experts
        device = layer.w13_weight.device
        if self._use_mxfp8_act_scaling:
            # FlashInfer's MXFP4 ABI requires a neutral per-expert global scale.
            self._mxfp4_weight_global_scale_tensor = torch.ones(
                E, dtype=torch.float32, device=device
            )

        # FlashInfer defaults alpha/beta to 1/0, so DSv4 only supplies its clamp.
        # Bailing clamps after SiLU (gemm1_clamp_limit), which the kernel only
        # implements in its SwigluStep variant.
        swiglu_limit = getattr(moe_runner_config, "swiglu_limit", None)
        gemm1_clamp_limit = getattr(moe_runner_config, "gemm1_clamp_limit", None)
        self._use_swiglu_step = (
            gemm1_clamp_limit is not None
            and getattr(moe_runner_config, "gemm1_alpha", None) is None
        )
        clamp_limit = (
            gemm1_clamp_limit if gemm1_clamp_limit is not None else swiglu_limit
        )
        if clamp_limit is not None:
            self._swiglu_limit_tensor = torch.full(
                (E,), float(clamp_limit), dtype=torch.float32, device=device
            )
        else:
            self._swiglu_limit_tensor = None

        # Register the fused func at runner construction so the FusedOpPool
        # lookup at `MoeRunner.__init__` finds it.
        import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401

        self.runner = MoeRunner(MoeRunnerBackend.FLASHINFER_MXFP4, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return self.runner.runner_backend

    def get_moe_routing_method_type(
        self, layer: Module, requested_method: RoutingMethodType | None
    ) -> RoutingMethodType | None:
        return requested_method

    def process_weights_after_loading(self, layer: Module) -> None:
        # Preserve the base FP4 post-load handling.
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        arch = "SM120" if self._use_mxfp8_act_scaling else "SM90"
        precision = (
            "W4A8" if self._use_sm90_humming or self._use_mxfp8_act_scaling else "W4A16"
        )
        log_info_on_rank0(
            logger,
            f"Preparing DSv4 MXFP4 experts for FlashInfer {arch} CUTLASS {precision} "
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
            w13_il, w13_s_il, w13_residual = _prepare_mxfp4_weights_inplace(
                layer.w13_weight.data,
                w13_scale_u8,
                use_humming=self._use_sm90_humming,
            )
            w2_il, w2_s_il, w2_residual = _prepare_mxfp4_weights_inplace(
                layer.w2_weight.data,
                w2_scale_u8,
                use_humming=self._use_sm90_humming,
            )
            if self._use_sm90_humming:
                layer.w13_humming_residual_scale = Parameter(
                    (w13_residual * 64.0).contiguous(), requires_grad=False
                )
                layer.w2_humming_residual_scale = Parameter(
                    (w2_residual * 64.0).contiguous(), requires_grad=False
                )
                layer.humming_fc2_act_scale = Parameter(
                    torch.ones((), dtype=torch.float32, device=w13_scale_u8.device),
                    requires_grad=False,
                )
            layer.w13_weight = Parameter(w13_il, requires_grad=False)
            layer.w2_weight = Parameter(w2_il, requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(w13_s_il, requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(w2_s_il, requires_grad=False)

        layer._dsv4_mxfp4_backend = (
            "flashinfer_cutlass_sm120"
            if self._use_mxfp8_act_scaling
            else (
                "flashinfer_cutlass_sm90_fp8"
                if self._use_sm90_humming
                else "flashinfer_cutlass_sm90"
            )
        )
        # Release cached SM90 preprocessing scratch per layer.
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
            w13_humming_residual_scale=getattr(
                layer, "w13_humming_residual_scale", None
            ),
            w2_humming_residual_scale=getattr(layer, "w2_humming_residual_scale", None),
            humming_fc2_act_scale=getattr(layer, "humming_fc2_act_scale", None),
            w13_bias=None,
            w2_bias=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=self._swiglu_limit_tensor,
            use_swiglu_step=self._use_swiglu_step,
            padded_hidden=None,
        )
        return self.runner.run(dispatch_output, quant_info)
