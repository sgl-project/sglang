"""MXFP4 x MXFP4 fused MoE method for SM120.

DeepSeek-V4-Flash ships routed experts as MXFP4 (E2M1 packed int8 weights +
E8M0 32-element block scales). This method loads them as-is and runs the experts
with FlashInfer's fused SwiGLU CuTe-DSL MoE kernels
(``launch_sm120_moe(quant_mode="mxfp4")``): MXFP4 weights x MXFP4 activations,
E8M0 self-scaling.

Weight loading is shared with the W4A8 method (``mxfp4_w4a8_moe.py``); only the
weight-scale swizzle target differs: W4A4 swizzles into the 128x4 layout then
converts to the MMA layout the fused kernel's ``_get_weight_views`` expects.

Selected by ``fp8.py`` on SM120 + MXFP4 experts + FlashInfer exposing
``launch_sm120_moe``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import log_info_on_rank0, set_weight_attrs
from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)

_FP4_BLOCK_K = 32  # E8M0 block size


class Mxfp4W4A4MoEMethod:
    """MXFP4 weights x MXFP4 activations MoE (SM120 fused CuTe-DSL)."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_moe_runner(self, layer, moe_runner_config):
        # The fused launch runs in apply(); the runner only carries MoE config
        # (routed_scaling_factor / swiglu_limit).
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # float32 placeholder scales; E8M0 are exact powers of two, so the HF
        # loader casts losslessly. process_weights_after_loading re-encodes them.
        w13_weight_scale = Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // _FP4_BLOCK_K,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // _FP4_BLOCK_K,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    @staticmethod
    def _to_e8m0_u8(scale_f32: torch.Tensor) -> torch.Tensor:
        """float32 scale magnitude -> E8M0 byte (uint8)."""
        return scale_f32.to(torch.float8_e8m0fnu).view(torch.uint8)

    def process_weights_after_loading(self, layer: Module) -> None:
        if not is_sm120_supported():
            raise RuntimeError(
                "Mxfp4W4A4MoEMethod requires SM120 (RTX PRO 6000 Blackwell)."
            )
        if getattr(layer, "_w4a4_weights_built", False):
            return

        # 128x4 weight-scale swizzle (shared with W4A8), then convert to the MMA
        # layout the fused kernel's _get_weight_views expects.
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

        from sglang.srt.layers.quantization.mxfp4_sm120_common import (
            swizzle_weight_scale_mxf4,
        )

        w13 = layer.w13_weight.data  # (E, 2I, H/2) int8, checkpoint [w1|w3]
        w2 = layer.w2_weight.data  # (E, H, I/2)  int8
        w13_s = layer.w13_weight_scale_inv.data
        w2_s = layer.w2_weight_scale_inv.data

        E = w13.shape[0]
        twoI = w13.shape[1]
        Hhalf = w13.shape[2]
        H = Hhalf * 2
        Ihalf = w2.shape[2]
        intermediate = Ihalf * 2

        # Fused kernel expects gate/up as [w3, w1] (up, gate), but the checkpoint
        # loads W13 as [w1, w3]; swap the I-row halves of weights and 3D scales.
        Irows = twoI // 2
        w13 = torch.cat([w13[:, Irows:, :], w13[:, :Irows, :]], dim=1).contiguous()
        w13_s = torch.cat(
            [w13_s[:, Irows:, :], w13_s[:, :Irows, :]], dim=1
        ).contiguous()

        w13_s_u8 = self._to_e8m0_u8(w13_s.to(torch.float32))
        w2_s_u8 = self._to_e8m0_u8(w2_s.to(torch.float32))

        # 128x4 block-scale swizzle (per expert), then convert to the MMA layout
        # the fused kernel reads (experts flattened into the leading row dim).
        w13_s_sw = swizzle_weight_scale_mxf4(w13_s_u8, E, twoI, H)
        w2_s_sw = swizzle_weight_scale_mxf4(w2_s_u8, E, H, intermediate)
        w13_sf_mma = convert_sf_to_mma_layout(
            w13_s_sw.reshape(E * w13_s_sw.shape[1], w13_s_sw.shape[2]),
            m=twoI,
            k=H,
            num_groups=E,
            sf_vec_size=_FP4_BLOCK_K,
        )
        w2_sf_mma = convert_sf_to_mma_layout(
            w2_s_sw.reshape(E * w2_s_sw.shape[1], w2_s_sw.shape[2]),
            m=H,
            k=intermediate,
            num_groups=E,
            sf_vec_size=_FP4_BLOCK_K,
        )

        layer.w13_weight = Parameter(
            w13.view(torch.uint8).contiguous(), requires_grad=False
        )
        layer.w2_weight = Parameter(
            w2.view(torch.uint8).contiguous(), requires_grad=False
        )
        layer.w13_weight_scale_inv = Parameter(
            w13_sf_mma.contiguous(), requires_grad=False
        )
        layer.w2_weight_scale_inv = Parameter(
            w2_sf_mma.contiguous(), requires_grad=False
        )
        layer._w4a4_H = H
        layer._w4a4_I = intermediate
        layer._w4a4_E = E
        layer._w4a4_weights_built = True

        # Build the FlashInfer weight views now (load time), not lazily under
        # CUDA-graph capture, so the ~48 MB/layer scale storage stays out of the
        # graph-private pool.
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _get_weight_views,
        )

        ones_e = torch.ones(E, device=layer.w13_weight.device, dtype=torch.float32)
        layer._w4a4_alpha = ones_e  # GEMM alpha = 1 (MXFP4 self-scales)
        layer._w4a4_weight_views = _get_weight_views(
            w1_fp4=layer.w13_weight,
            w1_blockscale=layer.w13_weight_scale_inv,
            w2_fp4=layer.w2_weight,
            w2_blockscale=layer.w2_weight_scale_inv,
            w1_alphas=ones_e,
            w2_alphas=ones_e,
            n=intermediate,
            k=H,
            activation_precision="fp4",
            quant_mode="mxfp4",
        )

        # Free the per-layer transients; the final Parameters hold contiguous
        # copies. Without this the allocator's reserve starves CUDA-graph capture.
        del w13, w2, w13_s, w2_s
        del w13_s_u8, w2_s_u8, w13_s_sw, w2_s_sw, w13_sf_mma, w2_sf_mma
        torch.cuda.empty_cache()

        log_info_on_rank0(
            logger,
            f"SM120 MXFP4 W4A4 experts ready "
            f"(E={E}, H={H}, I={intermediate}; layer: {self.prefix})",
        )

    def apply(
        self,
        layer: Module,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _get_cached_workspace,
            launch_sm120_moe,
            select_sm120_moe_backend,
        )

        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")

        hidden_states = dispatch_output.hidden_states
        M = hidden_states.shape[0]
        H = layer._w4a4_H
        intermediate = layer._w4a4_I
        E_local = layer._w4a4_E
        # Global expert count bounds topk-id range; local count sizes the expert
        # buffers. Under EP global > local and the kernel remaps global->local.
        E_global = int(getattr(layer, "num_experts", E_local))
        top_k = topk_output.topk_ids.shape[1]
        dev = hidden_states.device

        cfg = getattr(self.runner, "config", None)

        # MXFP4 self-scales: no activation global scale (GEMM alpha = 1).
        ones = layer._w4a4_alpha

        # One capped static workspace per (layer, top_k), allocated outside
        # CUDA-graph capture so decode never grows it. Larger batches pass
        # _workspace=None and let launch_sm120_moe size the right one.
        ws_cap = int(os.environ.get("SGLANG_MXFP4_STATIC_WS_CAP", "640"))
        routed_rows = M * top_k
        ws_key = getattr(layer, "_w4a4_ws_key", None)
        if ws_key != top_k:
            layer._w4a4_static_ws = _get_cached_workspace(
                backend="static",
                state_E=E_local,
                weight_E=E_global,
                routed_rows=ws_cap,
                k=H,
                n=intermediate,
                num_topk=top_k,
                device=dev,
                quant_mode="mxfp4",
                activation="silu",
            )
            layer._w4a4_ws_key = top_k

        backend = select_sm120_moe_backend(
            num_tokens=M, num_topk=top_k, quant_mode="mxfp4"
        )
        # EP forces static (kernel needs global->local remap); match the launcher.
        use_static = backend == "static" and routed_rows <= ws_cap
        workspace = layer._w4a4_static_ws if use_static else None

        output = torch.empty(M, H, dtype=hidden_states.dtype, device=dev)
        launch_sm120_moe(
            a=hidden_states,
            topk_ids=topk_output.topk_ids,
            topk_weights=topk_output.topk_weights,
            w1_weight=layer.w13_weight,
            w1_weight_sf=layer.w13_weight_scale_inv,
            w1_alpha=ones,
            fc2_input_scale=ones,
            w2_weight=layer.w2_weight,
            w2_weight_sf=layer.w2_weight_scale_inv,
            w2_alpha=ones,
            num_experts=E_global,
            top_k=top_k,
            num_local_experts=E_local,
            scatter_output=output,
            input_scales_are_reciprocal=False,
            fast_math=True,
            activation="silu",
            quant_mode="mxfp4",
            _weight_views=layer._w4a4_weight_views,
            _workspace=workspace,
        )

        routed_scaling_factor = (
            getattr(cfg, "routed_scaling_factor", None) if cfg else None
        )
        if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
            output = output * routed_scaling_factor
        return StandardCombineInput(hidden_states=output)
