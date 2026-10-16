from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from compressed_tensors.quantization import QuantizationArgs

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = ["CompressedTensorsW8A8MXFp8MoE"]


MXFP8_GROUP_SIZE = 32


class CompressedTensorsW8A8MXFp8MoE(CompressedTensorsMoEScheme):
    """OCP MXFP8 W8A8 MoE using grouped CUTLASS-style sgl-kernel dispatch."""

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
    ) -> None:
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        if weight_quant.group_size != MXFP8_GROUP_SIZE:
            raise ValueError(
                "MXFP8 MoE requires weight group_size=32; "
                f"got {weight_quant.group_size}."
            )
        if input_quant.group_size != MXFP8_GROUP_SIZE:
            raise ValueError(
                f"MXFP8 MoE requires input group_size=32; got {input_quant.group_size}."
            )
        if not input_quant.dynamic:
            raise ValueError("MXFP8 MoE currently requires dynamic activations.")

    @classmethod
    def get_min_capability(cls) -> int:
        # Backend capability is gated by the sgl-kernel runtime availability.
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        del params_dtype

        if hidden_size % MXFP8_GROUP_SIZE != 0:
            raise ValueError(
                f"MoE hidden_size={hidden_size} must be divisible by "
                f"{MXFP8_GROUP_SIZE}."
            )
        if intermediate_size_per_partition % MXFP8_GROUP_SIZE != 0:
            raise ValueError(
                "MoE intermediate_size_per_partition="
                f"{intermediate_size_per_partition} must be divisible by "
                f"{MXFP8_GROUP_SIZE}."
            )

        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
            intermediate_size_per_partition,
            is_aiter_moe=False,
            is_concat=True,
            is_packed=False,
        )
        if w13_up_dim % MXFP8_GROUP_SIZE != 0:
            raise ValueError(
                f"w13 K dimension={w13_up_dim} must be divisible by {MXFP8_GROUP_SIZE}."
            )
        if w2_down_dim % MXFP8_GROUP_SIZE != 0:
            raise ValueError(
                f"w2 K dimension={w2_down_dim} must be divisible by {MXFP8_GROUP_SIZE}."
            )

        extra_weight_attrs.update({"weight_padded": weight_padded})

        # Checkpoint layout:
        #   w13: [E, 2I, H], w13_scale: [E, 2I, H / 32]
        #   w2:  [E, H, I],  w2_scale:  [E, H, I / 32]
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w2_down_dim,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Preserve uint8 storage while loading: checkpoint UE8M0 bytes are
        # consumed as raw bytes by the grouped MXFP8 sgl-kernel path.
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size // MXFP8_GROUP_SIZE,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        # MXFP8 stores per-32 K-element scales, which maps to GROUP loading.
        scale_weight_attrs = dict(extra_weight_attrs)
        scale_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, scale_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w2_down_dim // MXFP8_GROUP_SIZE,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_weight_attrs)

        # MXFP8 activations are dynamically generated per expert GEMM.
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.w13_weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "MXFP8 MoE w13_weight must be torch.float8_e4m3fn; "
                f"got {layer.w13_weight.dtype}."
            )
        if layer.w2_weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "MXFP8 MoE w2_weight must be torch.float8_e4m3fn; "
                f"got {layer.w2_weight.dtype}."
            )
        if layer.w13_weight_scale.dtype != torch.uint8:
            raise TypeError(
                "MXFP8 MoE w13_weight_scale must contain UE8M0 uint8 bytes; "
                f"got {layer.w13_weight_scale.dtype}."
            )
        if layer.w2_weight_scale.dtype != torch.uint8:
            raise TypeError(
                "MXFP8 MoE w2_weight_scale must contain UE8M0 uint8 bytes; "
                f"got {layer.w2_weight_scale.dtype}."
            )

        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data.contiguous(),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight.data.contiguous(),
            requires_grad=False,
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data.contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data.contiguous(),
            requires_grad=False,
        )

    def create_moe_runner(
        self,
        layer: torch.nn.Module,
        moe_runner_config: MoeRunnerConfig,
    ) -> None:
        del layer
        self.moe_runner_config = moe_runner_config
        self._cutlass_buffers_ready = False

    def _ensure_cutlass_buffers_initialized(self, layer: torch.nn.Module) -> None:
        device = layer.w13_weight.device
        num_experts = layer.num_local_experts
        hidden_size = layer.w2_weight.shape[1]
        intermediate_size_per_partition = layer.w2_weight.shape[2]

        if getattr(self, "_cutlass_buffers_ready", False):
            if (
                self.ab_strides1.device == device
                and self.ab_strides1.numel() == num_experts
                and self.ab_strides1[0].item() == hidden_size
                and self.c_strides1[0].item() == 2 * intermediate_size_per_partition
                and self.ab_strides2[0].item() == intermediate_size_per_partition
                and self.c_strides2[0].item() == hidden_size
            ):
                return

        self.ab_strides1 = torch.full(
            (num_experts,), hidden_size, device=device, dtype=torch.int64
        )
        self.c_strides1 = torch.full(
            (num_experts,),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.ab_strides2 = torch.full(
            (num_experts,),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(
            (num_experts,), hidden_size, device=device, dtype=torch.int64
        )

        self.workspace = torch.empty(90000, device=device, dtype=torch.uint8)
        self.a_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.b_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.out_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.a_scales_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.b_scales_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.expert_offsets = torch.empty(
            num_experts + 1, device=device, dtype=torch.int32
        )
        self.problem_sizes1 = torch.empty(
            num_experts, 3, device=device, dtype=torch.int32
        )
        self.problem_sizes2 = torch.empty(
            num_experts, 3, device=device, dtype=torch.int32
        )

        self._cutlass_buffers_ready = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        try:
            from sgl_kernel import cutlass_fused_experts_fp8
        except ImportError:
            # Keep a local fallback for environments where the symbol is only
            # exposed via the SGLang wrapper module.
            from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8

        config = self.moe_runner_config
        if config.activation != "silu":
            raise NotImplementedError(
                "MXFP8 cutlass_fused_experts_fp8 path currently supports only SiLU."
            )
        if config.no_combine:
            raise NotImplementedError(
                "MXFP8 cutlass_fused_experts_fp8 path does not support no_combine."
            )
        if config.apply_router_weight_on_input:
            raise NotImplementedError(
                "apply_router_weight_on_input is unsupported for MXFP8 cutlass MoE."
            )

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        if x.ndim != 2:
            raise ValueError(
                "MXFP8 MoE expects flattened hidden states [num_tokens, hidden_size], "
                f"got {tuple(x.shape)}."
            )

        # topk_ids are global IDs while this rank owns local expert weights only.
        local_expert_offset = layer.moe_ep_rank * layer.num_local_experts
        topk_ids_local = topk_ids.to(torch.int32) - local_expert_offset
        local_mask = (topk_ids_local >= 0) & (topk_ids_local < layer.num_local_experts)
        if not local_mask.all().item():
            topk_ids_local = torch.where(
                local_mask,
                topk_ids_local,
                torch.zeros_like(topk_ids_local),
            )
            topk_weights = topk_weights * local_mask.to(topk_weights.dtype)

        self._ensure_cutlass_buffers_initialized(layer)

        output = cutlass_fused_experts_fp8(
            x,
            layer.w13_weight.transpose(1, 2),
            layer.w2_weight.transpose(1, 2),
            layer.w13_weight_scale.transpose(1, 2),
            layer.w2_weight_scale.transpose(1, 2),
            topk_weights,
            topk_ids_local,
            self.ab_strides1,
            self.c_strides1,
            self.ab_strides2,
            self.c_strides2,
            self.workspace,
            self.a_ptr,
            self.b_ptr,
            self.out_ptr,
            self.a_scales_ptr,
            self.b_scales_ptr,
            self.expert_offsets,
            self.problem_sizes1,
            self.problem_sizes2,
            use_fp8_blockscale=True,
            use_mxfp8=True,
        )

        return StandardCombineInput(hidden_states=output)
