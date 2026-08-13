from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

from sglang.srt.utils import set_weight_attrs
from sglang.srt.utils.common import is_sm90_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


class Mxfp4CutlassMoEMethod:
    """MXFP4A8 (weight E2M1 + block=32 E8M0 scale, activation FP8 e4m3) MoE
    method for sglang's own CUTLASS w4a8 grouped-GEMM backend (SM90/Hopper).

    Wraps the FP8 fp4-expert method the same way ``Mxfp4MarlinMoEMethod`` does:
    it consumes the standard fp4-expert checkpoint layout (int8 packed E2M1
    weights ``[E, 2*I, K//2]`` / ``[E, K, I//2]`` in native ``[gate; up]`` order,
    plus block=32 group scales), then at load time passes the packed nibbles
    through unchanged (the HF-natural byte layout is exactly what the kernel's
    per-nibble decode expects; see ``repack_hf_mxfp4_to_kernel``) and expands the
    group scale to bf16, and finally dispatches to ``cutlass_mxfp4a8_moe``.

    The int4a8 path is untouched; this is a parallel format.
    """

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix
        if not is_sm90_supported():
            raise RuntimeError(
                "moe_runner_backend=cutlass MXFP4A8 requires SM90 (Hopper)."
            )

    def create_moe_runner(self, layer, moe_runner_config):
        # The cutlass MXFP4A8 path calls its kernel directly from apply(), so no
        # MoeRunner abstraction is constructed (mirrors W4AFp8MoEMethod).
        self.moe_runner_config = moe_runner_config

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

        fp4_block_k = 32
        # CUTLASS grouped-GEMM dims must be % 128 == 0; DeepSeek-V4 experts
        # already satisfy this, so the checkpoint's native [gate; up] layout is
        # kept unpadded (identical to the int4a8 w4afp8 loader).
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition

        # Packed E2M1 weights: two 4-bit codes per int8 byte.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
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

        # Block=32 group scales, stored under the ``_inv`` names the fp4-expert
        # loader writes to. fp32 container holds the numerical 2**e directly;
        # normalized to bf16 in process_weights_after_loading.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

        self._create_cutlass_strides(
            layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size_per_partition,
        )

    def _create_cutlass_strides(
        self, layer, num_experts, hidden_size, intermediate_size
    ):
        """Pre-populate the per-expert CUTLASS grouped-GEMM strides / offsets,
        mirroring ``W4AFp8MoEMethod.create_weights`` (int4a8)."""
        device = layer.w13_weight.device
        self.a_strides1 = torch.full(
            (num_experts, 3), hidden_size, device=device, dtype=torch.int64
        )
        self.c_strides1 = torch.full(
            (num_experts, 3), 2 * intermediate_size, device=device, dtype=torch.int64
        )
        self.a_strides2 = torch.full(
            (num_experts, 3), intermediate_size, device=device, dtype=torch.int64
        )
        self.c_strides2 = torch.full(
            (num_experts, 3), hidden_size, device=device, dtype=torch.int64
        )
        self.b_strides1 = self.a_strides1
        self.s_strides13 = self.c_strides1
        self.b_strides2 = self.a_strides2
        self.s_strides2 = self.c_strides2
        self.expert_offsets = torch.empty(
            (num_experts + 1), dtype=torch.int32, device=device
        )
        self.problem_sizes1 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )
        self.problem_sizes2 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        """Convert the fp4-expert checkpoint weights into the sglang CUTLASS
        w4a8 MXFP4A8 kernel layout (reusing the int4a8 DirectConvert mainloop).

        1. E2M1 weights: pass the HF-packed nibbles through unchanged (viewed as
           int8). The HF-natural byte layout is exactly what the kernel's
           per-nibble decode expects; see ``repack_hf_mxfp4_to_kernel``. Applying
           the int4a8 ``order_map`` reorder here was verified to produce garbage
           (rel_mean ~= 1.2), so no reorder is done.
        2. Group scales: normalize to the numerical 2**e value in bf16 (reusing
           marlin's ``_normalize_scale_tensor`` to stay agnostic of the loader's
           placeholder dtype), then 4-wide ``interleave_scales`` for the post-MMA
           bf16 group-scale path.
        The checkpoint's native ``[gate; up]`` order already matches the kernel,
        so no de-interleave is needed.
        """
        from sglang.srt.layers.mxfp4a8_utils import repack_hf_mxfp4_to_kernel
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            _normalize_scale_tensor,
        )
        from sglang.srt.layers.quantization.w4afp8 import interleave_scales

        # --- weights: HF-natural nibble packing passed through as int8 ---
        w13 = repack_hf_mxfp4_to_kernel(layer.w13_weight.data).contiguous()
        w2 = repack_hf_mxfp4_to_kernel(layer.w2_weight.data).contiguous()
        layer.w13_weight = Parameter(w13, requires_grad=False)
        layer.w2_weight = Parameter(w2, requires_grad=False)

        # --- scales: -> numerical 2**e in bf16, then 4-wide interleave ---
        # 4-wide matches the mxfp4 kernel PackedScalesNum = TileK(128)/GroupSize(32).
        w13_scale = _normalize_scale_tensor(
            layer.w13_weight_scale_inv.data, torch.bfloat16
        )
        w13_scale = interleave_scales(w13_scale.contiguous(), group=4)
        layer.w13_weight_scale = Parameter(w13_scale, requires_grad=False)

        w2_scale = _normalize_scale_tensor(
            layer.w2_weight_scale_inv.data, torch.bfloat16
        )
        w2_scale = interleave_scales(w2_scale.contiguous(), group=4)
        layer.w2_weight_scale = Parameter(w2_scale, requires_grad=False)

        layer._dsv4_mxfp4_backend = "cutlass"

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.cutlass_mxfp4a8_moe import cutlass_mxfp4a8_moe
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        output = cutlass_mxfp4a8_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            self.a_strides1,
            self.b_strides1,
            self.c_strides1,
            self.a_strides2,
            self.b_strides2,
            self.c_strides2,
            self.s_strides13,
            self.s_strides2,
            self.expert_offsets,
            self.problem_sizes1,
            self.problem_sizes2,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,
            swiglu_limit=self.moe_runner_config.swiglu_limit,
        )
        return StandardCombineInput(hidden_states=output)
