from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

from sglang.srt.utils import log_info_on_rank0, set_weight_attrs
from sglang.srt.utils.common import is_sm90_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


def _preprocess_fused_weights(weight, raw_scale):
    from flashinfer.fused_moe import (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming,
    )

    # Humming's payload rewrite expands FP4 codes and scales into int64 indices.
    # Bound those load-time temporaries by processing independent experts.
    outputs = None
    for start in range(0, weight.shape[0], 32):
        stop = min(start + 32, weight.shape[0])
        chunk = preprocess_moe_weights_for_sm90_mixed_gemm_humming(
            weight[start:stop], raw_scale[start:stop]
        )
        if outputs is None:
            outputs = tuple(
                tensor.new_empty((weight.shape[0], *tensor.shape[1:]))
                for tensor in chunk
            )
        for output, tensor in zip(outputs, chunk):
            output[start:stop].copy_(tensor)
        del chunk, tensor
    return outputs


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
        # FusedMoE exposes this optional runner for overlap hooks.
        self.runner = None

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

        if layer.moe_ep_size != 1:
            raise ValueError("CUTLASS MXFP4 fused MoE currently requires --ep-size 1.")

        fp4_block_k = 32
        # Load the native shard first; pad gate/up halves independently after
        # loading so TP slicing still uses the checkpoint's intermediate size.
        if hidden_size % 128 or intermediate_size_per_partition % fp4_block_k:
            raise ValueError(
                "CUTLASS MXFP4 requires hidden % 128 and intermediate % 32."
            )
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self._use_legacy = layer.moe_ep_size > 1

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

        # Keep checkpoint scales in native E8M0, as the FlashInfer loader does.
        # The legacy path expands them to BF16 only after loading.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
                dtype=torch.float8_e8m0fnu,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
                dtype=torch.float8_e8m0fnu,
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
        source_versions = (
            layer.w13_weight._version,
            layer.w2_weight._version,
            layer.w13_weight_scale_inv._version,
            layer.w2_weight_scale_inv._version,
        )
        if getattr(layer, "_cutlass_mxfp4_source_versions", None) == source_versions:
            return

        from sglang.srt.layers.mxfp4a8_utils import repack_hf_mxfp4_to_kernel
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            _normalize_scale_tensor,
        )
        from sglang.srt.layers.quantization.mxfp4_padding import (
            pad_mxfp4_moe_intermediate,
        )
        from sglang.srt.layers.quantization.w4afp8 import interleave_scales

        padded = pad_mxfp4_moe_intermediate(
            layer.w13_weight.data,
            layer.w2_weight.data,
            layer.w13_weight_scale_inv.data,
            layer.w2_weight_scale_inv.data,
        )
        for name, tensor in zip(
            ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"),
            padded,
        ):
            getattr(layer, name).data = tensor
        del padded
        self._create_cutlass_strides(
            layer,
            num_experts=layer.w13_weight.shape[0],
            hidden_size=self.hidden_size,
            intermediate_size=layer.w2_weight.shape[-1] * 2,
        )

        def _raw_e8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
            if scale.dtype == torch.float8_e8m0fnu:
                return scale.view(torch.uint8).contiguous()
            if scale.dtype == torch.uint8:
                return scale.contiguous()
            if scale.dtype == torch.int8:
                return scale.view(torch.uint8).contiguous()
            return scale.to(torch.float8_e8m0fnu).view(torch.uint8).contiguous()

        # EP1 stores only the fused layout. Keeping both full weight layouts
        # exceeds H20 memory for V4.1 TP8 after intermediate padding.
        if not self._use_legacy:
            log_info_on_rank0(
                logger,
                f"Preparing DSv4 MXFP4 experts for CUTLASS SM90 W4A8 fused EP1 "
                f"(intermediate={self.intermediate_size_per_partition}, "
                f"padded={layer.w2_weight.shape[-1] * 2}, layer={self.prefix})...",
            )
            for stem in ("w13", "w2"):
                weight, offset, residual = _preprocess_fused_weights(
                    getattr(layer, f"{stem}_weight").data.view(torch.uint8),
                    _raw_e8m0_bytes(getattr(layer, f"{stem}_weight_scale_inv").data),
                )
                fused_weight = Parameter(
                    weight.view(torch.int8).contiguous(), requires_grad=False
                )
                fused_scale = Parameter(offset.contiguous(), requires_grad=False)
                setattr(layer, f"{stem}_weight", fused_weight)
                setattr(layer, f"{stem}_weight_fused", fused_weight)
                setattr(layer, f"{stem}_weight_scale_inv", fused_scale)
                setattr(layer, f"{stem}_weight_scale_fused", fused_scale)
                setattr(
                    layer,
                    f"{stem}_weight_residual_fused",
                    Parameter((residual * 64.0).contiguous(), requires_grad=False),
                )
            layer._dsv4_mxfp4_backend = "cutlass"
            layer._cutlass_mxfp4_source_versions = (
                layer.w13_weight._version,
                layer.w2_weight._version,
                layer.w13_weight_scale_inv._version,
                layer.w2_weight_scale_inv._version,
            )
            return

        # Preserve the legacy layout for EP fallback.
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
        layer._cutlass_mxfp4_source_versions = (
            layer.w13_weight._version,
            layer.w2_weight._version,
            layer.w13_weight_scale_inv._version,
            layer.w2_weight_scale_inv._version,
        )

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.cutlass_mxfp4a8_fused_moe import (
            cutlass_mxfp4a8_fused_moe,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        output = cutlass_mxfp4a8_fused_moe(
            x,
            layer.w13_weight if self._use_legacy else None,
            layer.w2_weight if self._use_legacy else None,
            layer.w13_weight_scale if self._use_legacy else None,
            layer.w2_weight_scale if self._use_legacy else None,
            getattr(layer, "w13_weight_fused", None),
            getattr(layer, "w2_weight_fused", None),
            getattr(layer, "w13_weight_scale_fused", None),
            getattr(layer, "w2_weight_scale_fused", None),
            getattr(layer, "w13_weight_residual_fused", None),
            getattr(layer, "w2_weight_residual_fused", None),
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
