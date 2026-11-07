"""Dense activation/quantization fusion for the block-FP8 DeepGEMM path and the
gfx95 aiter bpreshuffle block-FP8 path."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch

from sglang.kernels.fused_op import get_fused_op_backend
from sglang.srt.runtime_context import (
    get_exec,
    get_forward,
    get_lora,
    get_parallel,
    get_platform,
)
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.linear import RowParallelLinear


def aiter_takes_fp8_group_input(linear) -> bool:
    """Whether ``linear`` runs aiter's bpreshuffle block-FP8 GEMM, so it can take
    a pre-quantized ``(fp8, 1x128 group scale)`` input. Valid only after weight
    loading, which is where the bpreshuffle is applied."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    method = getattr(linear, "quant_method", None)
    weight = getattr(linear, "weight", None)
    return (
        getattr(linear, "aiter_bpreshuffled", False)
        and type(method) is Fp8LinearMethod
        and not (method.use_marlin or method.use_mxfp8 or method.block_fp8_as_mxfp8)
        and getattr(weight, "_ptpc_weight", None) is None
        and getattr(weight, "_mxscale_w_scale", None) is None
        and not hasattr(linear, "base_layer")
    )


def aiter_prefers_bf16_input(linear, num_tokens: int) -> bool:
    """Whether ``linear`` runs ``num_tokens`` rows faster on the gfx950 small-M
    GEMM, which quantizes its bf16 input itself, than on a pre-quantized input;
    a producer then keeps its output in bf16."""
    from sglang.kernels.ops.gemm import smallm_fp8_blockscale_gfx950 as smallm

    n, k = linear.weight.shape
    return linear.bias is None and smallm.prefers_bf16_input(num_tokens, n, k)


def fp8_group_quant_fusion_allowed() -> bool:
    """Whether this forward may hand an aiter bpreshuffle block-FP8 linear an
    input its producer already quantized, as the fused norm reads and the fused
    SiLU-and-mul do. The producers share this gate so they turn off together."""
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    deterministic = get_exec().deterministic
    return not (
        torch.compiler.is_compiling()
        or deterministic.enable_deterministic_inference
        or deterministic.rl_on_policy_target is not None
        or is_batch_invariant_mode_enabled()
        or get_lora().enable_lora
    )


class DenseSiluAndMulQuant:
    """Keep the normal activation unless down_proj can consume our FP8 tuple."""

    def __init__(self, down_proj: RowParallelLinear) -> None:
        self.enabled = False
        self.down_proj = down_proj
        self.quant_method = down_proj.quant_method
        self.aiter = is_hip() and get_bool_env_var("SGLANG_USE_AITER")
        # Resolved on the first decode: the bpreshuffle happens at weight load.
        self.aiter_enabled = None
        if not (is_cuda() and get_platform().is_sm100 and get_parallel().tp_size == 1):
            return

        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.linear import RowParallelLinear
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
        from sglang.srt.layers.quantization.fp8_utils import (
            _deepgemm_block_fp8_supported,
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

        method = self.quant_method
        if not (
            type(down_proj) is RowParallelLinear
            and type(method) is Fp8LinearMethod
            and method.weight_block_size == [128, 128]
            and method.quant_config.activation_scheme == "dynamic"
            and not (method.use_marlin or method.use_mxfp8 or method.block_fp8_as_mxfp8)
            and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and down_proj.input_is_parallel
        ):
            return
        backend = method.w8a8_block_fp8_linear
        if isinstance(backend, partial):
            if backend.args or set(backend.keywords) - {"act_scale_ue8m0"}:
                return
            backend = backend.func
        # The supported loader preserves matrix geometry, not tensor identity/device.
        self.weight_shape = (down_proj.output_size, down_proj.input_size_per_partition)
        self.enabled = (
            backend is deepgemm_w8a8_block_fp8_linear_with_fallback
            and _deepgemm_block_fp8_supported(
                self.weight_shape, method.weight_block_size, torch.bfloat16
            )
        )

    def forward_decode(
        self,
        x: torch.Tensor,
        act_fn: SiluAndMul,
        down_proj: RowParallelLinear,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Try fusion for decode; callers must bypass this helper in other modes."""
        if self.aiter:
            return self._forward_decode_aiter(x, act_fn, down_proj)
        if not self.enabled or torch.compiler.is_compiling():
            return act_fn(x)

        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
        from sglang.srt.layers.activation import SiluAndMul

        deterministic = get_exec().deterministic
        if (
            down_proj is not self.down_proj
            or down_proj.quant_method is not self.quant_method
            or down_proj.weight.shape != self.weight_shape
            or type(act_fn) is not SiluAndMul
            or (act_fn._forward_method or act_fn._resolve_forward_method())
            != act_fn.forward_cuda
            or act_fn.is_torch_compile
            or get_fused_op_backend() is not None
            or deterministic.enable_deterministic_inference
            or deterministic.rl_on_policy_target is not None
            or is_batch_invariant_mode_enabled()
            or get_lora().enable_lora
            or get_forward().sp_active
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or x.ndim != 2
            or x.shape[0] == 0
            or not x.is_contiguous()
            or x.data_ptr() % 32 != 0
            or x.stride(0) % 16 != 0
            or x.device != down_proj.weight.device
            or x.shape[1] != 2 * self.weight_shape[1]
        ):
            return act_fn(x)

        from sglang.kernels.ops.quantization.per_token_group_quant import (
            per_token_group_quant,
        )

        return per_token_group_quant(
            x,
            group_size=128,
            scale_ue8m0=True,
            column_major_scales=True,
            fuse_silu_and_mul=True,
            silu_and_mul_mode="dense",
        )

    def _forward_decode_aiter(
        self,
        x: torch.Tensor,
        act_fn: SiluAndMul,
        down_proj: RowParallelLinear,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from sglang.srt.layers.activation import SiluAndMul
        from sglang.srt.layers.quantization.fp8_utils import (
            emit_transposed_bpreshuffle_scale,
            view_aiter_fused_rms_transposed_fp8_scale,
        )

        if self.aiter_enabled is None:
            self.aiter_enabled = bool(
                aiter_takes_fp8_group_input(self.down_proj)
                and self.down_proj.input_size_per_partition % 128 == 0
            )
        if (
            not self.aiter_enabled
            or down_proj is not self.down_proj
            or type(act_fn) is not SiluAndMul
            or act_fn.is_torch_compile
            or not fp8_group_quant_fusion_allowed()
            or x.dtype != torch.bfloat16
            or x.ndim != 2
            or x.shape[0] == 0
            or x.stride(1) != 1
            or x.shape[1] != 2 * down_proj.input_size_per_partition
            or aiter_prefers_bf16_input(down_proj, x.shape[0])
        ):
            return act_fn(x)

        from sglang.kernels.ops.quantization.silu_mul_fp8_group_quant import (
            silu_mul_fp8_group_quant,
        )

        transpose = emit_transposed_bpreshuffle_scale(
            x.shape[0], on_bpreshuffle_gfx95=True
        )
        q, scale = silu_mul_fp8_group_quant(x, transpose_scale=transpose)
        if transpose:
            scale = view_aiter_fused_rms_transposed_fp8_scale(scale)
        return q, scale
