"""Dense activation/quantization fusion for the block-FP8 DeepGEMM path."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch

from sglang.kernels.fused_op import get_fused_op_backend
from sglang.srt.runtime_context import get_exec, get_forward, get_lora, get_platform
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.linear import RowParallelLinear


class DenseSiluAndMulQuant:
    """Keep the normal activation unless down_proj can consume our FP8 tuple."""

    def __init__(self, down_proj: RowParallelLinear) -> None:
        self.enabled = False
        self.down_proj = down_proj
        self.quant_method = down_proj.quant_method
        platform = get_platform()
        if not (is_cuda() and (platform.is_sm90 or platform.is_sm100)):
            return

        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.linear import RowParallelLinear
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
        from sglang.srt.layers.quantization.fp8_utils import (
            _deepgemm_block_fp8_supported,
            _fp8_block_gemv_enabled,
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

        # Match the linear's own activation quant: packed UE8M0 on SM100,
        # fp32 column-major TMA-aligned on SM90.
        self.scale_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        method = self.quant_method
        if not (
            type(down_proj) is RowParallelLinear
            and type(method) is Fp8LinearMethod
            and method.weight_block_size == [128, 128]
            and method.quant_config.activation_scheme == "dynamic"
            and not (method.use_marlin or method.use_mxfp8 or method.block_fp8_as_mxfp8)
            and (self.scale_ue8m0 or platform.is_sm90)
            and down_proj.input_is_parallel
        ):
            return
        backend = method.w8a8_block_fp8_linear
        if isinstance(backend, partial):
            if backend.args or set(backend.keywords) - {"act_scale_ue8m0"}:
                return
            # SM90 quantizes these checkpoints with row-major power-of-two scales.
            if platform.is_sm90:
                return
            backend = backend.func
        # The SM90 GEMV takes row-major scales.
        if _fp8_block_gemv_enabled():
            return
        # The supported loader preserves matrix geometry, not tensor identity/device.
        self.weight_shape = (down_proj.output_size, down_proj.input_size_per_partition)
        self.enabled = (
            backend is deepgemm_w8a8_block_fp8_linear_with_fallback
            and _deepgemm_block_fp8_supported(
                self.weight_shape, method.weight_block_size, torch.bfloat16
            )
        )

    def __call__(
        self,
        x: torch.Tensor,
        act_fn: SiluAndMul,
        down_proj: RowParallelLinear,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
            scale_ue8m0=self.scale_ue8m0,
            column_major_scales=True,
            fuse_silu_and_mul=True,
            silu_and_mul_mode="dense",
        )
