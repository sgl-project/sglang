"""Reusable model-layer integration for the small-batch dual GEMM kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.runtime_context import get_forward
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.kernels.ops.gemm.cutedsl_dual_gemm import (
        DualGemmActivationType,
        DualGemmQuantMode,
    )
    from sglang.srt.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )


class DualGemm:
    """Dispatch fused gate/up projection and activation for compatible layers."""

    def __init__(
        self,
        gate_up_proj: MergedColumnParallelLinear,
        down_proj: RowParallelLinear,
        hidden_size: int,
        activation: str,
    ) -> None:
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.max_tokens = 0
        self.activation_type: Optional[DualGemmActivationType] = None
        self.activation = activation
        self.mode = self._select_mode(hidden_size)

    def _select_mode(self, hidden_size: int) -> Optional[DualGemmQuantMode]:
        if not is_cuda():
            return None

        from sglang.kernels.ops.gemm.cutedsl_dual_gemm import (
            MAX_DUAL_GEMM_DECODE_TOKENS,
            DualGemmActivationType,
            DualGemmQuantMode,
            can_use_dual_gemm,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        try:
            self.activation_type = DualGemmActivationType[self.activation.upper()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported dual GEMM activation: {self.activation}"
            ) from exc

        gate_up_method = self.gate_up_proj.quant_method
        if isinstance(gate_up_method, UnquantizedLinearMethod):
            mode = (
                DualGemmQuantMode.UNQUANT
                if self.gate_up_proj.params_dtype in (torch.bfloat16, torch.float16)
                else None
            )
        else:
            from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
                CompressedTensorsLinearMethod,
            )
            from sglang.srt.layers.quantization.compressed_tensors.schemes import (
                CompressedTensorsW8A8Fp8,
            )
            from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

            layers = (self.gate_up_proj, self.down_proj)
            native_fp8 = all(
                isinstance(layer.quant_method, Fp8LinearMethod)
                and not (
                    layer.quant_method.block_quant
                    or layer.quant_method.use_marlin
                    or layer.quant_method.use_mxfp8
                )
                for layer in layers
            )
            compressed_fp8 = all(
                isinstance(layer.quant_method, CompressedTensorsLinearMethod)
                and isinstance(layer.scheme, CompressedTensorsW8A8Fp8)
                and layer.scheme.weight_block_size is None
                for layer in layers
            )
            if self.gate_up_proj.params_dtype in (
                torch.bfloat16,
                torch.float16,
            ) and (native_fp8 or compressed_fp8):
                mode = (
                    DualGemmQuantMode.DYNAMIC_PER_TOKEN
                    if getattr(self.down_proj, "input_scale", None) is None
                    else DualGemmQuantMode.STATIC_PER_TENSOR
                )
            else:
                mode = None

        local_intermediate_size = self.gate_up_proj.output_partition_sizes[0]
        if mode is not None and can_use_dual_gemm(
            MAX_DUAL_GEMM_DECODE_TOKENS,
            hidden_size,
            local_intermediate_size,
        ):
            self.max_tokens = MAX_DUAL_GEMM_DECODE_TOKENS
            return mode
        return None

    def can_run(self, x) -> bool:
        input_tensor = x[0] if isinstance(x, tuple) else x
        return (
            self.mode is not None
            and 1 <= input_tensor.shape[0] <= self.max_tokens
            and not (self.gate_up_proj.tp_size > 1 and get_forward().sp_active)
        )

    def __call__(self, x):
        if not self.mode.is_quantized:
            from sglang.kernels.ops.gemm import dual_gemm_swiglu

            return dual_gemm_swiglu(
                x, self.gate_up_proj.weight, activation_type=self.activation_type
            )

        from sglang.kernels.ops.gemm import dual_gemm_swiglu_fp8

        if isinstance(x, tuple):
            quantized_x, x_scale = x[:2]
            output_dtype = x[2] if len(x) > 2 else torch.bfloat16
        else:
            from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant

            output_dtype = x.dtype
            quantized_x, x_scale = scaled_fp8_quant(
                x,
                self.gate_up_proj.input_scale,
                use_per_token_if_dynamic=True,
            )

        quantized_activation, activation_scale = dual_gemm_swiglu_fp8(
            quantized_x,
            self.gate_up_proj.weight.T,
            x_scale,
            self.gate_up_proj.weight_scale,
            self.down_proj.input_scale,
            quant_mode=self.mode,
            activation_type=self.activation_type,
        )
        # The down projection consumes this tuple without quantizing again. The
        # original dtype controls its output type.
        return quantized_activation, activation_scale, output_dtype
