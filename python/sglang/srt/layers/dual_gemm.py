"""Reusable model-layer integration for the small-batch dual GEMM kernel."""

from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.kernels.arch import current_arch
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_forward, get_parallel
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.kernels.ops.gemm.cutedsl_dual_gemm import DualGemmQuantMode
    from sglang.srt.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )

# DualGemm.mode for the SM120 QuACK GEMM; the other modes are the CuteDSL kernel's.
QUACK_SWIGLU = "quack_swiglu"


class DualGemm:
    """Dispatch fused gate/up projection and activation for compatible layers."""

    def __init__(
        self,
        gate_up_proj: "MergedColumnParallelLinear",
        down_proj: "RowParallelLinear",
        hidden_size: int,
    ) -> None:
        self.down_proj = down_proj
        self.tp_size = get_parallel().tp_size
        self.max_tokens = 0
        # None turns DualGemm off.
        self.mode: Optional[Union[DualGemmQuantMode, str]] = (
            QUACK_SWIGLU
            if self._use_quack_swiglu_gemm(gate_up_proj)
            else self._select_mode(gate_up_proj, hidden_size)
        )

    @staticmethod
    def _use_quack_swiglu_gemm(gate_up_proj: "MergedColumnParallelLinear") -> bool:
        if not is_cuda():
            return False

        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        return (
            isinstance(gate_up_proj.quant_method, UnquantizedLinearMethod)
            and envs.SGLANG_USE_QUACK_SWIGLU_GEMM.get()
            and current_arch().is_sm(12, 0)
            # Batch invariance overrides aten::mm, which the fused GEMM never calls.
            and not get_exec().deterministic.enable_deterministic_inference
            # On-policy RL must match the trainer's unfused bf16 SwiGLU bit for bit.
            and get_exec().deterministic.rl_on_policy_target is None
        )

    def _select_mode(
        self,
        gate_up_proj: "MergedColumnParallelLinear",
        hidden_size: int,
    ) -> Optional["DualGemmQuantMode"]:
        if not is_cuda():
            return None

        from sglang.kernels.ops.gemm.cutedsl_dual_gemm import (
            MAX_DUAL_GEMM_DECODE_TOKENS,
            DualGemmQuantMode,
            can_use_dual_gemm,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        gate_up_method = gate_up_proj.quant_method
        if isinstance(gate_up_method, UnquantizedLinearMethod):
            mode = (
                DualGemmQuantMode.UNQUANT
                if gate_up_proj.params_dtype in (torch.bfloat16, torch.float16)
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

            layers = (gate_up_proj, self.down_proj)
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
            if gate_up_proj.params_dtype in (
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

        local_intermediate_size = gate_up_proj.output_partition_sizes[0]
        if mode is not None and can_use_dual_gemm(
            MAX_DUAL_GEMM_DECODE_TOKENS,
            hidden_size,
            local_intermediate_size,
            mode,
        ):
            self.max_tokens = MAX_DUAL_GEMM_DECODE_TOKENS
            return mode
        return None

    def can_run(self, x, gate_up_proj: "MergedColumnParallelLinear") -> bool:
        input_tensor = x[0] if isinstance(x, tuple) else x
        if (
            self.mode is None
            or hasattr(gate_up_proj, "base_layer")
            or (self.tp_size > 1 and get_forward().sp_active)
        ):
            return False
        if self.mode == QUACK_SWIGLU:
            from sglang.kernels.ops.gemm.quack_swiglu_gemm import (
                quack_swiglu_gemm_supported,
            )

            return not torch.compiler.is_compiling() and quack_swiglu_gemm_supported(
                input_tensor, gate_up_proj.weight
            )
        return 1 <= input_tensor.shape[0] <= self.max_tokens

    def __call__(self, x, gate_up_proj: "MergedColumnParallelLinear"):
        if self.mode == QUACK_SWIGLU:
            from sglang.kernels.ops.gemm import dual_gemm_swiglu_quack

            return dual_gemm_swiglu_quack(x, gate_up_proj.weight)

        if not self.mode.is_quantized:
            from sglang.kernels.ops.gemm import dual_gemm_swiglu

            return dual_gemm_swiglu(x, gate_up_proj.weight)

        from sglang.kernels.ops.gemm import dual_gemm_swiglu_fp8

        if isinstance(x, tuple):
            quantized_x, x_scale = x[:2]
            output_dtype = x[2] if len(x) > 2 else torch.bfloat16
        else:
            from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant

            output_dtype = x.dtype
            quantized_x, x_scale = scaled_fp8_quant(
                x,
                gate_up_proj.input_scale,
                use_per_token_if_dynamic=True,
            )

        quantized_activation, activation_scale = dual_gemm_swiglu_fp8(
            quantized_x,
            gate_up_proj.weight.T,
            x_scale,
            gate_up_proj.weight_scale,
            self.down_proj.input_scale,
            quant_mode=self.mode,
        )
        # The down projection consumes this tuple without quantizing again. The
        # original dtype controls its output type.
        return quantized_activation, activation_scale, output_dtype
