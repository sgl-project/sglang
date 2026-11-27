"""Reusable model-layer integration for the small-batch and SM100 NVFP4 dual GEMM kernels."""

import os
import sys
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.srt.utils import is_cuda, is_sm100_supported

if TYPE_CHECKING:
    from sglang.kernels.ops.gemm.cutedsl_dual_gemm import DualGemmQuantMode
    from sglang.srt.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )


class DualGemm:
    """Dispatch fused gate/up projection and activation for compatible layers."""

    def __init__(
        self,
        gate_up_proj: "MergedColumnParallelLinear",
        down_proj: "RowParallelLinear",
        hidden_size: int,
    ) -> None:
        self.down_proj = down_proj
        try:
            self.tp_size = get_parallel().tp_size
        except ValueError:
            self.tp_size = 1
        self.max_tokens = 0
        self.nvfp4_eligible = False
        self.mode = self._select_mode(gate_up_proj, hidden_size)

    def _select_mode(
        self,
        gate_up_proj: "MergedColumnParallelLinear",
        hidden_size: int,
    ) -> Optional["DualGemmQuantMode"]:
        if not is_cuda():
            return None

        local_intermediate_size = gate_up_proj.output_partition_sizes[0]
        gate_up_method = gate_up_proj.quant_method
        if type(gate_up_method).__name__ == "ModelOptFp4LinearMethod":
            if (
                is_sm100_supported()
                and getattr(gate_up_method, "quant_mode", None) == "w4a4"
                and hidden_size % 256 == 0
                and local_intermediate_size % 64 == 0
                and not getattr(gate_up_proj, "_interleave_for_swiglu_fusion", False)
            ):
                self.nvfp4_eligible = True
            return None

        from sglang.kernels.ops.gemm.cutedsl_dual_gemm import (
            MAX_DUAL_GEMM_DECODE_TOKENS,
            DualGemmQuantMode,
            can_use_dual_gemm,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

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
        if self.nvfp4_eligible:
            if isinstance(x, tuple) or x.ndim != 2:
                return False
            if os.environ.get("SGLANG_USE_CUTEDSL_NVFP4_DUAL_GEMM", "1") != "1":
                return False
            if getattr(gate_up_proj, "weights_padding_cols", 0) != 0:
                return False
            if not hasattr(gate_up_proj, "weight_scale_interleaved"):
                return False
            fp4_utils_mod = sys.modules.get("sglang.srt.layers.quantization.fp4_utils")
            if fp4_utils_mod is not None and hasattr(
                fp4_utils_mod, "get_fp4_gemm_runner_backend"
            ):
                if not fp4_utils_mod.get_fp4_gemm_runner_backend().supports_swiglu_fusion():
                    return False
            min_m = int(os.environ.get("SGLANG_CUTEDSL_MIN_M", "128"))
            return (
                x.shape[0] >= min_m
                and not hasattr(gate_up_proj, "base_layer")
                and not (self.tp_size > 1 and get_forward().sp_active)
            )

        input_tensor = x[0] if isinstance(x, tuple) else x
        return (
            self.mode is not None
            and 1 <= input_tensor.shape[0] <= self.max_tokens
            and not hasattr(gate_up_proj, "base_layer")
            and not (self.tp_size > 1 and get_forward().sp_active)
        )

    def _call_nvfp4(
        self, x: torch.Tensor, gate_up_proj: "MergedColumnParallelLinear"
    ) -> torch.Tensor:
        from sglang.kernels.ops.gemm import nvfp4_dual_gemm_swiglu

        fp4_utils_mod = sys.modules.get("sglang.srt.layers.quantization.fp4_utils")
        if (
            fp4_utils_mod is not None
            and getattr(fp4_utils_mod, "fp4_quantize", None) is not None
        ):
            fp4_quantize = fp4_utils_mod.fp4_quantize
        else:
            from flashinfer import fp4_quantize as _fi_fp4_quantize

            def fp4_quantize(inp: torch.Tensor, global_scale: torch.Tensor):
                return _fi_fp4_quantize(inp, global_scale, backend="cute-dsl")

        w_ptr = gate_up_proj.weight.data_ptr()
        s_ptr = gate_up_proj.weight_scale_interleaved.data_ptr()
        if (
            getattr(gate_up_proj, "_cutedsl_w_ptr", None) != w_ptr
            or getattr(gate_up_proj, "_cutedsl_s_ptr", None) != s_ptr
        ):
            b1, b2 = gate_up_proj.weight.chunk(2, dim=0)
            n_half = b1.shape[0]
            w_scale = gate_up_proj.weight_scale_interleaved
            if n_half % 128 == 0:
                sfb1, sfb2 = w_scale.chunk(2, dim=0)
                sfb1 = sfb1.contiguous()
                sfb2 = sfb2.contiguous()
            else:
                from sglang.kernels.ops.gemm.cutedsl_nvfp4_dual_gemm import (
                    permute_nvfp4_scale,
                )

                k_padded = w_scale.shape[1]
                raw_2n = (
                    w_scale.view((2 * n_half) // 128, k_padded // 4, 32, 4, 4)
                    .permute(0, 3, 2, 1, 4)
                    .reshape(2 * n_half, k_padded)
                )
                sfb1 = permute_nvfp4_scale(raw_2n[:n_half])
                sfb2 = permute_nvfp4_scale(raw_2n[n_half:])
            gate_up_proj._cutedsl_b1 = b1.contiguous()
            gate_up_proj._cutedsl_b2 = b2.contiguous()
            gate_up_proj._cutedsl_sfb1 = sfb1
            gate_up_proj._cutedsl_sfb2 = sfb2
            gate_up_proj._cutedsl_w_ptr = w_ptr
            gate_up_proj._cutedsl_s_ptr = s_ptr

        orig_m = x.shape[0]
        out_dtype = x.dtype
        if getattr(gate_up_proj.quant_config, "is_awq", False):
            x = x * gate_up_proj.pre_quant_scale

        a_fp4, sfa = fp4_quantize(x, gate_up_proj.input_scale_inv)
        pad_m = ((orig_m + 127) // 128) * 128
        if pad_m != orig_m:
            a_fp4 = torch.nn.functional.pad(a_fp4, (0, 0, 0, pad_m - orig_m))

        rcp_env = os.environ.get("SGLANG_CUTEDSL_HYBRID_RCP_MOD")
        nr_env = os.environ.get("SGLANG_CUTEDSL_NR_STEPS")
        rcp_override = int(rcp_env) if rcp_env is not None else None
        nr_override = int(nr_env) if nr_env is not None else None

        out = nvfp4_dual_gemm_swiglu(
            a_fp4,
            gate_up_proj._cutedsl_b1,
            gate_up_proj._cutedsl_b2,
            sfa,
            gate_up_proj._cutedsl_sfb1,
            gate_up_proj._cutedsl_sfb2,
            alpha=gate_up_proj.alpha,
            swizzled_scales=True,
            out_dtype=out_dtype,
            enable_pdl=True,
            pdl_safe=True,
            hybrid_rcp_mod_override=rcp_override,
            nr_steps_override=nr_override,
        )
        return out[:orig_m] if pad_m != orig_m else out

    def __call__(self, x, gate_up_proj: "MergedColumnParallelLinear"):
        if self.nvfp4_eligible and self.mode is None:
            return self._call_nvfp4(x, gate_up_proj)

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
