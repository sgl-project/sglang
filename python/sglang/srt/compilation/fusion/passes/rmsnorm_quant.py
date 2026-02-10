from typing import Optional

import torch
from sgl_kernel.utils import is_arch_support_pdl
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2

from sglang.srt.compilation.inductor_pass import SGLangPatternMatcherInductorPass
from sglang.srt.utils import direct_register_custom_op, get_bool_env_var

_use_flashinfer_rmsnorm_quant_ops = get_bool_env_var(
    "SGLANG_USE_FLASHINFER_RMSNORM_QUANT_OPS"
)

if _use_flashinfer_rmsnorm_quant_ops:
    import flashinfer

    def _flashinfer_rms_norm_quant(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        return flashinfer.norm.rmsnorm_quant(out, input, weight, scale, eps, enable_pdl)

    def _flashinfer_rms_norm_quant_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float,
        enable_pdl: Optional[bool],
    ) -> None:
        pass

    direct_register_custom_op(
        op_name="flashinfer_rmsnorm_quant",
        op_func=_flashinfer_rms_norm_quant,
        mutates_args=["out"],
        fake_impl=_flashinfer_rms_norm_quant_fake,
    )

    def _flashinfer_fused_add_rmsnorm_quant(
        out: torch.Tensor,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        return flashinfer.norm.fused_add_rmsnorm_quant(
            out, input, residual, weight, scale, eps, enable_pdl
        )

    def _flashinfer_fused_add_rmsnorm_quant_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-6,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        pass

    direct_register_custom_op(
        op_name="flashinfer_fused_add_rmsnorm_quant",
        op_func=_flashinfer_fused_add_rmsnorm_quant,
        mutates_args=["out", "residual"],
        fake_impl=_flashinfer_fused_add_rmsnorm_quant_fake,
    )


class RMSNormQuantPass(SGLangPatternMatcherInductorPass):
    def register_rmsnorm_quant_replacement_pattern(self):
        def pattern(x, rms_result, weight, scale, eps, output):
            rmsnorm = auto_functionalized_v2(
                torch.ops.sgl_kernel.rmsnorm.default,
                input=x,
                weight=weight,
                eps=eps,
                enable_pdl=is_arch_support_pdl(),
                _output_base_index=0,
                _all_bases=[rms_result],
            )

            per_tensor_quant_fp8 = auto_functionalized_v2(
                torch.ops.sglang.per_tensor_quant_fp8.default,
                input=rmsnorm[1],
                is_static=True,
                _output_q_base_index=0,
                _output_s_base_index=1,
                _all_bases=[output, scale],
            )
            return (
                per_tensor_quant_fp8[1],
                per_tensor_quant_fp8[2],
            )

        def replacement(x, rms_result, weight, scale, eps, output):
            if _use_flashinfer_rmsnorm_quant_ops:
                rms_norm_static_fp8_quant = auto_functionalized_v2(
                    torch.ops.sglang.flashinfer_rmsnorm_quant.default,
                    input=x,
                    weight=weight,
                    scale=scale,
                    eps=eps,
                    enable_pdl=is_arch_support_pdl(),
                    _out_base_index=0,
                    _all_bases=[output],
                )
            else:
                rms_norm_static_fp8_quant = auto_functionalized_v2(
                    torch.ops.sgl_kernel.rms_norm_static_fp8_quant.default,
                    input=x,
                    weight=weight,
                    scale=scale,
                    epsilon=eps,
                    _result_base_index=0,
                    _all_bases=[output],
                )
            return rms_norm_static_fp8_quant[1], scale

        example_inputs = [
            torch.empty(16, 16).half().cuda(),
            torch.empty(16, 16).half().cuda(),
            torch.empty(16).half().cuda(),
            torch.empty(()).cuda(),
            torch.empty(16, 16).to(dtype=torch.float8_e4m3fn).cuda(),
        ]

        for eps in self.pass_config.rms_norm_eps:
            self.register_replacement_pattern(
                pattern, replacement, example_inputs, scalar_workaround={"eps": eps}
            )

    def register_fused_add_rmsnorm_quant_replacement_pattern(self):
        def pattern(x, residual, weight, scale, result, eps):
            fused_add_rmsnorm = auto_functionalized_v2(
                torch.ops.sgl_kernel.fused_add_rmsnorm.default,
                weight=weight,
                eps=eps,
                enable_pdl=is_arch_support_pdl(),
                _input_base_index=0,
                _residual_base_index=1,
                _all_bases=[x, residual],
            )

            per_tensor_quant_fp8 = auto_functionalized_v2(
                torch.ops.sglang.per_tensor_quant_fp8.default,
                input=fused_add_rmsnorm[1],
                is_static=True,
                _output_q_base_index=0,
                _output_s_base_index=1,
                _all_bases=[result, scale],
            )

            return (
                per_tensor_quant_fp8[1],
                per_tensor_quant_fp8[2],
                fused_add_rmsnorm[1],
                fused_add_rmsnorm[2],
            )

        def replacement(x, residual, weight, scale, result, eps):
            if _use_flashinfer_rmsnorm_quant_ops:
                fused_add_rms_norm_static_fp8_quant = auto_functionalized_v2(
                    torch.ops.sglang.flashinfer_fused_add_rmsnorm_quant.default,
                    input=x,
                    weight=weight,
                    scale=scale,
                    eps=eps,
                    enable_pdl=is_arch_support_pdl(),
                    _out_base_index=0,
                    _residual_base_index=1,
                    _all_bases=[result, residual],
                )
            else:
                fused_add_rms_norm_static_fp8_quant = auto_functionalized_v2(
                    torch.ops.sgl_kernel.fused_add_rms_norm_static_fp8_quant.default,
                    input=x,
                    weight=weight,
                    scale=scale,
                    epsilon=eps,
                    _result_base_index=0,
                    _residual_base_index=1,
                    _all_bases=[result, residual],
                )
            return (
                fused_add_rms_norm_static_fp8_quant[1],
                scale,
                fused_add_rms_norm_static_fp8_quant[2],
                fused_add_rms_norm_static_fp8_quant[2],
            )

        example_inputs = [
            torch.empty(16, 16).half().cuda(),
            torch.empty(16, 16).half().cuda(),
            torch.empty(16).half().cuda(),
            torch.empty(()).cuda(),
            torch.empty(16, 16).to(dtype=torch.float8_e4m3fn).cuda(),
        ]

        for eps in self.pass_config.rms_norm_eps:
            self.register_replacement_pattern(
                pattern, replacement, example_inputs, scalar_workaround={"eps": eps}
            )

    def build_pass(self):
        self.register_rmsnorm_quant_replacement_pattern()
        self.register_fused_add_rmsnorm_quant_replacement_pattern()
