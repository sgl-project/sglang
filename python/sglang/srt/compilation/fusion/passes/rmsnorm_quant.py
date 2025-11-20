import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2

from sglang.srt.compilation.inductor_pass import SGLangPatternMatcherInductorPass


class RMSNormQuantPass(SGLangPatternMatcherInductorPass):
    def register_rmsnorm_quant_replacement_pattern(self):
        def pattern(x, rms_result, weight, scale, eps, output):
            rmsnorm = auto_functionalized_v2(
                torch.ops.sgl_kernel.rmsnorm.default,
                input=x,
                weight=weight,
                eps=eps,
                enable_pdl=False,
                _output_base_index=0,
                _all_bases=[rms_result],
            )

            sgl_per_tensor_quant_fp8 = auto_functionalized_v2(
                torch.ops.sgl_kernel.sgl_per_tensor_quant_fp8.default,
                input=rmsnorm[1],
                output_s=scale,
                is_static=True,
                _output_q_base_index=0,
                _all_bases=[output],
            )
            return sgl_per_tensor_quant_fp8[1]

        def replacement(x, rms_result, weight, scale, eps, output):
            rms_norm_static_fp8_quant = auto_functionalized_v2(
                torch.ops.sgl_kernel.rms_norm_static_fp8_quant.default,
                input=x,
                weight=weight,
                scale=scale,
                epsilon=eps,
                _result_base_index=0,
                _all_bases=[output],
            )
            return rms_norm_static_fp8_quant[1]

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
                enable_pdl=False,
                _input_base_index=0,
                _residual_base_index=1,
                _all_bases=[x, residual],
            )

            sgl_per_tensor_quant_fp8 = auto_functionalized_v2(
                torch.ops.sgl_kernel.sgl_per_tensor_quant_fp8.default,
                input=fused_add_rmsnorm[1],
                output_s=scale,
                is_static=True,
                _output_q_base_index=0,
                _all_bases=[result],
            )

            return (
                sgl_per_tensor_quant_fp8[1],
                fused_add_rmsnorm[1],
                fused_add_rmsnorm[2],
            )

        def replacement(x, residual, weight, scale, result, eps):
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
