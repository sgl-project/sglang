# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2

from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.compilation.fusion.pattern import OpPattern, pattern_builder
from sglang.srt.compilation.fusion.pattern.quant_fp8_pattern import (
    QuantFp8PatternRegistery,
    StaticQuantFp8Pattern,
)
from sglang.srt.compilation.fusion.pattern.rmsnorm_quant_fp8_pattern import (
    FusedAddRmsnormQuantFp8PatternRegistery,
    RmsnormQuantFp8PatternRegistery,
)
from sglang.srt.compilation.inductor_pass import SGLangPatternMatcherInductorPass


class RMSNormQuantPass(SGLangPatternMatcherInductorPass):
    def register_rmsnorm_quant_replacement_pattern(
        self, quant_fp8_op: OpPattern, rmsnorm_quant_fp8_op: OpPattern
    ):
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
            quant_fp8_op_result = quant_fp8_op.pattern(rmsnorm, output, scale)
            return (
                quant_fp8_op_result[0],
                quant_fp8_op_result[1],
            )

        def replacement(x, rms_result, weight, scale, eps, output):
            rmsnorm_quant_fp8_op_result = rmsnorm_quant_fp8_op.pattern(
                x, weight, scale, eps, output
            )
            if quant_fp8_op.op_type == StaticQuantFp8Pattern:
                repeated_scale = scale.view(1, 1).expand(x.shape[0], 1)
            else:
                repeated_scale = scale
            return rmsnorm_quant_fp8_op_result[1], repeated_scale

        M, N, K = 16, 16, 16
        example_inputs = [
            torch.empty(M, K).half().cuda(),
            torch.empty(N, K).half().cuda(),
            torch.empty(M).half().cuda(),
            torch.empty(()).cuda(),
            torch.empty(M, N).to(dtype=torch.float8_e4m3fn).cuda(),
        ]

        for eps in self.pass_config.rms_norm_eps:
            self.register_replacement_pattern(
                pattern, replacement, example_inputs, scalar_workaround={"eps": eps}
            )

    def register_fused_add_rmsnorm_quant_replacement_pattern(
        self,
        quant_fp8_op: OpPattern,
        fused_add_rmsnorm_quant_fp8_op: OpPattern,
    ):
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

            quant_fp8_op_result = quant_fp8_op.pattern(fused_add_rmsnorm, result, scale)

            return (
                quant_fp8_op_result[0],
                quant_fp8_op_result[1],
                fused_add_rmsnorm[1],
                fused_add_rmsnorm[2],
            )

        def replacement(x, residual, weight, scale, result, eps):
            fused_add_rmsnorm_quant_fp8_op_result = (
                fused_add_rmsnorm_quant_fp8_op.pattern(
                    x, residual, weight, scale, result, eps
                )
            )
            if quant_fp8_op.op_type == StaticQuantFp8Pattern:
                repeated_scale = scale.view(1, 1).expand(x.shape[0], 1)
            else:
                repeated_scale = scale
            return (
                fused_add_rmsnorm_quant_fp8_op_result[1],
                repeated_scale,
                fused_add_rmsnorm_quant_fp8_op_result[2],
                fused_add_rmsnorm_quant_fp8_op_result[2],
            )

        M, N, K = 16, 16, 16
        example_inputs = [
            torch.empty(M, K).half().cuda(),
            torch.empty(N, K).half().cuda(),
            torch.empty(M).half().cuda(),
            torch.empty(()).cuda(),
            torch.empty(M, N).to(dtype=torch.float8_e4m3fn).cuda(),
        ]

        for eps in self.pass_config.rms_norm_eps:
            self.register_replacement_pattern(
                pattern, replacement, example_inputs, scalar_workaround={"eps": eps}
            )

    def build_pass(self):
        pattern_builder(
            self.register_rmsnorm_quant_replacement_pattern,
            [QuantFp8PatternRegistery, RmsnormQuantFp8PatternRegistery],
        )
        pattern_builder(
            self.register_fused_add_rmsnorm_quant_replacement_pattern,
            [QuantFp8PatternRegistery, FusedAddRmsnormQuantFp8PatternRegistery],
        )
