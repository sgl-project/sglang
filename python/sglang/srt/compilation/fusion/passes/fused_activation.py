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

from sglang.srt.compilation.fusion.pattern import OpPattern, pattern_builder
from sglang.srt.compilation.fusion.pattern.dual_gemm_pattern import (
    DualGemmFp8PatternRegistery,
    DualGemmPatternRegistery,
)
from sglang.srt.compilation.fusion.pattern.gemm_fp8_pattern import (
    CutlassFp8ScaledMMPattern,
    GemmFp8PatternRegistery,
    TorchScaledMMPattern,
)
from sglang.srt.compilation.fusion.pattern.quant_fp8_pattern import (
    PerTensorQuantFp8Pattern,
    QuantFp8PatternRegistery,
    StaticQuantFp8Pattern,
)
from sglang.srt.compilation.inductor_pass import SGLangPatternMatcherInductorPass


class FusedActivationPass(SGLangPatternMatcherInductorPass):
    def register_dual_gemm_replacement_pattern(self, dual_gemm_op: OpPattern) -> None:
        def pattern(x, w, out):
            mm = torch.ops.aten.mm.default(x, w)
            silu_and_mul = auto_functionalized_v2(
                torch.ops.sgl_kernel.silu_and_mul.default,
                input=mm,
                _out_base_index=0,
                _all_bases=[out],
            )
            return silu_and_mul[1]

        def replacement(x, w, out):
            dual_gemm_fp8_op_result = dual_gemm_op.pattern(x, w, out)
            return dual_gemm_fp8_op_result

        M, K, N = 16, 16, 16
        example_inputs = [
            torch.empty(M, K).half().cuda(),  # X
            torch.empty(K, N).half().cuda().T,  # W.T
            torch.empty(M, N // 2).half().cuda(),  # out
        ]

        self.register_replacement_pattern(pattern, replacement, example_inputs)

    def register_dual_gemm_fp8_replacement_pattern(
        self,
        quant_fp8_op: OpPattern,
        gemm_fp8_op: OpPattern,
        dual_gemm_fp8_op: OpPattern,
    ) -> None:
        def pattern(x, w, x_scale, w_scale, o_scale, out, output_q):
            gemm_fp8_op_result = gemm_fp8_op.pattern(x, w, x_scale, w_scale, out.dtype)
            silu_and_mul = auto_functionalized_v2(
                torch.ops.sgl_kernel.silu_and_mul.default,
                input=gemm_fp8_op_result,
                _out_base_index=0,
                _all_bases=[out],
            )
            quant_fp8_op_result = quant_fp8_op.pattern(silu_and_mul, output_q, o_scale)
            return (
                quant_fp8_op_result[0],
                quant_fp8_op_result[1],
            )

        def replacement(x, w, x_scale, w_scale, o_scale, out, output_q):
            dual_gemm_fp8_op_result = dual_gemm_fp8_op.pattern(
                x, w, x_scale, w_scale, o_scale, output_q
            )
            if quant_fp8_op.op_type == StaticQuantFp8Pattern:
                repeated_o_scale = o_scale.view(1, 1).expand(x.shape[0], 1)
            else:
                repeated_o_scale = o_scale
            return dual_gemm_fp8_op_result, repeated_o_scale

        M, K, N = 16, 16, 16
        if quant_fp8_op.op_type == StaticQuantFp8Pattern:
            SM, SN = M, N
        else:
            SM, SN = 1, 1

        example_inputs = [
            torch.empty(M, K, device="cuda", dtype=torch.float8_e4m3fn),  # X
            torch.empty(K, N, device="cuda", dtype=torch.float8_e4m3fn).T,  # W.T
            torch.empty(SM, 1, device="cuda", dtype=torch.float32),  # X_Scale [M, 1]
            torch.empty(SN, 1, device="cuda", dtype=torch.float32),  # W_Scale [N, 1]
            torch.empty(
                1, device="cuda", dtype=torch.float32
            ),  # O_Scale (or (1,1) if needed)
            torch.empty(M, N // 2, device="cuda", dtype=torch.float16),  # out
            torch.empty(
                M, N // 2, device="cuda", dtype=torch.float8_e4m3fn
            ),  # output_q
        ]

        self.register_replacement_pattern(pattern, replacement, example_inputs)

    def build_pass(self):
        pattern_builder(
            self.register_dual_gemm_replacement_pattern,
            [DualGemmPatternRegistery],
        )

        pattern_builder(
            self.register_dual_gemm_fp8_replacement_pattern,
            [
                QuantFp8PatternRegistery,
                GemmFp8PatternRegistery,
                DualGemmFp8PatternRegistery,
            ],
            ignore_combinations=[
                (StaticQuantFp8Pattern, TorchScaledMMPattern),
                (PerTensorQuantFp8Pattern, CutlassFp8ScaledMMPattern),
            ],
        )
