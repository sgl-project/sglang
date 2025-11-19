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

# TODO: better handle operator import for registration
import sglang.srt.compilation.fusion.triton_ops.fused_swiglu  # noqa: F401
from sglang.srt.compilation.inductor_pass import SGLangPatternMatcherInductorPass


class FusedActivationPass(SGLangPatternMatcherInductorPass):
    def register_swiglu_replacement_pattern(self) -> None:

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
            return torch.ops.sglang.fused_swiglu.default(x, w)

        example_inputs = [
            torch.empty(16, 16).half().cuda(),  # X
            torch.empty(16, 16).half().cuda().T,  # W.T
            torch.empty(16, 8).half().cuda(),  # out
        ]

        self.register_replacement_pattern(pattern, replacement, example_inputs)

    def register_swiglu_fp8_replacement_pattern(self) -> None:

        def pattern(x, w, x_scale, w_scale, o_scale, out, output_q):
            mm = torch.ops.aten._scaled_mm.default(
                x, w, x_scale, w_scale, None, None, out.dtype
            )
            silu_and_mul = auto_functionalized_v2(
                torch.ops.sgl_kernel.silu_and_mul.default,
                input=mm,
                _out_base_index=0,
                _all_bases=[out],
            )
            sgl_per_tensor_quant_fp8 = auto_functionalized_v2(
                torch.ops.sgl_kernel.sgl_per_tensor_quant_fp8.default,
                input=silu_and_mul[1],
                output_s=o_scale,
                is_static=True,
                _output_q_base_index=0,
                _all_bases=[output_q],
            )
            return sgl_per_tensor_quant_fp8[1]

        def replacement(x, w, x_scale, w_scale, o_scale, out, output_q):
            return torch.ops.sglang.fused_swiglu.default(
                x, w, x_scale, w_scale, o_scale
            )

        example_inputs = [
            torch.empty(16, 16).to(dtype=torch.float8_e4m3fn).cuda(),  # X
            torch.empty(16, 16).to(dtype=torch.float8_e4m3fn).cuda().T,  # W.T
            torch.empty(()).cuda(),  # X_Scale
            torch.empty(()).cuda(),  # W_Scale
            torch.empty(()).cuda(),  # O_Scale
            torch.empty(16, 8).to(dtype=torch.float16).cuda(),  # out
            torch.empty(16, 8).to(dtype=torch.float16).cuda(),  # output_q
        ]

        self.register_replacement_pattern(pattern, replacement, example_inputs)

    def build_pass(self):
        self.register_swiglu_replacement_pattern()
        self.register_swiglu_fp8_replacement_pattern()
