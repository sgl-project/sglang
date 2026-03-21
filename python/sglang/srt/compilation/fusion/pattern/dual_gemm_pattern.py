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

from abc import abstractmethod

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2

from sglang.srt.compilation.fusion.pattern import OpPatternBase, OpPatternRegistery


def _is_cutedsl_dual_gemm_available():
    try:
        from sglang.jit_kernel.cutedsl_dual_gemm import cutedsl_dual_gemm  # noqa: F401

        return True
    except Exception:
        return False


class _DualGemmPattern(OpPatternBase):
    @staticmethod
    @abstractmethod
    def pattern(x, w, out):
        pass


# op: fused_ops/triton_ops/dual_gemm
# TODO: This is most probably broken, fix it for oss
class TritonFusedOpsDualGemmPattern(_DualGemmPattern):
    @staticmethod
    def pattern(x, w, out):
        return torch.ops.sglang.triton_dual_gemm.default(x, w)


# op: cutedsl_dual_gemm (registered via @register_custom_op decorator)
class CuteDSLDualGemmPattern(_DualGemmPattern):
    @staticmethod
    def pattern(x, w, out):
        result = auto_functionalized_v2(
            torch.ops.sglang.cutedsl_dual_gemm.default,
            x=x,
            w=w,
            _out_base_index=0,
            _all_bases=[out],
        )
        return result[1]


class _DualGemmPatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        if _is_cutedsl_dual_gemm_available():
            self.register_op_pattern(CuteDSLDualGemmPattern)
        else:
            self.register_op_pattern(TritonFusedOpsDualGemmPattern)


DualGemmPatternRegistery = _DualGemmPatternRegistery()


class _DualGemmFp8Pattern(OpPatternBase):
    @staticmethod
    @abstractmethod
    def pattern(x, w, x_scale, w_scale, o_scale, output_q):
        pass


# op: fused_ops/triton_ops/dual_gemm
# TODO: This is most probably broken, fix it for oss
class TritonFusedOpsDualGemmFp8Pattern(_DualGemmFp8Pattern):
    @staticmethod
    def pattern(x, w, x_scale, w_scale, o_scale, output_q):
        return torch.ops.sglang.triton_dual_gemm.default(
            x, w, x_scale, w_scale, o_scale
        )


# op: cutedsl_dual_gemm (registered via @register_custom_op decorator)
class CuteDSLDualGemmFp8Pattern(_DualGemmFp8Pattern):
    @staticmethod
    def pattern(x, w, x_scale, w_scale, o_scale, output_q):
        result = auto_functionalized_v2(
            torch.ops.sglang.cutedsl_dual_gemm.default,
            x=x,
            w=w,
            x_scale=x_scale,
            w_scale=w_scale,
            o_scale=o_scale,
            _out_base_index=0,
            _all_bases=[output_q],
        )
        return result[1]


class _DualGemmFp8PatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        if _is_cutedsl_dual_gemm_available():
            self.register_op_pattern(CuteDSLDualGemmFp8Pattern)
        else:
            self.register_op_pattern(TritonFusedOpsDualGemmFp8Pattern)


DualGemmFp8PatternRegistery = _DualGemmFp8PatternRegistery()
