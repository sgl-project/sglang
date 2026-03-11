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
        return torch.ops.sglang.dual_gemm.default(x, w)


# op: sgl-kernel/dual_mm
class CutlassDualGemmPattern(_DualGemmPattern):
    @staticmethod
    def pattern(x, w, out):
        w_gate, w_up = torch.split(w, w.shape[1] // 2, dim=1)
        dual_mm = auto_functionalized_v2(
            torch.sgl_kernel.dual_mm.out,
            a=x,
            b0=w_gate,
            b1=w_up,
            post_grad_name="",
            epilogue_name="silu",
            out_shape=(x.shape[0], w.shape[1] // 2),
            out_stride=out.stride(),
            out_dtype=out.dtype,
            a_scale=None,
            b_scale=None,
            o_scale=None,
            _out_base_index=0,
            _all_bases=[out],
        )
        return dual_mm[1]


class _DualGemmPatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        if False:
            self.register_op_pattern(CutlassDualGemmPattern)
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
        return torch.ops.sglang.dual_gemm.default(x, w, x_scale, w_scale, o_scale)


# op: sgl-kernel/dual_mm
class CutlassDualGemmFp8Pattern(_DualGemmFp8Pattern):
    @staticmethod
    def pattern(x, w, x_scale, w_scale, o_scale, output_q):
        w_gate, w_up = torch.split(w, w.shape[1] // 2, dim=1)
        dual_mm = auto_functionalized_v2(
            torch.ops.sgl_kernel.dual_mm.out,
            a=x,
            b0=w_gate,
            b1=w_up,
            post_grad_name="",
            epilogue_name="silu",
            out_shape=(x.shape[0], w.shape[1] // 2),
            out_stride=output_q.stride(),
            out_dtype=output_q.dtype,
            a_scale=x_scale,
            b_scale=w_scale,
            o_scale=1 / o_scale,
            _out_base_index=0,
            _all_bases=[output_q],
        )
        return dual_mm[1]


class _DualGemmFp8PatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        if False:
            self.register_op_pattern(CutlassDualGemmFp8Pattern)
        else:
            self.register_op_pattern(TritonFusedOpsDualGemmFp8Pattern)


DualGemmFp8PatternRegistery = _DualGemmFp8PatternRegistery()
