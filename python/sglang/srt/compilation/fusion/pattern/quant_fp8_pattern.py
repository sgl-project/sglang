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


class _QuantFp8Pattern(OpPatternBase):
    @staticmethod
    @abstractmethod
    def pattern(x, output, scale):
        pass


# op: sgl-kernel/per_tensor_quant_fp8
class PerTensorQuantFp8Pattern(_QuantFp8Pattern):
    @staticmethod
    def pattern(x, output, scale):
        per_tensor_quant_fp8 = auto_functionalized_v2(
            torch.ops.sglang.per_tensor_quant_fp8.default,
            input=x[1],
            is_static=True,
            _output_q_base_index=0,
            _output_s_base_index=1,
            _all_bases=[output, scale],
        )

        return (
            per_tensor_quant_fp8[1],
            per_tensor_quant_fp8[2],
        )


# op: fp8_kernels.py/static_quant_fp8_fwd
class StaticQuantFp8Pattern(_QuantFp8Pattern):
    @staticmethod
    def pattern(x, output, scale):
        static_quant_fp8 = auto_functionalized_v2(
            torch.ops.sglang.static_quant_fp8.default,
            x=x[1],
            x_s=scale,
            repeat_scale=True,
            _x_q_base_index=0,
            _all_bases=[output],
        )

        return (
            static_quant_fp8[1],
            static_quant_fp8[0],
        )


class _QuantFp8PatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        self.register_op_pattern(PerTensorQuantFp8Pattern)
        self.register_op_pattern(StaticQuantFp8Pattern)


QuantFp8PatternRegistery = _QuantFp8PatternRegistery()
