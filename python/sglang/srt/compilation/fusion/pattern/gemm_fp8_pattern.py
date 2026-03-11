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

from sglang.srt.compilation.fusion.pattern import OpPatternBase, OpPatternRegistery


class _GemmFp8Pattern(OpPatternBase):
    @staticmethod
    @abstractmethod
    def pattern(x, w, x_scale, w_scale, out_dtype):
        pass


# op: aten/_scaled_mm
class TorchScaledMMPattern(_GemmFp8Pattern):
    @staticmethod
    def pattern(x, w, x_scale, w_scale, out_dtype):
        return torch.ops.aten._scaled_mm.default(
            x, w, x_scale, w_scale, None, None, out_dtype
        )


# op: sgl-kernel/fp8_scaled_mm
class CutlassFp8ScaledMMPattern(_GemmFp8Pattern):
    @staticmethod
    def pattern(x, w, x_scale, w_scale, out_dtype):
        return torch.ops.sgl_kernel.fp8_scaled_mm.default(
            x, w, x_scale, w_scale, out_dtype, None
        )


class _GemmFp8PatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        self.register_op_pattern(TorchScaledMMPattern)
        self.register_op_pattern(CutlassFp8ScaledMMPattern)


GemmFp8PatternRegistery = _GemmFp8PatternRegistery()
