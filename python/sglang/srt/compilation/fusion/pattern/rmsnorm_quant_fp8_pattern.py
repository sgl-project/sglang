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

from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.compilation.fusion.pattern import OpPatternBase, OpPatternRegistery
from sglang.srt.utils import is_flashinfer_rmsnorm_quant_kernels_available


class _RmsnormQuantFp8Pattern(OpPatternBase):
    @staticmethod
    @abstractmethod
    def pattern(x, weight, scale, eps, output):
        pass


# op: flashinfer/rmsnorm_quant
class FlashinferRmsnormQuantFp8Pattern(_RmsnormQuantFp8Pattern):
    @staticmethod
    def pattern(x, weight, scale, eps, output):
        return auto_functionalized_v2(
            torch.ops.sglang.flashinfer_rmsnorm_quant.default,
            input=x,
            weight=weight,
            scale=scale,
            eps=eps,
            enable_pdl=is_arch_support_pdl(),
            _out_base_index=0,
            _all_bases=[output],
        )


# op: sgl-kernel/rmsnorm_quant
class SglangRmsnormQuantFp8Pattern(_RmsnormQuantFp8Pattern):
    @staticmethod
    def pattern(x, weight, scale, eps, output):
        return auto_functionalized_v2(
            torch.ops.sgl_kernel.rms_norm_static_fp8_quant.default,
            input=x,
            weight=weight,
            scale=scale,
            epsilon=eps,
            _result_base_index=0,
            _all_bases=[output],
        )


class _RmsnormQuantFp8PatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        if is_flashinfer_rmsnorm_quant_kernels_available():
            self.register_op_pattern(FlashinferRmsnormQuantFp8Pattern)
        else:
            self.register_op_pattern(SglangRmsnormQuantFp8Pattern)


RmsnormQuantFp8PatternRegistery = _RmsnormQuantFp8PatternRegistery()


class _FusedAddRmsnormQuantFp8Pattern(OpPatternBase):
    @staticmethod
    @abstractmethod
    def pattern(x, residual, weight, scale, result, eps):
        pass


# op: flashinfer/fused_add_rmsnorm_quant
class FlashinferFusedAddRmsnormQuantFp8Pattern(_FusedAddRmsnormQuantFp8Pattern):
    @staticmethod
    def pattern(x, residual, weight, scale, result, eps):
        return auto_functionalized_v2(
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


# op: sgl-kernel/fused_add_rmsnorm_quant
class SglangFusedAddRmsnormQuantFp8Pattern(_FusedAddRmsnormQuantFp8Pattern):
    @staticmethod
    def pattern(x, residual, weight, scale, result, eps):
        return auto_functionalized_v2(
            torch.ops.sgl_kernel.fused_add_rms_norm_static_fp8_quant.default,
            input=x,
            weight=weight,
            scale=scale,
            epsilon=eps,
            _result_base_index=0,
            _residual_base_index=1,
            _all_bases=[result, residual],
        )


class _FusedAddRmsnormQuantFp8PatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        if is_flashinfer_rmsnorm_quant_kernels_available():
            self.register_op_pattern(FlashinferFusedAddRmsnormQuantFp8Pattern)
        else:
            self.register_op_pattern(SglangFusedAddRmsnormQuantFp8Pattern)


FusedAddRmsnormQuantFp8PatternRegistery = _FusedAddRmsnormQuantFp8PatternRegistery()
