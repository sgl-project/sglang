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


def _is_jit_rmsnorm_per_tensor_quant_available():
    try:
        from sglang.jit_kernel.norm import rmsnorm_per_tensor_quant  # noqa: F401
        from sglang.jit_kernel.norm import (  # noqa: F401
            fused_add_rmsnorm_per_tensor_quant,
        )

        return True
    except Exception:
        return False


class _RmsnormPerTensorQuantFp8Pattern(OpPatternBase):
    @staticmethod
    @abstractmethod
    def pattern(x, weight, scale, eps, output):
        pass


# op: jit/rmsnorm_per_tensor_quant
class JitRmsnormPerTensorQuantFp8Pattern(_RmsnormPerTensorQuantFp8Pattern):
    @staticmethod
    def pattern(x, weight, scale, eps, output):
        return auto_functionalized_v2(
            torch.ops.sglang.jit_rmsnorm_per_tensor_quant.default,
            input=x,
            weight=weight,
            scale=scale,
            eps=eps,
            _out_base_index=0,
            _all_bases=[output],
        )


# op: flashinfer/rmsnorm_quant
class FlashinferRmsnormQuantFp8Pattern(_RmsnormPerTensorQuantFp8Pattern):
    @staticmethod
    def pattern(x, weight, scale, eps, output):
        return auto_functionalized_v2(
            torch.ops.sglang.flashinfer_rmsnorm_per_tensor_quant.default,
            input=x,
            weight=weight,
            scale=scale,
            eps=eps,
            enable_pdl=is_arch_support_pdl(),
            _out_base_index=0,
            _all_bases=[output],
        )


class _RmsnormPerTensorQuantFp8PatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        self.register_op_pattern(JitRmsnormPerTensorQuantFp8Pattern)

        # if _is_jit_rmsnorm_per_tensor_quant_available():
        #     self.register_op_pattern(JitRmsnormPerTensorQuantFp8Pattern)
        # else:
        #     raise RuntimeError("sglang.jit_kernel.norm is not available")


RmsnormPerTensorQuantFp8PatternRegistery = _RmsnormPerTensorQuantFp8PatternRegistery()


class _FusedAddRmsnormPerTensorQuantFp8Pattern(OpPatternBase):
    @staticmethod
    @abstractmethod
    def pattern(x, residual, weight, scale, result, eps):
        pass


# op: jit/fused_add_rmsnorm_per_tensor_quant
class JitFusedAddRmsnormPerTensorQuantFp8Pattern(
    _FusedAddRmsnormPerTensorQuantFp8Pattern
):
    @staticmethod
    def pattern(x, residual, weight, scale, result, eps):
        return auto_functionalized_v2(
            torch.ops.sglang.jit_fused_add_rmsnorm_per_tensor_quant.default,
            input=x,
            residual=residual,
            weight=weight,
            scale=scale,
            eps=eps,
            _out_base_index=0,
            _residual_base_index=1,
            _all_bases=[result, residual],
        )


# op: flashinfer/fused_add_rmsnorm_quant
class FlashinferFusedAddRmsnormQuantFp8Pattern(
    _FusedAddRmsnormPerTensorQuantFp8Pattern
):
    @staticmethod
    def pattern(x, residual, weight, scale, result, eps):
        return auto_functionalized_v2(
            torch.ops.sglang.flashinfer_fused_add_rmsnorm_per_tensor_quant.default,
            input=x,
            weight=weight,
            scale=scale,
            eps=eps,
            enable_pdl=is_arch_support_pdl(),
            _out_base_index=0,
            _residual_base_index=1,
            _all_bases=[result, residual],
        )


class _FusedAddRmsnormPerTensorQuantFp8PatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        self.register_op_pattern(FlashinferFusedAddRmsnormQuantFp8Pattern)

        # if _is_jit_rmsnorm_per_tensor_quant_available():
        #     self.register_op_pattern(JitFusedAddRmsnormPerTensorQuantFp8Pattern)
        # else:
        #     raise RuntimeError("sglang.jit_kernel.norm is not available")


FusedAddRmsnormPerTensorQuantFp8PatternRegistery = (
    _FusedAddRmsnormPerTensorQuantFp8PatternRegistery()
)
