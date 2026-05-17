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


def _is_jit_rmsnorm_quant_available():
    try:
        from sglang.jit_kernel.norm import fused_add_rmsnorm_quant  # noqa: F401
        from sglang.jit_kernel.norm import rmsnorm_quant  # noqa: F401

        return True
    except Exception:
        return False


class _RmsnormQuantFp8Pattern(OpPatternBase):
    @staticmethod
    @abstractmethod
    def pattern(x, weight, scale, eps, output):
        pass


# op: jit/rmsnorm_quant
class JitRmsnormQuantFp8Pattern(_RmsnormQuantFp8Pattern):
    @staticmethod
    def pattern(x, weight, scale, eps, output):
        return auto_functionalized_v2(
            torch.ops.sglang.jit_rmsnorm_quant.default,
            input=x,
            weight=weight,
            scale=scale,
            eps=eps,
            _out_base_index=0,
            _all_bases=[output],
        )


class _RmsnormQuantFp8PatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        if _is_jit_rmsnorm_quant_available():
            self.register_op_pattern(JitRmsnormQuantFp8Pattern)
        else:
            raise RuntimeError("sglang.jit_kernel.norm is not available")


RmsnormQuantFp8PatternRegistery = _RmsnormQuantFp8PatternRegistery()


class _FusedAddRmsnormQuantFp8Pattern(OpPatternBase):
    @staticmethod
    @abstractmethod
    def pattern(x, residual, weight, scale, result, eps):
        pass


# op: jit/fused_add_rmsnorm_quant
class JitFusedAddRmsnormQuantFp8Pattern(_FusedAddRmsnormQuantFp8Pattern):
    @staticmethod
    def pattern(x, residual, weight, scale, result, eps):
        return auto_functionalized_v2(
            torch.ops.sglang.jit_fused_add_rmsnorm_quant.default,
            input=x,
            residual=residual,
            weight=weight,
            scale=scale,
            eps=eps,
            _out_base_index=0,
            _residual_base_index=1,
            _all_bases=[result, residual],
        )


class _FusedAddRmsnormQuantFp8PatternRegistery(OpPatternRegistery):
    def build_op_pattern_registery(self):
        if _is_jit_rmsnorm_quant_available():
            self.register_op_pattern(JitFusedAddRmsnormQuantFp8Pattern)
        else:
            raise RuntimeError("sglang.jit_kernel.norm is not available")


FusedAddRmsnormQuantFp8PatternRegistery = _FusedAddRmsnormQuantFp8PatternRegistery()
