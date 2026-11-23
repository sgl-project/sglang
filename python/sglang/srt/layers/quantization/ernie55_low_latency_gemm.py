# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
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
from __future__ import annotations

import torch

from sglang.srt.layers.quantization.halops_mxfp8 import HALOPS_MODE
from sglang.srt.layers.quantization.mxfp8_block_convert import MXFP8_BLOCK_SIZE
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


class Ernie5_5LowLatencyGemm:
    MAX_TOKENS = 16
    PLAN_ATTR = "_low_latency_gemm_plan"

    def __init__(self, weight: torch.Tensor, sfb: torch.Tensor):
        self.weight = weight
        self.sfb = sfb

    @classmethod
    def register_hooks(cls) -> None:
        fp8 = "sglang.srt.layers.quantization.fp8.Fp8LinearMethod."
        HookRegistry.register(
            fp8 + "process_weights_after_loading_block_quant",
            cls._process_weights,
            HookType.AROUND,
        )
        HookRegistry.register(fp8 + "apply", cls._apply, HookType.AROUND)
        HookRegistry.apply_hooks()

    @staticmethod
    def block_scale_group32(
        scale: torch.Tensor, n: int, k: int, block_size: list
    ) -> torch.Tensor:
        """UE8M0 exponents of an FP8 weight scale, one per output row and 32 columns.

        ``scale`` is either fp32 with one power-of-two value per ``block_size``
        tile, or per-row UE8M0 bytes (int32-packed or uint8) per ``block_k``
        columns. Leading group dimensions are kept.
        """
        block_n, block_k = block_size
        if scale.dtype == torch.float32:
            bits = scale.contiguous().view(torch.int32)
            if bool((bits & 0x807FFFFF).any()):
                raise ValueError("fp32 FP8 block scales must be powers of two")
            rows = ((bits >> 23) & 0xFF).to(torch.uint8)
            rows = rows.repeat_interleave(block_n, dim=-2)
        else:
            packed = scale.contiguous()
            rows = packed.reshape(-1).view(torch.uint8).view(*packed.shape[:-1], -1)
        rows = rows[..., :n, : -(-k // block_k)]
        rows = rows.repeat_interleave(block_k // MXFP8_BLOCK_SIZE, dim=-1)
        return rows[..., : k // MXFP8_BLOCK_SIZE].contiguous()

    @classmethod
    def split_rows(cls, layer: torch.nn.Module, rows: int) -> tuple:
        plan = getattr(layer, cls.PLAN_ATTR)
        return tuple(
            cls(plan.weight[s], plan.sfb[s]) for s in (slice(rows), slice(rows, None))
        )

    def run(self, x: tuple, sm_count: int = -1) -> torch.Tensor:
        from halops import low_latency_mma_gemm

        m = x[0].numel() // x[0].shape[-1]
        q, sfa = x[0].reshape(m, -1), x[1].reshape(m, -1)
        out = q.new_empty(m, self.weight.shape[0], dtype=torch.bfloat16)
        low_latency_mma_gemm(q, self.weight, sfa, self.sfb, out, sm_count=sm_count)
        return out

    @classmethod
    def _process_weights(cls, original_fn, method, layer):
        out = original_fn(method, layer)
        weight, scale = layer.weight.data, layer.weight_scale_inv.data
        if scale.dim() == 2:
            sfb = cls.block_scale_group32(
                scale, *weight.shape, method.weight_block_size
            )
            setattr(layer, cls.PLAN_ATTR, cls(weight, sfb))
        return out

    @classmethod
    def _apply(cls, original_fn, method, layer, x, bias=None):
        plan = getattr(layer, cls.PLAN_ATTR, None)
        if plan is None:
            return original_fn(method, layer, x, bias)
        a = x[0] if isinstance(x, tuple) else x
        m = a.numel() // a.shape[-1]
        if not isinstance(x, tuple):
            if not (get_is_capture_mode() and 1 <= m <= cls.MAX_TOKENS):
                return original_fn(method, layer, x, bias)
            from halops import group_wise_quant

            x = group_wise_quant(
                x.reshape(m, -1).contiguous(), mode=HALOPS_MODE, use_swizzle=False
            )
        return plan.run(x).view(*a.shape[:-1], -1)
