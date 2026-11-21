# SPDX-License-Identifier: Apache-2.0
"""Opt-in MXFP8 copy of a BF16 target lm_head: ``--lm-head-dtype mxfp8``.

At decode the lm_head is a ``[rows, hidden] x [hidden, vocab]`` GEMM with few
rows and a large vocabulary, so it streams the weight. ``Mxfp8LmHead`` keeps an
MXFP8 copy of the weight (E4M3 values, one E8M0 scale per 32 elements along K,
scales in the cuBLASLt 128x4 swizzled layout), quantizes the hidden rows the
same way on every call and runs cuBLASLt's block-scaled GEMM, which returns
BF16 logits like the BF16 head. The BF16 weight stays in place, so anything
else that projects through it, a speculative draft for example, is unchanged.

A model opts in from its target ``post_load_weights`` through
``LogitsProcessor.prepare_lm_head_dtype``; ``LM_HEAD_DTYPE_SUPPORTED_ARCHS``
lists the architectures that do.
"""

from __future__ import annotations

from typing import Optional

import torch

LM_HEAD_DTYPE_SUPPORTED_ARCHS = frozenset({"DeepseekV4ForCausalLM"})

# Weight rows quantized per step of a (re)build. A multiple of 128, so every
# step writes whole 128-row tiles of the swizzled scales.
_REBUILD_ROWS = 16384

# A single logits row keeps the BF16 GEMM: on SM107 with the DeepSeek-V4 head
# (129280 x 5120), MXFP8 is 5-7% slower than BF16 at one row and faster from two.
MIN_ROWS = 2


def _swizzled_scale_bytes(rows: int, k: int) -> int:
    """Bytes of the 128x4-swizzled E8M0 scales of a ``[rows, k]`` MXFP8 tensor."""
    return -(-rows // 128) * 128 * (-(-(k // 32) // 4) * 4)


def mxfp8_lm_head_unsupported_reason(weight: Optional[torch.Tensor]) -> Optional[str]:
    """Why ``weight`` cannot get an MXFP8 copy, or None when it can."""
    if not isinstance(weight, torch.Tensor):
        return "the lm_head has no weight tensor"
    if weight.dtype != torch.bfloat16:
        return f"the lm_head weight is {weight.dtype}, not torch.bfloat16"
    if weight.dim() != 2:
        return f"the lm_head weight has {weight.dim()} dims, not 2"
    rows, k = weight.shape
    if k % 32 != 0:
        return f"the hidden size {k} is not a multiple of 32"
    if rows % 16 != 0:
        return f"the vocabulary shard of {rows} rows is not a multiple of 16"
    if not weight.is_cuda:
        return "the lm_head weight is not on a CUDA device"
    return None


class Mxfp8LmHead:
    """MXFP8 copy of a BF16 lm_head weight and the GEMM that reads it."""

    def __init__(self, weight: torch.Tensor):
        reason = mxfp8_lm_head_unsupported_reason(weight)
        if reason is not None:
            raise ValueError(f"No MXFP8 lm_head: {reason}.")
        rows, k = weight.shape
        self.weight = weight
        self.qweight = torch.empty(
            (rows, k), dtype=torch.float8_e4m3fn, device=weight.device
        )
        self.weight_scale = torch.empty(
            _swizzled_scale_bytes(rows, k), dtype=torch.uint8, device=weight.device
        )
        self.rebuild()

    @property
    def nbytes(self) -> int:
        return self.qweight.nbytes + self.weight_scale.nbytes

    @torch.no_grad()
    def rebuild(self) -> None:
        """Quantize ``self.weight`` into the copy's buffers in place, so CUDA
        graphs captured before a weight reload keep reading valid memory."""
        from sglang.kernels.ops.quantization.mxfp8_swizzled_triton import (
            mxfp8_quantize_swizzled,
        )

        rows, k = self.weight.shape
        tile_bytes = _swizzled_scale_bytes(128, k)
        weight = self.weight.detach()
        for start in range(0, rows, _REBUILD_ROWS):
            chunk = weight[start : start + _REBUILD_ROWS]
            qweight, scale = mxfp8_quantize_swizzled(chunk)
            self.qweight[start : start + chunk.shape[0]].copy_(qweight)
            offset = start // 128 * tile_bytes
            self.weight_scale[offset : offset + scale.numel()].copy_(scale)

    def applies(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
        return (
            weight is self.weight
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_cuda
            and hidden_states.dim() == 2
            and hidden_states.shape[0] >= MIN_ROWS
            and not torch.compiler.is_compiling()
        )

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

        from sglang.kernels.ops.quantization.mxfp8_swizzled_triton import (
            mxfp8_quantize_swizzled,
        )

        qinput, input_scale = mxfp8_quantize_swizzled(hidden_states.contiguous())
        return scaled_mm(
            qinput,
            self.qweight.t(),
            scale_a=input_scale.view(torch.float8_e8m0fnu),
            scale_recipe_a=ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            scale_b=self.weight_scale.view(torch.float8_e8m0fnu),
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            output_dtype=torch.bfloat16,
        )


__all__ = [
    "LM_HEAD_DTYPE_SUPPORTED_ARCHS",
    "MIN_ROWS",
    "Mxfp8LmHead",
    "mxfp8_lm_head_unsupported_reason",
]
