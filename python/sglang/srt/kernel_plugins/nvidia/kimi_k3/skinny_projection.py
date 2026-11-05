# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Prepared B1/B2 projection using SGLang's existing SIMT tiny GEMM.

The related vLLM skinny GEMM uses a different grid and reduction schedule.  This
adapter reuses the product SGLang kernel; it makes no numerical-equivalence or
performance claim about that vLLM implementation.  The SGLang kernel accumulates
BF16 inputs in FP32 and converts the result to the caller's BF16 output.
"""

from __future__ import annotations

from typing import Any

import torch

# These are the only SGLang internals read by this adapter.  In particular, the
# import and JIT warmup happen during prepare(), never on the serving path.
SGLANG_READS = {
    "kernels/ops/gemm/tiny_gemm.py": ("can_use_tiny_gemm", "tiny_gemm_bf16"),
}

_MAX_M = 2
# tiny_gemm.cuh's N variant uses kMaxVecBytes, which is 32 on Blackwell and
# 16 on older architectures.  Requiring 32 here safely admits either target.
# Its K variant uses 16-byte vectors and also satisfies this stricter contract.
_ALIGNMENT = 32
_ROW_ELEMENTS = _ALIGNMENT // 2  # BF16 elements per Blackwell vector.


def _load_backend() -> tuple[Any, Any]:
    from sglang.kernels.ops.gemm.tiny_gemm import (
        can_use_tiny_gemm,
        tiny_gemm_bf16,
    )

    return can_use_tiny_gemm, tiny_gemm_bf16


class SkinnyProjection:
    """Wrap an already prepared projection, retaining its unsupported path.

    Call ``prepare()`` after final weight placement and before CUDA capture.
    ``run`` only uses tiny GEMM for warmed batches and compatible tensor views;
    every other call delegates to the existing fallback without a JIT attempt.
    """

    def __init__(
        self, weight: torch.Tensor, fallback: Any, batches: tuple[int, ...] = (1, 2)
    ) -> None:
        if any(type(batch) is not int or batch not in (1, 2) for batch in batches):
            raise ValueError("skinny projection batches must be B1 and/or B2")
        self.weight = weight
        self.fallback = fallback
        self.batches = tuple(dict.fromkeys(batches))
        self._prepared = False
        self._tiny: Any = None
        self._shape = tuple(weight.shape)
        self._device = weight.device

    def _weight_admitted(self) -> bool:
        weight = self.weight
        return bool(
            len(self._shape) == 2
            and all(size > 0 for size in self._shape)
            and tuple(weight.shape) == self._shape
            and weight.dtype == torch.bfloat16
            and weight.device.type == "cuda"
            and weight.device == self._device
            and weight.is_contiguous()
            and weight.data_ptr() % _ALIGNMENT == 0
        )

    def prepare(self) -> SkinnyProjection:
        if self._prepared:
            return self
        if not self.batches or not self._weight_admitted():
            self._prepared = True
            return self
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare skinny projection before CUDA capture")
            can_use, tiny = _load_backend()
            n, k = self._shape
            if can_use(n, k, max_m=_MAX_M):
                for batch in self.batches:
                    x = self.weight.new_zeros((batch, k))
                    out = self.weight.new_empty((batch, n))
                    tiny(x, self.weight, out=out, max_m=_MAX_M)
                self._tiny = tiny
        self._prepared = True
        return self

    def _admitted(self, x: torch.Tensor, out: torch.Tensor) -> bool:
        if self._tiny is None or not self._weight_admitted():
            return False
        n, k = self._shape
        return bool(
            x.ndim == 2
            and x.shape[0] in self.batches
            and x.shape[1] == k
            and x.dtype == torch.bfloat16
            and x.device == self._device
            and x.stride(1) == 1
            and x.stride(0) >= 0
            and x.stride(0) % _ROW_ELEMENTS == 0
            and x.data_ptr() % _ALIGNMENT == 0
            and tuple(out.shape) == (x.shape[0], n)
            and out.dtype == torch.bfloat16
            and out.device == self._device
            and out.is_contiguous()
        )

    def run(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self._admitted(x, out):
            self._tiny(x, self.weight, out=out, max_m=_MAX_M)
            return out
        return self.fallback.run(x, out)
