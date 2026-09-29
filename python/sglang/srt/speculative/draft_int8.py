"""Opt-in int8 weights for the large linears of a DFLASH or DSpark draft.

``SGLANG_SPEC_DRAFT_INT8_WEIGHTS=1`` gives every large half-precision linear
of the draft an int8 copy of its weight (one fp32 scale per output row, no
calibration). Steps of up to 16 rows, which is every decode step of the draft,
read the copy through ``gemm.int8_weight_only_gemv``; anything else, such as the
draft's prefill over a long prompt, keeps the dense weight and the layer's
original method.

Only the draft changes. It proposes tokens and the target still verifies every
one of them, so a coarser draft can move the acceptance length and nothing
else.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Below this a matrix is a few megabytes: not where the draft's step goes.
MIN_PARAMS = 1 << 22
_BLOCK_N, _BLOCK_K = 64, 256


@torch.no_grad()
def quantize_int8_rowwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[N, K]`` dense -> (``[N, K]`` int8, ``[N]`` fp32), symmetric per row."""
    w = weight.float()
    scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


class Int8WeightOnlyLinearMethod:
    """Stands in for a linear layer's quant method: rows the kernel serves read
    the int8 copy, the rest go to the original method."""

    def __init__(self, *, original, weight: torch.Tensor):
        self.original = original
        self.weight_int8, self.scale = quantize_int8_rowwise(weight)
        self.nbytes = (
            self.weight_int8.numel() * self.weight_int8.element_size()
            + self.scale.numel() * self.scale.element_size()
        )
        # Resolved once here, not on every step. Off CUDA (a CPU test) the
        # kernel is never reached, so Triton is not imported.
        self._gemv = self._supported = None
        if self.weight_int8.is_cuda:
            from sglang.kernels.ops.gemm.int8_weight_only_gemv import (
                int8_weight_only_gemv,
                int8_weight_only_gemv_supported,
            )

            self._gemv = int8_weight_only_gemv
            self._supported = int8_weight_only_gemv_supported

    def serves(self, x: torch.Tensor) -> bool:
        return self._supported is not None and self._supported(x=x, w=self.weight_int8)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return self._gemv(x=x, w=self.weight_int8, scale=self.scale)

    def apply(self, layer, x, bias=None):
        if self.serves(x):
            out = self.compute(x)
            return out if bias is None else out + bias
        return self.original.apply(layer, x, bias)

    def __getattr__(self, name):
        return getattr(self.__dict__["original"], name)


def wants_int8_weights(weight, min_params: int = MIN_PARAMS) -> bool:
    if weight is None or weight.ndim != 2:
        return False
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    n, k = weight.shape
    return n * k >= min_params and n % _BLOCK_N == 0 and k % _BLOCK_K == 0


@torch.no_grad()
def apply_draft_int8_weights(
    model, *, min_params: int = MIN_PARAMS, linear_base=None
) -> int:
    """Wrap every large linear of ``model``; returns how many were wrapped."""
    if linear_base is None:
        from sglang.srt.layers.linear import LinearBase as linear_base
    layers = dense = packed = 0
    for module in model.modules():
        if not isinstance(module, linear_base):
            continue
        weight = getattr(module, "weight", None)
        if not wants_int8_weights(weight, min_params):
            continue
        if isinstance(module.quant_method, Int8WeightOnlyLinearMethod):
            continue
        module.quant_method = Int8WeightOnlyLinearMethod(
            original=module.quant_method, weight=weight.data
        )
        layers += 1
        dense += weight.numel() * weight.element_size()
        packed += module.quant_method.nbytes
    logger.info(
        "Draft int8 weights: %d linears, %.2f GB -> %.2f GB read per step",
        layers,
        dense / 1e9,
        packed / 1e9,
    )
    return layers
