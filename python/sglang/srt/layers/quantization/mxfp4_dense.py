"""Opt-in MXFP4 for dense BF16 projections on gfx950.

An MXFP4 checkpoint that quantizes only its routed experts leaves the rest of the
body BF16, and aiter's tuned BF16 dispatch already serves those shapes with the
best BF16 kernel it has -- on Qwen3.5-397B's widest projections it reaches roughly
60% of the device's BF16 peak. Going faster therefore means narrower inputs rather
than a different BF16 kernel, and MXFP4 measures ~2x on those shapes once M is
large enough to amortize quantizing the activation.

The BF16 weight stays live next to the MXFP4 copy, because MXFP4 only wins above
roughly a thousand tokens: below that the GEMM is not compute-bound, the extra
activation quantization dominates, and decode -- where a precision loss would
compound across every step -- keeps running exactly as before.

This is a *model-driven* method: nothing selects it from a checkpoint's quant
config. A model asks for it per projection via :func:`enable_mxfp4_dense`, which
is what keeps the policy (which projections, and above which token count) in the
model that was measured rather than in a global switch.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

logger = logging.getLogger(__name__)


class Mxfp4DenseLinearMethod(UnquantizedLinearMethod):
    """BF16 linear that switches to MXFP4 once the token count makes it pay.

    Weight creation and loading are the base method's, so the checkpoint still
    loads as BF16 and every consumer that reads ``layer.weight`` keeps working.
    ``apply_into`` is deliberately not overridden: it writes into caller-owned
    storage, which the MXFP4 GEMM cannot do without an extra copy that would eat
    the win.
    """

    def __init__(self, min_tokens: int):
        super().__init__()
        self.min_tokens = min_tokens
        self._w4: Optional[torch.Tensor] = None
        self._w4_scale: Optional[torch.Tensor] = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if self._w4 is not None:
            return
        from sglang.kernels.ops.gemm import mxfp4_dense_aiter_hip as mxfp4

        weight = getattr(layer, "weight", None)
        if weight is None or not mxfp4.packable(weight.data):
            # Leaving the MXFP4 buffers unset keeps this layer on BF16 forever.
            return
        self._w4, self._w4_scale = mxfp4.pack(weight.data)

    def _use_mxfp4(self, x: torch.Tensor) -> bool:
        return (
            self._w4 is not None
            # Fused all-reduce hands some projections a (bf16, fp8, scale) tuple.
            and isinstance(x, torch.Tensor)
            and x.dim() == 2
            and x.dtype == torch.bfloat16
            and x.shape[0] >= self.min_tokens
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._use_mxfp4(x):
            return super().apply(layer, x, bias)
        from sglang.kernels.ops.gemm import mxfp4_dense_aiter_hip as mxfp4

        out = mxfp4.run(x, self._w4, self._w4_scale)
        return out if bias is None else out.add_(bias)


def enable_mxfp4_dense(layer: torch.nn.Module, min_tokens: int) -> bool:
    """Put ``layer`` on the MXFP4 path above ``min_tokens`` tokens; report if it took.

    Must run before weights are loaded, so the replacement method is the one the
    loader calls ``process_weights_after_loading`` on. Only an unquantized layer
    can be converted -- anything else already has a quantized kernel that this
    would be undoing.
    """
    from sglang.kernels.ops.gemm import mxfp4_dense_aiter_hip as mxfp4

    if not mxfp4.supported():
        return False
    method = getattr(layer, "quant_method", None)
    if type(method) is not UnquantizedLinearMethod:
        return False
    layer.quant_method = Mxfp4DenseLinearMethod(min_tokens)
    return True


def mxfp4_dense_extra_bytes(layer: torch.nn.Module) -> int:
    """MXFP4 bytes ``layer`` holds on top of its BF16 weight, for load-time reporting."""
    method = getattr(layer, "quant_method", None)
    if not isinstance(method, Mxfp4DenseLinearMethod) or method._w4 is None:
        return 0
    return method._w4.nbytes + method._w4_scale.nbytes
