"""Opt-in microscaling (MXFP6/MXFP4) for dense BF16 projections on gfx950.

An MX checkpoint that quantizes only its routed experts leaves the rest of the
body BF16, and aiter's tuned BF16 dispatch already serves those shapes with the
best BF16 kernel it has -- on Qwen3.5-397B's widest projections it reaches roughly
60% of the device's BF16 peak. Going faster therefore means narrower inputs rather
than a different BF16 kernel.

Which narrower format to use is an accuracy question, and on this model it is not
close. Measured on the four widest projections at prefill token counts:

    format   speedup over tuned BF16   relative error   gsm8k
    MXFP4    ~2.6x                     ~16%             -8.5 points, 8.9% invalid
    MXFP6    ~1.9x                     ~4%              (see tests/campaign)
    FP8      ~1.4x                     ~3.7%            n/a -- too slow to matter

MXFP6 is the default because it reaches FP8's error at appreciably more speed
than FP8 can manage here. CDNA4 is what makes that possible: the matrix core
runs MXFP6 at the MXFP4 rate (10.1 PFLOPS each) while MXFP8 gets half (5), so
the two extra mantissa bits over MXFP4 cost nothing in peak terms. MXFP4 stays
selectable for measurement, but it is not a sensible production choice on this
model.

The BF16 weight stays live next to the packed copy, because these formats only
win above roughly a thousand tokens: below that the GEMM is not compute-bound,
the extra activation quantization dominates, and decode -- where a precision
loss would compound across every step -- keeps running exactly as before.

This is a *model-driven* method: nothing selects it from a checkpoint's quant
config. A model asks for it per projection via :func:`enable_mx_dense`, which is
what keeps the policy (which projections, which format, and above which token
count) in the model that was measured rather than in a global switch.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

logger = logging.getLogger(__name__)

# Formats this method can install, cheapest-error-first. Both adapters expose the
# same supported/packable/pack/run surface so the method body is format-agnostic.
MX_DENSE_FORMATS = ("mxfp6", "mxfp4")
MX_DENSE_DEFAULT_FORMAT = "mxfp6"

# One line per process, so a server log shows whether the path is live, in which
# format, and the threshold it uses. Which projections were chosen is the
# caller's to report.
_announced = False


def _adapter(fmt: str):
    if fmt == "mxfp6":
        from sglang.kernels.ops.gemm import mxfp6_dense_aiter_hip as mod
    elif fmt == "mxfp4":
        from sglang.kernels.ops.gemm import mxfp4_dense_aiter_hip as mod
    else:
        raise ValueError(
            f"unknown dense MX format {fmt!r}; pick from {list(MX_DENSE_FORMATS)}"
        )
    return mod


class MxDenseLinearMethod(UnquantizedLinearMethod):
    """BF16 linear that switches to a microscaling format once tokens make it pay.

    Weight creation and loading are the base method's, so the checkpoint still
    loads as BF16 and every consumer that reads ``layer.weight`` keeps working.
    ``apply_into`` is deliberately not overridden: it writes into caller-owned
    storage, which these GEMMs cannot do without an extra copy that would eat
    the win.
    """

    def __init__(self, fmt: str, min_tokens: int):
        super().__init__()
        self.fmt = fmt
        self.min_tokens = min_tokens
        self._mx = _adapter(fmt)
        self._packed: Optional[torch.Tensor] = None
        self._scale: Optional[torch.Tensor] = None
        self._out_features: Optional[int] = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if self._packed is not None:
            return
        weight = getattr(layer, "weight", None)
        if weight is None or not self._mx.packable(weight.data):
            # Leaving the packed buffers unset keeps this layer on BF16 forever.
            return
        self._out_features = weight.shape[0]
        self._packed, self._scale = self._mx.pack(weight.data)

    def _use_mx(self, x: torch.Tensor) -> bool:
        return (
            self._packed is not None
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
        if not self._use_mx(x):
            return super().apply(layer, x, bias)
        out = self._mx.run(x, self._packed, self._scale, self._out_features)
        return out if bias is None else out.add_(bias)


def enable_mx_dense(
    layer: torch.nn.Module,
    min_tokens: int,
    fmt: str = MX_DENSE_DEFAULT_FORMAT,
) -> bool:
    """Put ``layer`` on the MX path above ``min_tokens`` tokens; report if it took.

    Must run before weights are loaded, so the replacement method is the one the
    loader calls ``process_weights_after_loading`` on. Only an unquantized layer
    can be converted -- anything else already has a quantized kernel that this
    would be undoing.
    """
    if not _adapter(fmt).supported():
        return False
    method = getattr(layer, "quant_method", None)
    if type(method) is not UnquantizedLinearMethod:
        return False
    layer.quant_method = MxDenseLinearMethod(fmt, min_tokens)
    global _announced
    if not _announced:
        _announced = True
        logger.info(
            "Dense %s is enabled for forward passes of at least %d tokens; "
            "the BF16 weights stay live for everything below that.",
            fmt.upper(),
            min_tokens,
        )
    return True
