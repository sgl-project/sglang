"""Opt-in MXFP6 for dense BF16 projections on gfx950.

An MX checkpoint that quantizes only its routed experts leaves the rest of the
body BF16, and aiter's tuned BF16 dispatch already serves those shapes with the
best BF16 kernel it has -- on Qwen3.5-397B's widest projections it reaches roughly
60% of the device's BF16 peak. Going faster therefore means narrower inputs rather
than a different BF16 kernel.

MXFP6 rather than MXFP4 because CDNA4's matrix core runs both at the same rate
(10.1 PFLOPS, against 5 for MXFP8), so MXFP6's two extra mantissa bits are free
in peak terms. On these projections that is ~1.9x over tuned BF16 at ~4% relative
error, where MXFP4 is ~2.6x at ~16% -- and ~16% cost 8.5 points of gsm8k with
one output in eleven unparseable, while MXFP6 is accuracy-neutral.

The BF16 weight stays live next to the packed copy, because MXFP6 only wins above
roughly a thousand tokens: below that the GEMM is not compute-bound, the extra
activation quantization dominates, and decode -- where a precision loss would
compound across every step -- keeps running exactly as before.

Nothing here decides *which* layers to convert. A model registers its own policy
(which projections, minimum width, token threshold) and quark routes matching
excluded layers to this method from ``get_quant_method`` -- see
``layers.quantization.quark.dense_mx`` and ``models.qwen3_5_dense_mx``. That keeps
the tuned names with the model that was measured and out of this file.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.gemm import mxfp6_dense_aiter_hip as mxfp6
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

logger = logging.getLogger(__name__)

# One line per process, so a server log shows whether the path is live and the
# threshold it uses. Which projections were chosen is the caller's to report.
_announced = False


def mx_dense_supported() -> bool:
    """Whether this device has the MXFP6 kernels at all."""
    return mxfp6.supported()


class MxDenseLinearMethod(UnquantizedLinearMethod):
    """BF16 linear that switches to MXFP6 once the token count makes it pay.

    Weight creation and loading are the base method's, so the checkpoint still
    loads as BF16 and every consumer that reads ``layer.weight`` keeps working.
    ``apply_into`` is deliberately not overridden: it writes into caller-owned
    storage, which this GEMM cannot do without an extra copy that would eat the
    win.
    """

    def __init__(self, min_tokens: int):
        super().__init__()
        self.min_tokens = min_tokens
        self._packed: Optional[torch.Tensor] = None
        self._scale: Optional[torch.Tensor] = None
        self._out_features: Optional[int] = None
        global _announced
        if not _announced:
            _announced = True
            logger.info(
                "Dense MXFP6 is enabled for forward passes of at least %d tokens; "
                "the BF16 weights stay live for everything below that.",
                min_tokens,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if self._packed is not None:
            return
        weight = getattr(layer, "weight", None)
        if weight is None or not mxfp6.packable(weight.data):
            # Leaving the packed buffers unset keeps this layer on BF16 forever.
            return
        self._out_features = weight.shape[0]
        self._packed, self._scale = mxfp6.pack(weight.data)

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
        out = mxfp6.run(x, self._packed, self._scale, self._out_features)
        return out if bias is None else out.add_(bias)
