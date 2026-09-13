# SPDX-License-Identifier: Apache-2.0

"""Fail-closed adapter for AITER's gfx950 fused Qwen3-Next GDN decode.

One AITER Gluon launch replaces the four-kernel decode chain
``fused_qkvzba_split_reshape_cat -> causal_conv1d_update ->
fused_recurrent_gated_delta_rule_packed_decode -> gated RMSNorm``, folding the
output gated RMSNorm into the recurrence kernel. The gate reaches the kernel via
the attempt-and-verify stash on the attention layer (see qwen3_next.py), the
same handoff Kimi-K3 uses for its fused KDA decode.

Everything is probed, nothing is assumed: platform, opt-in env var, AITER import,
and a per-call ``covered()`` shape/dtype contract. Any miss leaves the stash
unconsumed and the caller runs the ordinary unfused chain.

SGLang authors no Gluon; this only consumes AITER's, and Triton/Gluon
availability is probed rather than pinned.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch

from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_HEAD_DIM = 128
_CONV_WIDTH = 4


def enabled() -> bool:
    """Opt-in. Off by default, like the other fused-decode backends."""
    return os.environ.get("SGLANG_QWEN3_NEXT_GDN_FUSED_BACKEND", "").lower() == "aiter"


def _ops():
    try:
        from aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz import (
            fused_gdn_decode_qkvz,
            fused_gdn_decode_qkvz_supported,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        logger.debug("aiter fused GDN decode import failed: %s", exc)
        return None, None
    return fused_gdn_decode_qkvz, fused_gdn_decode_qkvz_supported


def available() -> bool:
    """Platform, opt-in, import, and Gluon toolchain, in that order."""
    if not is_hip() or not enabled() or not torch.cuda.is_available():
        return False
    run, _ = _ops()
    if run is None:
        return False
    try:
        # The AITER kernel is Gluon; older Triton fails at compile time with an
        # opaque error rather than an ImportError, so probe the module directly.
        import triton.experimental.gluon  # noqa: F401
    except ImportError as exc:
        logger.info("aiter fused GDN decode disabled, no Triton Gluon: %s", exc)
        return False
    return True


def covered(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    activation: Optional[str],
    quant_dtype: Optional[torch.dtype],
) -> Tuple[bool, str]:
    """Per-call contract check. Returns ``(covered, reason)``.

    Delegates the tensor contract to AITER's own predicate so the two cannot
    drift, and adds the one thing AITER cannot see: the model's output-gate
    activation, which the kernel hard-codes to SiLU.
    """
    if activation not in ("silu", "swish"):
        return False, f"kernel fuses a SiLU output gate, model uses {activation!r}"
    if conv_bias is None:
        return False, "conv bias is required"
    _, supported = _ops()
    if supported is None:
        return False, "aiter fused GDN decode unavailable"
    return supported(
        projected_qkvz,
        projected_ba,
        conv_state,
        ssm_state,
        state_indices,
        conv_weight,
        conv_bias,
        quant_dtype,
    )


def run(
    *,
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: float,
    norm_eps: float,
    pad_slot_id: int,
    quant_dtype: Optional[torch.dtype] = None,
):
    """Run the fused decode. Mutates ``conv_state`` and ``ssm_state`` in place.

    Returns ``(out, quantized, scales)``; the latter two are ``None`` unless
    ``quant_dtype`` is given.
    """
    run_fn, _ = _ops()
    return run_fn(
        projected_qkvz,
        projected_ba,
        conv_state,
        ssm_state,
        state_indices,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        norm_weight,
        scale=scale,
        norm_eps=norm_eps,
        quant_dtype=quant_dtype,
        pad_slot_id=pad_slot_id,
    )
