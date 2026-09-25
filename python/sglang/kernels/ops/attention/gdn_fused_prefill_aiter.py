# SPDX-License-Identifier: Apache-2.0

"""Fail-closed adapter for AITER's gfx950 fused Qwen3-Next GDN *prefill*.

A fused AITER Gluon op replaces the four-kernel prefill chain
``causal_conv1d_split_qkv -> fused_gdn_gating -> chunk_gated_delta_rule ->
gated RMSNorm`` with a tight set of Gluon launches that fuse the work
intra-kernel (the gated RMSNorm and the per-head group-128 FP8 quant fold into
the epilogue launch -- no separate quant kernel). It is the prefill sibling of
:mod:`gdn_fused_decode_aiter` and reaches the kernel via the same
attempt-and-verify stash on the attention layer (see qwen3_next.py).

Everything is probed, nothing is assumed: platform + AITER enablement, AITER
import, Triton/Gluon toolchain, and a per-call ``covered()`` shape/dtype
contract. Any miss leaves the stash unconsumed and the caller runs the ordinary
unfused chain. It rides the standard AMD switch (``SGLANG_USE_AITER`` /
``--enable-aiter``) rather than a bespoke per-kernel flag, so it is on by default
wherever AITER is.

SGLang authors no Gluon; this only consumes AITER's, and Triton >= 3.8 is probed
rather than pinned.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

# On wherever AITER is: the standard SGLANG_USE_AITER switch (--enable-aiter) on
# HIP, not a bespoke per-kernel flag. gfx950 / Triton 3.8 specifics gate below.
_use_aiter = bool(envs.SGLANG_USE_AITER.get()) and is_hip()


def _ops():
    try:
        from aiter.ops.triton.gated_delta_net.fused_gdn_prefill_qkvz import (
            fused_gdn_prefill_qkvz,
            fused_gdn_prefill_qkvz_supported,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        logger.debug("aiter fused GDN prefill import failed: %s", exc)
        return None, None
    return fused_gdn_prefill_qkvz, fused_gdn_prefill_qkvz_supported


def available() -> bool:
    """AITER-on-HIP, import, and Gluon toolchain, in that order."""
    if not _use_aiter or not torch.cuda.is_available():
        return False
    run, _ = _ops()
    if run is None:
        return False
    import triton

    # The tiles use the Triton 3.8 Gluon dialect. ROCm backported an earlier,
    # incompatible Gluon into some 3.7 builds where the import succeeds but the
    # kernel fails to compile, so gate on the version, not just the import.
    matched = re.match(r"(\d+)\.(\d+)", triton.__version__ or "")
    if matched is None or (int(matched.group(1)), int(matched.group(2))) < (3, 8):
        logger.info(
            "aiter fused GDN prefill disabled, needs Triton >= 3.8, got %s",
            triton.__version__,
        )
        return False
    try:
        import triton.experimental.gluon  # noqa: F401
    except ImportError as exc:
        logger.info("aiter fused GDN prefill disabled, no Triton Gluon: %s", exc)
        return False
    return True


def covered(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    activation: Optional[str],
    quant_dtype: Optional[torch.dtype],
) -> Tuple[bool, str]:
    """Per-call contract check. Returns ``(covered, reason)``.

    Delegates the tensor/shape/(tokens,batch) contract to AITER's own predicate
    so the two cannot drift, and adds the one thing AITER cannot see: the model's
    output-gate activation, which the kernel hard-codes to SiLU.
    """
    if activation not in ("silu", "swish"):
        return False, f"kernel fuses a SiLU output gate, model uses {activation!r}"
    if conv_bias is None:
        return False, "conv bias is required"
    _, supported = _ops()
    if supported is None:
        return False, "aiter fused GDN prefill unavailable"
    return supported(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        quant_dtype,
    )


def run(
    *,
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: float,
    norm_eps: float,
):
    """Run the fused prefill. Mutates ``conv_state`` and ``delta_state`` in place.

    Returns ``(normalized, conv_state, delta_state, quantized, scales)``.
    ``quantized``/``scales`` are the per-head group-128 FP8 activations a
    block-FP8 ``out_proj`` consumes directly -- the path the measured prefill
    uplift runs on (FP8 model); ``normalized`` (bf16, post gated-RMSNorm) is the
    fallback when ``out_proj`` is not block-FP8.
    """
    run_fn, _ = _ops()
    return run_fn(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        norm_weight,
        scale=scale,
        eps=norm_eps,
    )
