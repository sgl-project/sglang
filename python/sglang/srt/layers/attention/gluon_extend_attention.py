"""Gluon extend-attention wrapper for gfx950."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


_GLUON_SUPPORTED_HEAD_DIMS = {64, 128, 256}
_FP8_KV_DTYPES = {torch.float8_e4m3fn, torch.float8_e4m3fnuz}
_GLUON_SUPPORTED_FP8_KV_DTYPES = {torch.float8_e4m3fn}

_GLUON_FN: Optional[Callable] = None


def _try_import_gluon() -> bool:
    """Cache the Gluon entry point if it imports cleanly."""
    global _GLUON_FN
    if _GLUON_FN is not None:
        return True
    try:
        from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention import (
            gluon_extend_attention_fwd,
        )
    except Exception as e:
        logger.warning(
            f"Failed to import Gluon extend attention: {e!r}. "
            f"Falling back to Triton."
        )
        return False
    _GLUON_FN = gluon_extend_attention_fwd
    return True


def is_gluon_extend_available() -> bool:
    """Return True iff the vendored Gluon kernel package imports cleanly."""
    return _try_import_gluon()


def _gluon_supports(
    q_extend: torch.Tensor,
    v_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    custom_mask,
    is_causal: bool,
) -> bool:
    """Fast guard for unsupported Gluon shapes."""
    return _gluon_unsupported_reason(
        q_extend, v_extend, k_buffer, custom_mask, is_causal
    ) is None


def _gluon_unsupported_reason(
    q_extend: torch.Tensor,
    v_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    custom_mask,
    is_causal: bool,
) -> Optional[str]:
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    if Lq not in _GLUON_SUPPORTED_HEAD_DIMS:
        return f"unsupported_head_dim_{Lq}"
    if Lq != Lv:
        return f"mismatched_qv_dim_{Lq}_{Lv}"
    kv_is_fp8 = k_buffer.dtype in _FP8_KV_DTYPES
    if kv_is_fp8 and k_buffer.dtype not in _GLUON_SUPPORTED_FP8_KV_DTYPES:
        return f"unsupported_fp8_dtype_{k_buffer.dtype}"
    if kv_is_fp8 and Lq == 256:
        return "unsupported_fp8_d256"
    if kv_is_fp8 and custom_mask is not None and Lq <= 128:
        return "unsupported_fp8_custom_mask"
    return None


def make_extend_attention_fwd(triton_fallback: Callable) -> Callable:
    """Return a Gluon wrapper that falls back to Triton."""
    if not _try_import_gluon():
        return triton_fallback

    gluon_fn = _GLUON_FN

    def _fwd(
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        custom_mask, is_causal, mask_indptr, max_len_extend,
        k_scale=1.0, v_scale=1.0, sm_scale=None,
        logit_cap=0.0, skip_prefix_custom_mask=True,
        sliding_window_size=-1, sinks=None,
        window_kv_offsets=None, xai_temperature_len=-1,
        total_prefix_len=None, total_extend_len=None, min_len_extend=None,
    ):
        def _fallback():
            return triton_fallback(
                q_extend, k_extend, v_extend, o_extend,
                k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
                custom_mask, is_causal, mask_indptr, max_len_extend,
                k_scale, v_scale, sm_scale,
                logit_cap=logit_cap,
                skip_prefix_custom_mask=skip_prefix_custom_mask,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=xai_temperature_len,
            )

        unsupported_reason = _gluon_unsupported_reason(
            q_extend, v_extend, k_buffer, custom_mask, is_causal
        )
        if unsupported_reason is not None:
            return _fallback()

        try:
            return gluon_fn(
                q_extend, k_extend, v_extend, o_extend,
                k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
                custom_mask, is_causal, mask_indptr, max_len_extend,
                k_scale=k_scale,
                v_scale=v_scale,
                sm_scale=sm_scale,
                logit_cap=logit_cap,
                skip_prefix_custom_mask=skip_prefix_custom_mask,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=xai_temperature_len,
                total_prefix_len=total_prefix_len,
                total_extend_len=total_extend_len,
                min_len_extend=min_len_extend,
            )
        except Exception as e:
            logger.warning(
                f"Gluon extend attention raised {type(e).__name__}: {e!r}. "
                f"Falling back to Triton for this call (q={tuple(q_extend.shape)} "
                f"kv_pool={tuple(k_buffer.shape)} causal={is_causal})."
            )
            return _fallback()

    return _fwd
