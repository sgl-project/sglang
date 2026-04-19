"""Gluon extend-attention wrapper for MI350X (gfx950 / CDNA 4).

Drop-in replacement for ``triton_ops.extend_attention.extend_attention_fwd``,
auto-enabled on gfx950 by :class:`TritonAttnBackend` when the user has
not passed ``--disable-gluon-extend-attention``. Kernel sources are
vendored under
``sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention``; no
external paths or imports required.

Falls back transparently to the Triton reference for unsupported
head-dims or on any runtime error.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


_GLUON_SUPPORTED_HEAD_DIMS = {64, 128, 256}

_GLUON_FN: Optional[Callable] = None
_PREWARM_FN: Optional[Callable] = None
_PREWARM_MODEL_FN: Optional[Callable] = None
_PREWARMED_MODELS: set = set()


def _try_import_gluon() -> bool:
    """Populate ``_GLUON_FN`` / ``_PREWARM_FN`` from the vendored package.

    Idempotent. Returns True on success, False if the import raised (we
    will then stay on the Triton fallback).
    """
    global _GLUON_FN, _PREWARM_FN, _PREWARM_MODEL_FN
    if _GLUON_FN is not None:
        return True
    try:
        from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention import (
            gluon_extend_attention_fwd,
            prewarm_extend_attention,
            prewarm_for_model as _pfm,
        )
        _GLUON_FN = gluon_extend_attention_fwd
        _PREWARM_FN = prewarm_extend_attention
        _PREWARM_MODEL_FN = _pfm
        return True
    except Exception as e:
        logger.warning(
            f"Failed to import Gluon extend attention: {e!r}. "
            f"Falling back to Triton."
        )
        return False


_HAS_SINK_ARCHS = {
    "GptOssForCausalLM",
    "GptOssMoeForCausalLM",
    "Grok1ForCausalLM",
}


def _build_layer_spec_from_hf_config(hf_config) -> Optional[list]:
    """Derive a per-layer attention-pattern list for ``prewarm_for_model``.

    Returning the actual ``HAS_SINK`` / ``SLIDING_WINDOW_SIZE`` /
    ``LOGIT_CAP`` tuples the runtime will ask for avoids a JIT miss per
    SWA layer on the first E2E prefill.

    SGLang uses exclusive sliding windows while HF uses inclusive, so
    the runtime ``layer.sliding_window_size`` is ``config.sliding_window
    - 1``; we must prewarm with that same off-by-one value.
    """
    if hf_config is None:
        return None
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_layers is None:
        return None

    architectures = getattr(hf_config, "architectures", None) or []
    arch = architectures[0] if architectures else ""
    has_sink = arch in _HAS_SINK_ARCHS

    raw_sliding_window = getattr(hf_config, "sliding_window", None) or -1
    sliding_window = raw_sliding_window - 1 if raw_sliding_window > 0 else -1
    logit_cap = float(getattr(hf_config, "attn_logit_softcapping", None) or 0.0)
    xai_temp_len = int(getattr(hf_config, "xai_temperature_len", None) or -1)
    layer_types = getattr(hf_config, "layer_types", None)

    layers = []
    for i in range(num_layers):
        if layer_types is not None and i < len(layer_types):
            is_sliding = layer_types[i] in ("sliding_attention", "sliding")
            sw = sliding_window if is_sliding else -1
        else:
            sw = sliding_window if sliding_window > 0 else -1
        layers.append({
            "sliding_window_size": sw,
            "has_sink": has_sink,
            "logit_cap": logit_cap,
            "xai_temperature_len": xai_temp_len,
            "is_causal": True,
        })
    return layers


def prewarm_for_model(
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_causal_modes=(True,),
    hf_config=None,
) -> None:
    """Warm the Gluon JIT cache at server boot.

    Called once per ``(head_dim, num_q_heads, num_kv_heads)`` tuple by
    :class:`TritonAttnBackend`. When ``hf_config`` is supplied the
    model-specific per-layer spec (sinks, alternating SWA, logit cap)
    is used so the fast-path cache gets populated for every kernel
    variant runtime will ask for. No-op if Gluon import failed.
    """
    if _PREWARM_FN is None:
        return
    if head_dim not in _GLUON_SUPPORTED_HEAD_DIMS:
        return

    layer_spec = _build_layer_spec_from_hf_config(hf_config)
    key_layers = tuple(
        (l["sliding_window_size"], l["has_sink"], l["logit_cap"],
         l["xai_temperature_len"], l["is_causal"])
        for l in (layer_spec or [])
    )
    key = (head_dim, num_q_heads, num_kv_heads, tuple(sorted(is_causal_modes)),
           key_layers)
    if key in _PREWARMED_MODELS:
        return
    _PREWARMED_MODELS.add(key)

    if layer_spec and _PREWARM_MODEL_FN is not None:
        try:
            r = _PREWARM_MODEL_FN(
                layer_spec,
                head_dim=head_dim,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                parallel=8,
                verbose=False,
            )
            logger.info(
                f"Gluon prewarm D={head_dim} H={num_q_heads} kvH={num_kv_heads} "
                f"(model-aware): {r['num_patterns']} patterns, "
                f"{r['total_variants']} variants, {r['wall_time']:.1f}s"
            )
            return
        except Exception as e:
            logger.warning(
                f"Gluon model-aware prewarm failed: {e!r}. "
                f"Falling back to generic prewarm."
            )

    try:
        r = _PREWARM_FN(
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            is_causal_modes=is_causal_modes,
            parallel=8,
            verbose=False,
        )
        logger.info(
            f"Gluon prewarm D={head_dim} H={num_q_heads} kvH={num_kv_heads}: "
            f"{r['num_variants']} variants in {r['wall_time']:.1f}s"
        )
    except Exception as e:
        logger.warning(f"Gluon prewarm failed (non-fatal): {e!r}")


def is_gluon_extend_available() -> bool:
    """Return True iff the vendored Gluon kernel package imports cleanly."""
    return _try_import_gluon()


def make_extend_attention_fwd(triton_fallback: Callable) -> Callable:
    """Return a drop-in replacement for ``extend_attention_fwd``.

    The returned callable has the same positional/keyword signature as the
    Triton reference plus three optional dispatch hints
    (``total_prefix_len`` / ``total_extend_len`` / ``min_len_extend``)
    that :class:`TritonAttnBackend` fills from CPU tensors in
    :class:`ForwardMetadata`.

    Head-dims not in ``{64, 128, 256}`` and any runtime exception route
    to ``triton_fallback``.
    """
    if not _try_import_gluon():
        return triton_fallback

    gluon_fn = _GLUON_FN

    def _fwd(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_len_extend,
        k_scale=1.0,
        v_scale=1.0,
        sm_scale=None,
        logit_cap=0.0,
        skip_prefix_custom_mask=True,
        sliding_window_size=-1,
        sinks=None,
        window_kv_offsets=None,
        xai_temperature_len=-1,
        total_prefix_len=None,
        total_extend_len=None,
        min_len_extend=None,
    ):
        Lq = q_extend.shape[-1]
        if Lq not in _GLUON_SUPPORTED_HEAD_DIMS:
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
                f"Falling back to Triton for this call (shape="
                f"q={tuple(q_extend.shape)} kv_pool={tuple(k_buffer.shape)} "
                f"causal={is_causal})."
            )
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

    return _fwd
