"""Gluon extend-attention wrapper for MI350/MI355 (gfx950 / CDNA 4).

Drop-in replacement for ``triton_ops.extend_attention.extend_attention_fwd``,
installed by :class:`TritonAttnBackend` when the user passes
``--enable-gluon-extend-attention`` on gfx950. Kernel sources are
vendored under
``sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention``; no
external paths or imports required.

Falls back transparently to the Triton reference for unsupported
head-dims or on any runtime error.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


# Optional per-call shape profiler. Enable with
# ``SGLANG_GLUON_DEBUG=profile_shapes=<path>``. Buckets (path, shape,
# feature) counts and dumps on exit / SIGUSR1 / every N calls.
from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention._debug import DEBUG

_PROFILE_PATH = DEBUG.profile_shapes_path
_PROFILE_COUNTS: dict = {}
_PROFILE_LOCK = threading.Lock()


def _profile_record(
    path: str,
    q_extend, k_buffer, custom_mask,
    qo_indptr, kv_indptr,
    sliding_window_size, sinks, logit_cap,
    max_len_extend,
) -> None:
    try:
        batch_size = qo_indptr.shape[0] - 1
        total_extend = int(q_extend.shape[0])
        total_prefix = int(kv_indptr[-1].item()) if kv_indptr is not None else -1
        Lq = int(q_extend.shape[-1])
        H = int(q_extend.shape[1])
        kv_is_fp8 = k_buffer.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        key = (
            path, Lq, H, batch_size, int(max_len_extend),
            total_extend, total_prefix,
            bool(kv_is_fp8), bool(custom_mask is not None),
            int(sliding_window_size), bool(sinks is not None),
            float(logit_cap) > 0.0,
        )
    except Exception:
        return
    with _PROFILE_LOCK:
        _PROFILE_COUNTS[key] = _PROFILE_COUNTS.get(key, 0) + 1


def _profile_dump() -> None:
    if not _PROFILE_PATH or not _PROFILE_COUNTS:
        return
    try:
        rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)
    except Exception:
        rank = 0
    out = f"{_PROFILE_PATH}.rank{rank}.tsv"
    try:
        with open(out, "w") as fh:
            fh.write(
                "path\tLq\tH\tB\tmax_len_extend\ttotal_extend\ttotal_prefix"
                "\tkv_fp8\tcustom_mask\tsliding_window\tsinks\tlogit_cap"
                "\tcount\n"
            )
            for key, count in sorted(
                _PROFILE_COUNTS.items(), key=lambda kv: -kv[1]
            ):
                fh.write("\t".join(str(x) for x in (*key, count)) + "\n")
    except Exception as e:
        logger.warning(f"Gluon shape-profile dump failed: {e!r}")


_PROFILE_DUMP_AFTER_N = DEBUG.profile_dump_after
_PROFILE_TOTAL = 0


def _profile_maybe_dump_live() -> None:
    """Periodic flush when ``profile_dump_after=<N>`` is set; safe to tail."""
    global _PROFILE_TOTAL
    if not _PROFILE_PATH or _PROFILE_DUMP_AFTER_N <= 0:
        return
    with _PROFILE_LOCK:
        _PROFILE_TOTAL += 1
        total = _PROFILE_TOTAL
    if total % _PROFILE_DUMP_AFTER_N == 0:
        _profile_dump()


if _PROFILE_PATH:
    atexit.register(_profile_dump)
    try:
        import signal
        signal.signal(signal.SIGUSR1, lambda *a: _profile_dump())
    except Exception:
        pass
    logger.warning(
        f"Gluon shape profiler active (SGLANG_GLUON_DEBUG=profile_shapes); "
        f"per-call counts dumped to {_PROFILE_PATH}.rank<N>.tsv on shutdown, "
        f"SIGUSR1, or every {_PROFILE_DUMP_AFTER_N} calls (0=off)."
    )


_GLUON_SUPPORTED_HEAD_DIMS = {64, 128, 256}

_GLUON_FN: Optional[Callable] = None

# Optional numerical-parity check: ``SGLANG_GLUON_DEBUG=compare=<N>`` runs
# BOTH Gluon and Triton for the first N calls after startup and logs
# max|diff| / mean|diff|. Unset is a no-op on the hot path.
_COMPARE_REMAINING = DEBUG.compare_remaining
_COMPARE_LOCK = threading.Lock()
_COMPARE_PATH = DEBUG.compare_log
_COMPARE_INDEX = 0


def _compare_remaining_fetch_dec() -> int:
    """Atomic ``n -= 1; return old_n`` under the compare lock."""
    global _COMPARE_REMAINING, _COMPARE_INDEX
    if _COMPARE_REMAINING <= 0:
        return 0
    r = _COMPARE_REMAINING
    _COMPARE_REMAINING -= 1
    _COMPARE_INDEX += 1
    return r


def _compare_log_diff(
    o_gluon, o_triton_ref,
    q_extend, k_buffer, kv_indptr,
    max_len_extend, sliding_window_size, sinks, logit_cap,
    custom_mask, is_causal,
) -> None:
    """Log max/mean abs diff. Only invoked when ``compare=<N>`` is active."""
    try:
        diff = (o_gluon.float() - o_triton_ref.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        ref_mag = o_triton_ref.float().abs()
        max_rel = (diff / (ref_mag + 1e-6)).max().item()
        batch_size = int(kv_indptr.shape[0] - 1) if kv_indptr is not None else -1
        total_prefix = int(kv_indptr[-1].item()) if kv_indptr is not None else -1
        Lq = int(q_extend.shape[-1])
        H = int(q_extend.shape[1])
        total_extend = int(q_extend.shape[0])
        kv_is_fp8 = k_buffer.dtype in _FP8_KV_DTYPES
        line = (
            f"[gluon-compare #{_COMPARE_INDEX}] "
            f"max|diff|={max_abs:.4f} mean|diff|={mean_abs:.5f} "
            f"max_rel={max_rel:.3f} | "
            f"Lq={Lq} H={H} B={batch_size} "
            f"max_ext={int(max_len_extend)} total_ext={total_extend} "
            f"total_pfx={total_prefix} kv_fp8={kv_is_fp8} "
            f"custom_mask={custom_mask is not None} causal={bool(is_causal)} "
            f"swa={int(sliding_window_size)} sinks={sinks is not None} "
            f"logit_cap>0={float(logit_cap) > 0.0}"
        )
        logger.warning(line)
        if _COMPARE_PATH:
            try:
                rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)
            except Exception:
                rank = 0
            with open(f"{_COMPARE_PATH}.rank{rank}.log", "a") as fh:
                fh.write(line + "\n")
    except Exception as e:
        logger.warning(f"Gluon COMPARE: diff-logging failed: {e!r}")


def _try_import_gluon() -> bool:
    """Populate the Gluon entry point from the vendored package.
    Idempotent; returns True on success, False if the import raised
    (wrapper stays on the Triton fallback)."""
    global _GLUON_FN
    if _GLUON_FN is not None:
        return True
    try:
        from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention import (
            gluon_extend_attention_fwd,
        )
        _GLUON_FN = gluon_extend_attention_fwd
        return True
    except Exception as e:
        logger.warning(
            f"Failed to import Gluon extend attention: {e!r}. "
            f"Falling back to Triton."
        )
        return False


def is_gluon_extend_available() -> bool:
    """Return True iff the vendored Gluon kernel package imports cleanly."""
    return _try_import_gluon()


_FP8_KV_DTYPES = {torch.float8_e4m3fn, torch.float8_e4m3fnuz}


def _gluon_supports(
    q_extend: torch.Tensor,
    v_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    custom_mask,
    is_causal: bool,
) -> bool:
    """Cheap pre-dispatch filter for shape / feature combos the Gluon
    kernel does not implement. Keep in sync with the ``raise`` guards
    at the top of ``gluon_extend_attention_fwd``; falling through to
    those would work but costs a raised ``ValueError`` per call."""
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    if Lq not in _GLUON_SUPPORTED_HEAD_DIMS:
        return False
    if Lq != Lv:
        return False
    if not is_causal:
        return False
    kv_is_fp8 = k_buffer.dtype in _FP8_KV_DTYPES
    if kv_is_fp8 and Lq == 256:
        return False
    if kv_is_fp8 and custom_mask is not None and Lq <= 128:
        return False
    return True


def make_extend_attention_fwd(triton_fallback: Callable) -> Callable:
    """Return a drop-in replacement for ``extend_attention_fwd``.

    Signature matches the Triton reference plus three optional dispatch
    hints (``total_prefix_len`` / ``total_extend_len`` / ``min_len_extend``)
    that :class:`TritonAttnBackend` fills from :class:`ForwardMetadata`.
    Any unsupported shape and any runtime exception routes to
    ``triton_fallback`` — never silent wrong output.
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
        if not _gluon_supports(q_extend, v_extend, k_buffer, custom_mask, is_causal):
            if _PROFILE_PATH:
                _profile_record(
                    "triton_fallback",
                    q_extend, k_buffer, custom_mask,
                    qo_indptr, kv_indptr,
                    sliding_window_size, sinks, logit_cap,
                    max_len_extend,
                )
                _profile_maybe_dump_live()
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
        if _PROFILE_PATH:
            _profile_record(
                "gluon",
                q_extend, k_buffer, custom_mask,
                qo_indptr, kv_indptr,
                sliding_window_size, sinks, logit_cap,
                max_len_extend,
            )
            _profile_maybe_dump_live()
        # Side-by-side Triton parity check; only runs when
        # ``SGLANG_GLUON_DEBUG=compare=<N>`` is set and we still have budget.
        _compare_this_call = False
        if _COMPARE_REMAINING > 0:
            with _COMPARE_LOCK:
                try:
                    _remaining = _compare_remaining_fetch_dec()
                    _compare_this_call = _remaining > 0
                except Exception:
                    _compare_this_call = False
        if _compare_this_call:
            try:
                _o_triton_ref = torch.empty_like(o_extend)
                triton_fallback(
                    q_extend, k_extend, v_extend, _o_triton_ref,
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
            except Exception as e:
                logger.warning(
                    f"Gluon COMPARE: Triton ref call failed: {e!r}; "
                    f"skipping this call's comparison."
                )
                _compare_this_call = False
        try:
            ret = gluon_fn(
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
            if _compare_this_call:
                _compare_log_diff(
                    o_extend, _o_triton_ref,
                    q_extend, k_buffer, kv_indptr,
                    max_len_extend, sliding_window_size, sinks, logit_cap,
                    custom_mask, is_causal,
                )
            return ret
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
