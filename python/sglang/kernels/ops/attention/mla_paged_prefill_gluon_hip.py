"""Opt-in Kimi-K3 gfx950 TP8 MLA paged prefill attention."""

from __future__ import annotations

import functools
import os

import torch

from sglang.kernels.ops.attention.mla_paged_decode_gluon_hip import (
    qualified_k3_mla_backend,
)
from sglang.srt.utils.common import rank0_log


def enabled() -> bool:
    return (
        os.environ.get("SGLANG_ROCM_K3_MLA_PREFILL_FUSED_BACKEND", "").lower()
        == "gluon"
    )


def can_install(backend, model_runner) -> bool:
    return enabled() and qualified_k3_mla_backend(backend, model_runner)


def covered(
    backend,
    q,
    k,
    v,
    out,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    *,
    custom_mask,
    is_causal,
    mask_indptr,
    max_query_length,
    k_scale,
    v_scale,
    sm_scale,
    logit_cap,
    sliding_window_size,
    sinks,
    window_kv_offsets,
    xai_temperature_len,
    lse,
    skip_prefix,
    skip_extend,
    page_size,
    score_mod,
    aux_tensors,
    lengths,
    identity_kv_indices,
) -> bool:
    """Admit only ordinary eager absorbed-MLA extend attention."""
    rows = q.shape[0] if isinstance(q, torch.Tensor) and q.ndim == 3 else 0
    requests = qo_indptr.numel() - 1
    return (
        not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
        and isinstance(lengths, (list, tuple))
        and len(lengths) == requests
        and 1 <= requests <= 8
        and all(type(length) is int and length > 0 for length in lengths)
        and sum(lengths) == rows
        and 1024 <= rows <= 8192
        and type(max_query_length) is int
        and max_query_length == max(lengths)
        and custom_mask is None
        and is_causal is True
        and mask_indptr is None
        and logit_cap == 0.0
        and sliding_window_size == -1
        and sinks is None
        and window_kv_offsets is None
        and xai_temperature_len <= 0
        and lse is None
        and not skip_prefix
        and not skip_extend
        and not identity_kv_indices
        and page_size == 1
        and score_mod is None
        and aux_tensors is None
        and type(k_scale) in (float, int)
        and k_scale == 1.0
        and type(v_scale) in (float, int)
        and v_scale == 1.0
        and type(sm_scale) in (float, int)
        and sm_scale == 192**-0.5
        and tuple(q.shape) == (rows, 12, 576)
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
        and tuple(k.shape) == (rows, 1, 576)
        and tuple(v.shape) == (rows, 1, 512)
        and k.dtype == v.dtype == torch.bfloat16
        and k.is_contiguous()
        and v.is_contiguous()
        and tuple(k_buffer.shape) == (655361, 1, 576)
        and tuple(v_buffer.shape) == (655361, 1, 512)
        and k_buffer.dtype == v_buffer.dtype == torch.float8_e4m3fn
        and k_buffer.stride() == v_buffer.stride() == (576, 576, 1)
        and k_buffer.untyped_storage().data_ptr()
        == v_buffer.untyped_storage().data_ptr()
        and qo_indptr.dtype in (torch.int32, torch.int64)
        and kv_indptr.dtype == torch.int32
        and tuple(qo_indptr.shape) == tuple(kv_indptr.shape) == (requests + 1,)
        and qo_indptr.is_contiguous()
        and kv_indptr.is_contiguous()
        and kv_indices.dtype == torch.int64
        and kv_indices.is_contiguous()
        and kv_indices.ndim == 1
        and 0 < kv_indices.numel() <= 196608
        and kv_indices.numel() < requests * backend.max_context_len
        and tuple(out.shape) == (rows, 12, 512)
        and out.dtype == torch.bfloat16
        and out.is_contiguous()
        and all(
            tensor.device == q.device
            for tensor in (
                k,
                v,
                out,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
            )
        )
    )


def run(
    q,
    k,
    v,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    *,
    scale,
    max_query_length,
    out,
):
    from sglang.kernels.ops.attention.mla_gluon.paged_prefill import (
        paged_attention_prefill,
    )

    return paged_attention_prefill(
        q,
        k,
        v,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        scale=scale,
        max_query_length=max_query_length,
        output_tensor=out,
    )


def install(backend, model_runner) -> bool:
    """Replace qualified eager prefill calls while retaining native fallback."""
    if not can_install(backend, model_runner):
        return False
    original = backend.extend_attention_fwd

    @functools.wraps(original)
    def extend(
        q,
        k,
        v,
        out,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_query_length,
        k_scale,
        v_scale,
        sm_scale=None,
        logit_cap=0.0,
        skip_prefix_custom_mask=True,
        sliding_window_size=-1,
        sinks=None,
        window_kv_offsets=None,
        xai_temperature_len=-1,
        lse_extend=None,
        skip_prefix=False,
        skip_extend=False,
        page_size=1,
        score_mod=None,
        aux_tensors=None,
        extend_seq_lens_cpu=None,
        identity_kv_indices=False,
    ):
        if covered(
            backend,
            q,
            k,
            v,
            out,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask=custom_mask,
            is_causal=is_causal,
            mask_indptr=mask_indptr,
            max_query_length=max_query_length,
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=xai_temperature_len,
            lse=lse_extend,
            skip_prefix=skip_prefix,
            skip_extend=skip_extend,
            page_size=page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            lengths=extend_seq_lens_cpu,
            identity_kv_indices=identity_kv_indices,
        ):
            result = run(
                q,
                k,
                v,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                scale=sm_scale,
                max_query_length=max_query_length,
                out=out,
            )
            if (
                result.dtype != out.dtype
                or tuple(result.shape) != tuple(out.shape)
                or result.device != out.device
                or result.untyped_storage().data_ptr()
                != out.untyped_storage().data_ptr()
            ):
                raise RuntimeError("Kimi-K3 MLA paged prefill output ABI changed")
            return None
        return original(
            q,
            k,
            v,
            out,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            is_causal,
            mask_indptr,
            max_query_length,
            k_scale,
            v_scale,
            sm_scale,
            logit_cap=logit_cap,
            skip_prefix_custom_mask=skip_prefix_custom_mask,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=xai_temperature_len,
            lse_extend=lse_extend,
            skip_prefix=skip_prefix,
            skip_extend=skip_extend,
            page_size=page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            identity_kv_indices=identity_kv_indices,
        )

    backend.extend_attention_fwd = extend
    backend._k3_gluon_mla_prefill_installed = True
    rank0_log("K3 Gluon MLA paged prefill enabled: M=1024..8192, requests=1..8.")
    return True
