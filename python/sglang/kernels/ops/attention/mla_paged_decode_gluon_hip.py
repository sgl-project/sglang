"""Opt-in Kimi-K3 gfx950 TP8 MLA paged decode attention.

The native Triton backend retains KV writes, physical-index construction, and
output allocation.  This adapter replaces only the read-only decode attention
calculation on the exact qualified cache and metadata layout.
"""

from __future__ import annotations

import functools
import os

import torch

from sglang.kernels.ops.attention.kda_whole_layer_gluon_hip import (
    _has_required_gluon_api,
    _rocm_arch,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import rank0_log


def enabled() -> bool:
    return (
        os.environ.get("SGLANG_ROCM_K3_MLA_DECODE_FUSED_BACKEND", "").lower() == "gluon"
    )


def entrypoint_name(rows: int) -> str | None:
    if rows in (1, 2, 4, 8, 64, 128, 256):
        return f"paged_attention_decode_m{rows}"
    if rows in (12, 16):
        return "paged_attention_decode_m12_16"
    if rows in (24, 32):
        return "paged_attention_decode_m24_32"
    return None


def qualified_k3_mla_backend(backend, model_runner) -> bool:
    """Qualify the common loaded Kimi-K3 1M-context TP8 MLA contract."""
    model_config = model_runner.model_config
    hf_config = model_config.hf_config
    architectures = tuple(getattr(hf_config, "architectures", ()) or ())
    server_args = get_server_args()
    return (
        is_hip()
        and _has_required_gluon_api()
        and _rocm_arch(model_runner.gpu_id) == "gfx950"
        and any(name.startswith("KimiK3") for name in architectures)
        and backend.use_mla
        and backend.dcp_size == 1
        and backend.page_size == 1
        and not model_runner.kv_index_translator.is_translating
        and backend.max_context_len == 1048576
        and not backend.enable_deterministic
        and not model_runner.is_draft_worker
        and not getattr(server_args, "enable_lora", False)
        and not getattr(server_args, "speculative_algorithm", None)
        and get_parallel().attn_tp_size == 8
        and backend.num_head == 12
        and model_runner.kv_cache_dtype == torch.float8_e4m3fn
        and (
            model_config.qk_nope_head_dim,
            model_config.qk_rope_head_dim,
            model_config.kv_lora_rank,
            model_config.v_head_dim,
        )
        == (128, 64, 512, 128)
    )


def can_install(backend, model_runner) -> bool:
    """Admit only an explicitly enabled qualified decode backend."""
    return enabled() and qualified_k3_mla_backend(backend, model_runner)


def covered(
    backend,
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    sm_scale,
    k_scale,
    v_scale,
    *,
    logit_cap,
    sinks,
    xai_temperature_len,
    has_mla,
    use_pdl,
    page_size,
    score_mod,
    aux_tensors,
) -> bool:
    """Validate the mutable CUDA-graph views before selecting a kernel."""
    rows = q.shape[0] if isinstance(q, torch.Tensor) and q.ndim == 3 else 0
    return (
        entrypoint_name(rows) is not None
        and has_mla
        and not use_pdl
        and page_size == 1
        and logit_cap == 0.0
        and sinks is None
        and xai_temperature_len <= 0
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
        and q.stride(-1) == 1
        and tuple(k_buffer.shape) == (655361, 1, 576)
        and tuple(v_buffer.shape) == (655361, 1, 512)
        and k_buffer.dtype == v_buffer.dtype == torch.float8_e4m3fn
        and k_buffer.stride() == v_buffer.stride() == (576, 576, 1)
        and k_buffer.untyped_storage().data_ptr()
        == v_buffer.untyped_storage().data_ptr()
        and kv_indptr.dtype == torch.int32
        and kv_indptr.is_contiguous()
        and tuple(kv_indptr.shape) == (rows + 1,)
        and kv_indices.dtype == torch.int64
        and kv_indices.is_contiguous()
        and tuple(kv_indices.shape) == (268435456,)
        and tuple(o.shape) == (rows, 12, 512)
        and o.dtype == torch.bfloat16
        and o.is_contiguous()
        and all(
            tensor.device == q.device
            for tensor in (k_buffer, v_buffer, o, kv_indptr, kv_indices)
        )
        and backend.max_context_len == 1048576
    )


def run(q, k_buffer, v_buffer, kv_indptr, kv_indices, *, scale, max_context, out):
    from sglang.kernels.ops.attention.mla_gluon import paged_decode

    name = entrypoint_name(q.shape[0])
    if name is None:
        raise ValueError(f"Unqualified Kimi-K3 MLA paged decode M={q.shape[0]}")
    return getattr(paged_decode, name)(
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        scale=scale,
        max_context=max_context,
        output_tensor=out,
    )


def install(backend, model_runner) -> bool:
    """Replace the backend's decode callable without changing native fallback."""
    if not can_install(backend, model_runner):
        return False
    original = backend.decode_attention_fwd

    @functools.wraps(original)
    def decode(
        q,
        k_buffer,
        v_buffer,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        k_scale,
        v_scale,
        logit_cap=0.0,
        sinks=None,
        xai_temperature_len=-1,
        has_mla=False,
        use_pdl=False,
        page_size=1,
        score_mod=None,
        aux_tensors=None,
        enable_lean=None,
        lean_Mp=None,
        lean_Lp=None,
        lean_Op=None,
        lean_locks=None,
    ):
        if covered(
            backend,
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            sm_scale,
            k_scale,
            v_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            has_mla=has_mla,
            use_pdl=use_pdl,
            page_size=page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        ):
            result = run(
                q,
                k_buffer,
                v_buffer,
                kv_indptr,
                kv_indices,
                scale=sm_scale,
                max_context=backend.max_context_len,
                out=o,
            )
            if (
                result.dtype != o.dtype
                or tuple(result.shape) != tuple(o.shape)
                or result.device != o.device
                or result.untyped_storage().data_ptr() != o.untyped_storage().data_ptr()
            ):
                raise RuntimeError("Kimi-K3 MLA paged decode output ABI changed")
            return None
        return original(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            k_scale,
            v_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            has_mla=has_mla,
            use_pdl=use_pdl,
            page_size=page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            enable_lean=enable_lean,
            lean_Mp=lean_Mp,
            lean_Lp=lean_Lp,
            lean_Op=lean_Op,
            lean_locks=lean_locks,
        )

    backend.decode_attention_fwd = decode
    backend._k3_gluon_mla_decode_installed = True
    rank0_log("K3 Gluon MLA paged decode enabled: M=1,2,4,8,12,16,24,32,64,128,256.")
    return True
