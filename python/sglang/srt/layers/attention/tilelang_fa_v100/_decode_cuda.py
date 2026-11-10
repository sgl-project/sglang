"""Opt-in hand-written CUDA long-context grouped-decode partial (SM70).

Replaces the TileLang split-KV decode partial with a native SM70 kernel for the
exact Qwen3.8-27B TP4 shape (H6 / Hkv1 / D256, E5M2 byte KV, page size 16).
The kernel streams each split's K/V from the paged cache exactly once, which
cuts DRAM traffic versus the TileLang codegen on the same layout. The partial
output ABI matches ``_decode_partial_kernel`` exactly so downstream code can
reuse the unchanged TileLang combine kernel.

Gated by ``SGLANG_V100_DECODE_CUDA=1``; falls back to TileLang otherwise.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import torch
from sglang.kernels.sm70_paths import sm70_csrc
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SRC_PATH = sm70_csrc("sm70_longctx_decode.cu")
_EXT = None
_OPS_LOAD_ATTEMPTED = False

PAGE_SIZE = 16  # fixed page granularity supported by the CUDA kernel
QSA_DECODE_TARGET_CTAS = 160
QSA_DECODE_TOKENS_PER_SPLIT = 32


def sm70_cuda_decode_enabled() -> bool:
    """Whether the CUDA decode partial is requested.

    Defaults to on for SM70 (the kernel is bit-exact and faster than the
    TileLang codegen on the same layout); set ``SGLANG_V100_DECODE_CUDA=0`` to
    fall back to the TileLang partial.
    """
    return os.environ.get("SGLANG_V100_DECODE_CUDA", "1") == "1"


def sm70_cuda_decode_available() -> bool:
    """Whether the CUDA partial can be used on this GPU."""
    if not sm70_cuda_decode_enabled():
        return False
    if not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability()
    except Exception:
        return False
    return capability == (7, 0)


def _load_sm70_cuda_decode_ops():
    """Lazy-load the standalone SM70 long-context decode extension (JIT-built)."""
    global _EXT, _OPS_LOAD_ATTEMPTED
    if _EXT is not None:
        return _EXT
    if _OPS_LOAD_ATTEMPTED:
        return None
    _OPS_LOAD_ATTEMPTED = True
    if not _SRC_PATH.is_file():
        logger.warning("SM70 CUDA decode partial source not found: %s", _SRC_PATH)
        return None
    from torch.utils.cpp_extension import load_inline

    build_directory = os.environ.get(
        "SGLANG_V100_DECODE_CUDA_BUILD_DIR", "/tmp/sglang_sm70_longctx_decode"
    )
    os.makedirs(build_directory, exist_ok=True)
    try:
        _EXT = load_inline(
            name="sglang_sm70_longctx_decode_v100",
            cpp_sources="",
            cuda_sources=_SRC_PATH.read_text(),
            functions=None,
            is_python_module=True,
            verbose=False,
            build_directory=build_directory,
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        )
    except Exception:  # pragma: no cover - build/environment failures
        logger.exception("SM70 CUDA decode partial failed to build")
        return None
    logger.info_once("SM70 (V100): CUDA long-context decode partial loaded.")
    return _EXT


def sm70_cuda_decode_partial(
    q,
    k_cache,
    v_cache,
    page_table,
    seq_lens,
    max_splits,
    min_tokens_per_split,
    softmax_scale,
    k_scale,
    v_scale,
):
    """Run the CUDA split-KV partial, returning (partial_o, partial_lse)."""
    ext = _load_sm70_cuda_decode_ops()
    if ext is None:
        raise RuntimeError(
            "SM70 CUDA decode partial requested but extension is unavailable."
        )
    batch, heads, dim = q.shape
    partial_o = torch.empty(
        (batch, max_splits, heads, dim), dtype=torch.float16, device=q.device
    )
    partial_lse = torch.empty(
        (batch, max_splits, heads), dtype=torch.float32, device=q.device
    )
    ext.sm70_longctx_decode(
        q.contiguous(),
        k_cache.view(torch.uint8).contiguous(),
        v_cache.view(torch.uint8).contiguous(),
        page_table.to(dtype=torch.int32).contiguous(),
        seq_lens.to(dtype=torch.int32).contiguous(),
        int(max_splits),
        int(min_tokens_per_split),
        float(softmax_scale),
        float(k_scale),
        float(v_scale),
        partial_o,
        partial_lse,
    )
    return partial_o, partial_lse


def sm70_cuda_qsa_prefill(
    q,
    k_cache,
    v_cache,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    softmax_scale,
    page_size=None,
):
    """Run exact-shape QSA chunk-prefill directly from E5M2 or FP16 KV."""
    if envs.SGLANG_SM70_QSA_GROUPED_PREFILL.get():
        from ._grouped_qsa import can_use_grouped_qsa, grouped_qsa_prefill

        if can_use_grouped_qsa(q, k_cache, v_cache, indices, page_size):
            result = grouped_qsa_prefill(
                q,
                k_cache,
                v_cache,
                req_to_token,
                req_indices,
                indices,
                seq_lens,
                softmax_scale,
                page_size,
            )
            if result is not None:
                return result
    ext = _load_sm70_cuda_decode_ops()
    if ext is None:
        raise RuntimeError("SM70 CUDA QSA prefill extension is unavailable.")
    output = torch.empty_like(q)
    ext.sm70_qsa_prefill(
        q.contiguous(),
        _qsa_cache_view(k_cache),
        _qsa_cache_view(v_cache),
        req_to_token.to(dtype=torch.int32).contiguous(),
        req_indices.to(dtype=torch.int32).contiguous(),
        indices.to(dtype=torch.int32).contiguous(),
        seq_lens.to(dtype=torch.int32).contiguous(),
        float(softmax_scale),
        output,
    )
    return output


def _qsa_cache_view(cache: torch.Tensor) -> torch.Tensor:
    """View the paged KV cache for the CUDA op.

    The E5M2 pool is one byte per element and is read as uint8; an FP16 pool is
    read as half. Reinterpreting an FP16 cache to bytes would double the
    element count, so the view is dtype-dependent.
    """
    if cache.dtype == torch.float8_e5m2:
        return cache.view(torch.uint8).contiguous()
    return cache.contiguous()


def sm70_cuda_qsa_decode(
    q,
    k_cache,
    v_cache,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    softmax_scale,
):
    """Run QSA split-KV decode directly from selected E5M2 or FP16 cache rows."""
    ext = _load_sm70_cuda_decode_ops()
    if ext is None:
        raise RuntimeError("SM70 CUDA QSA decode extension is unavailable.")
    batch, heads, dim = q.shape
    max_splits = max(1, math.ceil(QSA_DECODE_TARGET_CTAS / batch))
    partial_o = torch.empty(
        (batch, max_splits, heads, dim), dtype=torch.float16, device=q.device
    )
    partial_lse = torch.empty(
        (batch, max_splits, heads), dtype=torch.float32, device=q.device
    )
    seq_lens = seq_lens.to(dtype=torch.int32).contiguous()
    indices = indices.to(dtype=torch.int32).contiguous()
    ext.sm70_qsa_decode(
        q.contiguous(),
        _qsa_cache_view(k_cache),
        _qsa_cache_view(v_cache),
        req_to_token.to(dtype=torch.int32).contiguous(),
        req_indices.to(dtype=torch.int32).contiguous(),
        indices,
        seq_lens,
        max_splits,
        QSA_DECODE_TOKENS_PER_SPLIT,
        float(softmax_scale),
        partial_o,
        partial_lse,
    )
    if (
        1 <= batch <= 4
        and heads == 6
        and dim == 256
        and max_splits <= 160
        and torch.cuda.get_device_capability(q.device) == (7, 0)
        and envs.SGLANG_SM70_QSA_COMBINE.get()
    ):
        from sglang.kernels.ops.attention.sm70_qsa_combine import combine

        return combine(
            partial_o,
            partial_lse,
            seq_lens,
            indices.shape[1],
            QSA_DECODE_TOKENS_PER_SPLIT,
        )
    from ._kernels_paged_decode import _decode_combine_kernel

    combine = _decode_combine_kernel(
        batch,
        heads,
        dim,
        max_splits,
        256,
        QSA_DECODE_TOKENS_PER_SPLIT,
        selected_tokens=indices.shape[1],
    )
    return combine(partial_o, partial_lse, seq_lens)


def sm70_cuda_qsa_indexer_decode(
    q,
    k_cache,
    page_table,
    context_lens,
    max_model_len,
    score_scale,
):
    """Score compressed QSA index keys without Volta MMA head padding."""
    ext = _load_sm70_cuda_decode_ops()
    if ext is None:
        raise RuntimeError("SM70 CUDA QSA indexer extension is unavailable.")
    logits = torch.empty(
        (q.shape[0], max_model_len), dtype=torch.float32, device=q.device
    )
    ext.sm70_qsa_indexer_decode(
        q.contiguous(),
        k_cache.contiguous(),
        page_table.to(dtype=torch.int32).contiguous(),
        context_lens.to(dtype=torch.int32).contiguous(),
        int(max_model_len),
        float(score_scale),
        logits,
    )
    return logits
