"""TLX adapter for SGLang's vectorized-5D paged-decode path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

try:
    from triton.language.extra.tlx.ops import (
        allocate_pa_decode_workspace,
        can_use_pa_decode_tlx,
        get_pa_decode_config,
        pa_decode_tlx,
    )

    _TLX_IMPORT_ERROR = None
except ImportError as error:  # The stock SGLang Triton package has no TLX.
    allocate_pa_decode_workspace = None
    can_use_pa_decode_tlx = None
    get_pa_decode_config = None
    pa_decode_tlx = None
    _TLX_IMPORT_ERROR = error


def tlx_pa_decode_available() -> bool:
    return _TLX_IMPORT_ERROR is None


def _auto_prefers_tlx(batch_size: int, max_context_len: int, page_size: int) -> bool:
    """Return whether repeatable complete-wrapper results favor TLX.

    Keep automatic routing on Gluon until the fixed-clock MI350 sweep has a
    repeatable TLX winner.  Forced ``tlx`` mode remains available for testing
    and graph-captured deployments.
    """
    del batch_size, max_context_len, page_size
    return False


def can_forward_decode_vectorized_5d_tlx(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sinks,
) -> bool:
    if not tlx_pa_decode_available():
        return False
    if layer.qk_head_dim != 64 or layer.v_head_dim != 64:
        return False
    if backend.kv_cache_dtype != k_cache.dtype:
        return False
    if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        return False
    return can_use_pa_decode_tlx(
        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        k_cache,
        v_cache,
        query_length=1,
        sliding_window=0,
        sinks=sinks,
    )


def _get_config(backend, query, key_cache, value_cache, block_tables, max_context_len):
    cache = getattr(backend, "_tlx_pa_decode_configs", None)
    if cache is None:
        cache = {}
        backend._tlx_pa_decode_configs = cache
    key = (
        query.device,
        query.dtype,
        tuple(query.shape),
        tuple(key_cache.shape[1:]),
        tuple(value_cache.shape[1:]),
        tuple(block_tables.shape),
        max_context_len,
    )
    config = cache.get(key)
    if config is None:
        config = get_pa_decode_config(
            query,
            key_cache,
            value_cache,
            block_tables,
            query_length=1,
            max_context_len=max_context_len,
        )
        cache[key] = config
    return config


def _get_workspace(backend, query, key_cache, config):
    cache = getattr(backend, "_tlx_pa_decode_workspaces", None)
    if cache is None:
        cache = {}
        backend._tlx_pa_decode_workspaces = cache
    key = (
        query.device,
        query.dtype,
        tuple(query.shape),
        key_cache.shape[1],
        config,
    )
    if key not in cache:
        cache[key] = allocate_pa_decode_workspace(query, key_cache, config)
    return cache[key]


def forward_decode_vectorized_5d_tlx(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    sinks,
) -> None:
    """Run TLX against SGLang's existing packed cache and page table."""
    query = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    output_view = output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
    block_tables = backend.forward_metadata.kv_indices
    max_context_len = int(backend.forward_metadata.max_kv_len)
    config = _get_config(
        backend, query, k_cache, v_cache, block_tables, max_context_len
    )
    workspace = _get_workspace(backend, query, k_cache, config)
    pa_decode_tlx(
        output_view,
        query,
        k_cache,
        v_cache,
        forward_batch.seq_lens,
        block_tables,
        layer.scaling,
        query_length=1,
        max_context_len=max_context_len,
        workspace=workspace,
        config=config,
    )


def should_use_tlx_decode(
    mode: str,
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sinks,
) -> bool:
    if mode not in ("gluon", "tlx", "auto"):
        raise ValueError(
            f"Invalid SGLANG_AITER_5D_DECODE_BACKEND={mode!r}; "
            "expected gluon, tlx, or auto"
        )
    if mode == "gluon":
        return False
    cache = getattr(backend, "_tlx_pa_decode_eligibility", None)
    if cache is None:
        cache = {}
        backend._tlx_pa_decode_eligibility = cache
    key = (
        q.device,
        q.dtype,
        tuple(q.shape),
        tuple(k_cache.shape[1:]),
        tuple(v_cache.shape[1:]),
        layer.qk_head_dim,
        layer.v_head_dim,
        layer.sliding_window_size,
        sinks is None,
    )
    supported = cache.get(key)
    if supported is None:
        supported = can_forward_decode_vectorized_5d_tlx(
            backend, q, layer, forward_batch, k_cache, v_cache, sinks
        )
        cache[key] = supported
    if mode == "tlx":
        if not supported:
            detail = f": {_TLX_IMPORT_ERROR}" if _TLX_IMPORT_ERROR is not None else ""
            raise RuntimeError(
                "TLX paged decode was forced but is unavailable or unsupported"
                f"{detail}"
            )
        return True
    if not supported:
        return False
    return _auto_prefers_tlx(
        forward_batch.batch_size,
        int(backend.forward_metadata.max_kv_len),
        k_cache.shape[3],
    )
