"""TLX adapter for SGLang's vectorized-5D paged-decode path."""

from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.kvcache.aiter_unified_attention import (
    scatter_ragged_to_page_table_kernel,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_TLX_OPS: ModuleType | None = None
_TLX_IMPORT_ERROR: ImportError | None = None
_TLX_IMPORT_ATTEMPTED = False
_TLX_IMPORT_LOCK = threading.Lock()


def _load_tlx_ops() -> ModuleType | None:
    """Load TLX without replacing SGLang's regular Triton package.

    A namespaced ``fbtriton`` installation is preferred so stock ``triton``
    remains available to the rest of SGLang.  The fallback keeps the adapter
    usable with today's replacement-style fbtriton wheel, which still exposes
    its Python modules under the top-level ``triton`` package.

    Loading is deliberately lazy: selecting the default Gluon path must not
    initialize fbtriton or change which compiler normal SGLang kernels use.
    """
    global _TLX_IMPORT_ATTEMPTED, _TLX_IMPORT_ERROR, _TLX_OPS
    if _TLX_IMPORT_ATTEMPTED:
        return _TLX_OPS

    with _TLX_IMPORT_LOCK:
        if _TLX_IMPORT_ATTEMPTED:
            return _TLX_OPS
        _TLX_IMPORT_ATTEMPTED = True
        errors = []
        for package in ("fbtriton", "triton"):
            try:
                _TLX_OPS = importlib.import_module(f"{package}.language.extra.tlx.ops")
                _TLX_IMPORT_ERROR = None
                return _TLX_OPS
            except ImportError as error:
                errors.append(f"{package}: {error}")

        _TLX_IMPORT_ERROR = ImportError("; ".join(errors))
        return None


def _require_tlx_ops() -> ModuleType:
    ops = _load_tlx_ops()
    if ops is None:
        raise RuntimeError(f"TLX paged decode is unavailable: {_TLX_IMPORT_ERROR}")
    return ops


def tlx_pa_decode_available() -> bool:
    return _load_tlx_ops() is not None


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
    ops = _load_tlx_ops()
    if ops is None:
        return False
    if layer.qk_head_dim != 64 or layer.v_head_dim != 64:
        return False
    if backend.kv_cache_dtype != k_cache.dtype:
        return False
    if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        return False
    return ops.can_use_pa_decode_tlx(
        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        k_cache,
        v_cache,
        query_length=1,
        sliding_window=0,
        sinks=sinks,
    )


def _get_block_tables(backend, forward_batch, max_context_len):
    """Return the block-level page table expected by TLX.

    Normal AITER decode metadata stores a ragged, flattened list of physical
    token slots.  TLX consumes one physical page id per logical page instead.
    Build that compact view once per forward-metadata object and reuse it for
    every attention layer in the decode step.
    """
    metadata = backend.forward_metadata
    block_tables = metadata.kv_indices
    if block_tables.ndim == 2:
        return block_tables
    if block_tables.ndim != 1 or metadata.kv_indptr is None:
        raise ValueError(
            "TLX paged decode expects a 2D page table or ragged 1D kv_indices "
            "with kv_indptr"
        )

    capturing = torch.cuda.is_current_stream_capturing()
    cached = getattr(backend, "_tlx_page_table_metadata", None) is metadata
    if cached and (
        not capturing
        or getattr(backend, "_tlx_page_table_captured_metadata", None) is metadata
    ):
        return backend._tlx_page_table

    batch_size = forward_batch.batch_size
    page_size = backend.page_size
    max_blocks = (max_context_len + page_size - 1) // page_size
    page_table = (
        backend._tlx_page_table
        if cached
        else torch.zeros(
            (batch_size, max_blocks), dtype=torch.int32, device=block_tables.device
        )
    )
    block_size = 1024
    grid = (batch_size, (max(max_blocks, 1) + block_size - 1) // block_size)
    scatter_ragged_to_page_table_kernel[grid](
        block_tables,
        metadata.kv_indptr,
        page_table,
        page_table.stride(0),
        None,
        None,
        PAGE_SIZE=page_size,
        BLOCK_SIZE=block_size,
        HAS_SWA=False,
    )
    backend._tlx_page_table_metadata = metadata
    backend._tlx_page_table = page_table
    if capturing:
        # Capture the scatter once so graph replay refreshes the page table
        # from SGLang's updated ragged metadata instead of reusing warmup data.
        backend._tlx_page_table_captured_metadata = metadata
    return page_table


def _get_config(
    backend,
    query,
    key_cache,
    value_cache,
    block_tables,
    max_context_len,
):
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
        config = _require_tlx_ops().get_pa_decode_config(
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
        cache[key] = _require_tlx_ops().allocate_pa_decode_workspace(
            query, key_cache, config
        )
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
    max_context_len = int(backend.forward_metadata.max_kv_len)
    block_tables = _get_block_tables(backend, forward_batch, max_context_len)
    config = _get_config(
        backend,
        query,
        k_cache,
        v_cache,
        block_tables,
        max_context_len,
    )
    workspace = _get_workspace(backend, query, k_cache, config)
    _require_tlx_ops().pa_decode_tlx(
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
