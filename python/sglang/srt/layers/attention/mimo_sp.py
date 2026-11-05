# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
# Adapted from luocheng25/sglang PR #40, e3ef41b18796dfb3659992e32b1269d0bf3db05e.
"""MiMo token-sharded prefill attention with the existing head-sharded KV cache.

Attention collectives stay within each DP group's attention TP group. Expert
routing and dispatch/combine are owned by the caller's existing MoE backend.
"""

import itertools
import logging
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.mimo_v2_utils import mimo_effective_prefix_lengths


@lru_cache(maxsize=2)
def _log_path(name):
    logging.getLogger(__name__).info("Using MiMo EP prefill %s", name)


_SP_SWA_PAGE = 64


def _mimo_attn_tp_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather along dim 0 over attention TP (1-byte dtypes as bytes: no FP8 collectives)."""
    tensor = tensor.contiguous()
    data = tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor
    out = data.new_empty((get_attention_tp_size() * data.shape[0], *data.shape[1:]))
    get_attention_tp_group().all_gather_into_tensor(out, data)
    return out.view(tensor.dtype)


@dataclass
class _MimoSpSwaMetadata:
    """Per-forward layout of one rank's sliding-window attention under SGLANG_MIMO_SWA_SP.

    Keys of every request segment in this rank's token slice are packed into page-64
    SHUFFLE-5D pages: cached keys left of the chunk, keys from earlier slices, then the
    slice's own keys. kv_src indexes rows of cat([cached, tails, own, zero])."""

    key: Tuple[int, int]
    q_rows: int
    tail_rows: int
    cache_rows: int
    cache_page: Optional[torch.Tensor]
    cache_offset: Optional[torch.Tensor]
    kv_src: torch.Tensor
    cu_seqlens_q: torch.Tensor
    kv_indptr: torch.Tensor
    page_ids: torch.Tensor
    last_page_lens: torch.Tensor
    max_q: int
    max_kv: int


def _mimo_sp_swa_metadata(
    forward_batch: ForwardBatch, window: int, rows: int
) -> _MimoSpSwaMetadata:
    key = (window, rows)
    meta = getattr(forward_batch, "_mimo_sp_swa_metadata", None)
    if meta is not None and meta.key == key:
        return meta
    attn_tp, rank = get_attention_tp_size(), get_attention_tp_rank()
    lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
    prefixes = mimo_effective_prefix_lengths(forward_batch)
    if prefixes is None:
        raise ValueError("SWA SP needs effective prefix lengths")
    tail = min(window, rows)
    cache_counts = [min(p, window) if n > 0 else 0 for p, n in zip(prefixes, lens)]
    cache_ends = list(itertools.accumulate(cache_counts))
    cache_rows = cache_ends[-1] if cache_ends else 0
    tail_base = cache_rows
    own_base = tail_base + attn_tp * tail
    zero_row = own_base + rows
    s0, s1 = rank * rows, (rank + 1) * rows
    kv_src, cu_q, kv_indptr, last_page = [], [0], [0], []
    max_q = max_kv = 0
    start = 0
    for i, (length, prefix) in enumerate(zip(lens, prefixes)):
        a, b = start, start + length
        start = b
        qa, qb = max(a, s0), min(b, s1)
        if qa >= qb:
            continue
        from_batch = min(qa - a, window)
        from_cache = min(prefix, window - from_batch)
        parts = []
        if from_cache:
            parts.append(torch.arange(cache_ends[i] - from_cache, cache_ends[i]))
        if from_batch:
            t = torch.arange(qa - from_batch, qa)
            sl = t // rows
            parts.append(tail_base + sl * tail + (t - sl * rows) - (rows - tail))
        parts.append(torch.arange(own_base + qa - s0, own_base + qb - s0))
        idx = torch.cat(parts)
        keys = idx.numel()
        pages = -(-keys // _SP_SWA_PAGE)
        pad = pages * _SP_SWA_PAGE - keys
        if pad:
            idx = torch.cat([idx, torch.full((pad,), zero_row, dtype=idx.dtype)])
        kv_src.append(idx)
        cu_q.append(cu_q[-1] + qb - qa)
        kv_indptr.append(kv_indptr[-1] + pages)
        last_page.append(keys - _SP_SWA_PAGE * (pages - 1))
        max_q, max_kv = max(max_q, qb - qa), max(max_kv, keys)

    device = forward_batch.positions.device
    cache_page = cache_offset = None
    if cache_rows:
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )

        req_rows = torch.cat(
            [torch.full((c,), i) for i, c in enumerate(cache_counts) if c]
        ).to(device)
        positions = torch.cat(
            [torch.arange(p - c, p) for p, c in zip(prefixes, cache_counts) if c]
        ).to(device)
        full_loc = get_req_to_token_pool().req_to_token[
            forward_batch.req_pool_indices[req_rows].long(), positions
        ]
        swa_loc = (
            get_token_to_kv_pool()
            .translate_loc_from_full_to_swa(full_loc.long())
            .long()
        )
        cache_page, cache_offset = swa_loc // _SP_SWA_PAGE, swa_loc % _SP_SWA_PAGE

    def to_device(values, dtype):
        return torch.tensor(values, dtype=dtype).to(device, non_blocking=True)

    meta = _MimoSpSwaMetadata(
        key=key,
        q_rows=cu_q[-1],
        tail_rows=tail,
        cache_rows=cache_rows,
        cache_page=cache_page,
        cache_offset=cache_offset,
        kv_src=(torch.cat(kv_src) if kv_src else torch.empty(0, dtype=torch.int64)).to(
            device, non_blocking=True
        ),
        cu_seqlens_q=to_device(cu_q, torch.int32),
        kv_indptr=to_device(kv_indptr, torch.int32),
        page_ids=torch.arange(kv_indptr[-1], dtype=torch.int32, device=device),
        last_page_lens=to_device(last_page, torch.int32),
        max_q=max_q,
        max_kv=max_kv,
    )
    forward_batch._mimo_sp_swa_metadata = meta
    return meta


def run_mimo_sp_swa(
    self,
    positions: torch.Tensor,
    hidden_states,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """SWA prefill on this rank's token slice with all heads (SGLANG_MIMO_SWA_SP)."""
    _log_path("SWA sequence parallel (64Q/8KV, head-sharded persistent cache)")
    from aiter.ops.flydsl import flydsl_paged_attention_swa_bf16

    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    from sglang.srt.model_executor.forward_context import get_attn_backend

    attn_tp, rank = get_attention_tp_size(), get_attention_tp_rank()
    group = get_attention_tp_group()
    rows = (
        hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    ).shape[0]
    heads, kv_heads = self.total_num_heads, self.total_num_kv_heads
    dq, dv = self.head_dim, self.original_v_head_dim
    local_kv_heads = kv_heads // attn_tp
    # Same window as the attention-TP kernel call (see run_mimo_flypa_swa_prefill).
    window = self.attn.sliding_window_size - int(self.attn.tp_k_head_num == 2)
    meta = _mimo_sp_swa_metadata(forward_batch, window, rows)

    qkv, _ = self.qkv_proj_full(hidden_states)
    q, k, v = qkv.split([heads * dq, kv_heads * dq, kv_heads * dv], dim=-1)
    q, k = self.rotary_emb(positions[rank * rows : (rank + 1) * rows], q, k)
    if self.v_scale is not None:
        v = v * self.v_scale
    own = torch.cat(
        [k.reshape(rows, kv_heads, dq), v.reshape(rows, kv_heads, dv)], dim=-1
    )

    # This rank's KV heads for every token of the DP group -> the head-sharded cache.
    send = own.view(rows, attn_tp, local_kv_heads, dq + dv).transpose(0, 1).contiguous()
    received = torch.empty_like(send)
    group.all_to_all_single(received, send)
    received = received.view(attn_tp * rows, local_kv_heads, dq + dv)
    backend = get_attn_backend()
    pool = backend.token_to_kv_pool
    pool.set_kv_buffer(
        self.attn,
        KVWriteLoc(
            forward_batch.out_cache_loc,
            getattr(backend.forward_metadata, "swa_out_cache_loc", None),
        ),
        received[..., :dq].contiguous(),
        received[..., dq:].contiguous(),
        None,
        None,
    )

    # Window halos: the last keys of every slice, and cached keys left of each chunk.
    sources = []
    if meta.cache_rows:
        k_cache = pool.get_key_buffer(self.attn.layer_id)
        v_cache = pool.get_value_buffer(self.attn.layer_id)
        page, offset = meta.cache_page, meta.cache_offset
        cached = torch.cat(
            [
                k_cache[page, :, :, offset, :].reshape(-1, local_kv_heads, dq),
                v_cache[page, :, offset // 8, :, offset % 8],
            ],
            dim=-1,
        )
        cached = _mimo_attn_tp_gather(cached).view(
            attn_tp, meta.cache_rows, local_kv_heads, dq + dv
        )
        sources.append(
            cached.transpose(0, 1).reshape(meta.cache_rows, kv_heads, dq + dv)
        )
    sources.append(_mimo_attn_tp_gather(own[rows - meta.tail_rows :].contiguous()))
    sources.append(own)
    sources.append(own.new_zeros((1, kv_heads, dq + dv)))
    keys = torch.cat(sources)[meta.kv_src]
    pages = keys.shape[0] // _SP_SWA_PAGE
    k_pages = (
        keys[..., :dq]
        .reshape(pages, _SP_SWA_PAGE, kv_heads, dq // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_pages = (
        keys[..., dq:]
        .reshape(pages, _SP_SWA_PAGE // 8, 8, kv_heads, dv)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )

    out = (q.new_zeros if meta.q_rows < rows else q.new_empty)((rows, heads, dv))
    if meta.q_rows:
        unit = torch.ones(1, dtype=torch.float32, device=q.device)
        flydsl_paged_attention_swa_bf16(
            q.view(rows, heads, dq)[: meta.q_rows],
            k_pages,
            v_pages,
            meta.cu_seqlens_q,
            meta.kv_indptr,
            meta.page_ids,
            meta.max_q,
            meta.max_kv,
            window_left=window,
            kv_last_page_lens=meta.last_page_lens,
            q_descale=unit,
            k_descale=unit,
            v_descale=unit,
            out=out[: meta.q_rows],
            sink_ptr=self.attention_sink_full,
            softmax_scale=self.scaling,
            stream=torch.cuda.current_stream(q.device),
        )
    return self.o_proj.quant_method.apply(
        SimpleNamespace(weight=self.o_proj_full), out.view(rows, heads * dv)
    )
