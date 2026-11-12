# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
from __future__ import annotations

from typing import Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_dcp,
)
from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.srt.runtime_context import get_parallel

TRTLLM_MAX_BATCH_ROWS = 65535
# FlashInfer carves heads * rows * 256 float2 softmax slots out of the workspace per call;
# the rest is kernel scratch.
_SOFTMAX_BYTES_PER_ROW_HEAD = 256 * 8
_SCRATCH_RESERVE_BYTES = 128 << 20


def trtllm_rows_per_call(*, workspace_bytes: int, heads: int) -> int:
    rows = (workspace_bytes - _SCRATCH_RESERVE_BYTES) // (
        heads * _SOFTMAX_BYTES_PER_ROW_HEAD
    )
    return max(1, min(TRTLLM_MAX_BATCH_ROWS, rows))


def window_row_tiles(rows: int, wave_rows: int, tail_max: int) -> list:
    """K31HT_DRAFT_DCP1_TILE: row tiles for one DCP-1 draft window call.

    At 12 heads the trtllm-gen sparse kernel runs one CTA per row up to one
    wave (wave_rows = SM count) and is priced per wave: a call one row past
    a wave boundary pays a whole extra wave. When the last, partial wave
    holds at most ``tail_max`` rows, those rows run as their own call, where
    the kernel picks its multi-CTA form. Returns [(0, rows)] for no split.
    """
    if wave_rows <= 0 or rows <= wave_rows:
        return [(0, rows)]
    tail = rows % wave_rows
    if tail == 0 or tail > tail_max:
        return [(0, rows)]
    head = rows - tail
    return [(0, head), (head, rows)]


def _k31ht_tiled_trtllm_sparse_attention(
    *, query, block_tables, seq_lens, cu_seqlens_q, need_lse, lse, step, row_tiles, attention_kwargs
):
    """K31HT_DRAFT_DCP1_TILE: dcp_trtllm_sparse_attention's per-call loop over the row tiles of
    window_row_tiles, each tile cut at ``step`` rows."""
    import flashinfer.decode

    rows, _, heads, _ = query.shape
    tiles = row_tiles
    if tiles[0][0] != 0 or tiles[-1][1] != rows or any(
        a[1] != b[0] for a, b in zip(tiles, tiles[1:])
    ):
        raise ValueError(f"row_tiles {tiles} do not cover [0, {rows})")
    bounds = [
        (start, min(start + step, hi))
        for lo, hi in tiles
        for start in range(lo, hi, step)
    ]
    if len(bounds) > 1:
        # Calls share the workspace and the multi-CTA counter buffer: keep
        # them strictly ordered.
        attention_kwargs["enable_pdl"] = False
    top_k_lens = attention_kwargs.pop("sparse_mla_top_k_lens", None)
    outputs = []
    for start, end in bounds:
        result = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query[start:end],
            block_tables=block_tables[start:end],
            seq_lens=seq_lens[start:end],
            return_lse=need_lse,
            lse=lse[start:end] if need_lse else None,
            sparse_mla_top_k_lens=(
                None if top_k_lens is None else top_k_lens[start:end]
            ),
            **attention_kwargs,
        )
        output = result[0] if need_lse else result
        output = output.view(end - start, heads, -1)
        if need_lse:
            fixup_zero_kv_rows(
                output,
                lse[start:end],
                seq_lens[start:end],
                cu_seqlens_q[: end - start + 1],
                1,
            )
        outputs.append(output)
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output.view(rows, 1, heads, -1), lse


class DcpPageTableCache(msgspec.Struct):
    """NOTE(vincentzed): GLM-5.2/5.3 reuse the previous layer's top-k tensor in
    57 of 78 layers; remapping once per fresh top-k cuts 78 launches/step to 21.
    """

    metadata: object = None
    layer_id: int = -1
    topk_indices: Optional[torch.Tensor] = None
    page_table: Optional[torch.Tensor] = None
    counts: Optional[torch.Tensor] = None


def localize_dcp_page_table(
    cache: DcpPageTableCache,
    *,
    metadata: object,
    layer_id: int,
    topk_indices: Optional[torch.Tensor],
    page_table: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reuse = (
        topk_indices is not None
        and cache.metadata is metadata
        and cache.layer_id == layer_id - 1
        and cache.topk_indices is topk_indices
    )
    if not reuse:
        cache.metadata = metadata
        cache.topk_indices = topk_indices
        cache.page_table, cache.counts = transform_index_page_table_dcp(
            page_table, dcp_size, dcp_rank
        )
    cache.layer_id = layer_id
    return cache.page_table, cache.counts


def dcp_trtllm_sparse_attention(
    *,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    # False also skips the zero-KV-row fixup: every row must own a key.
    row_tiles: Optional[list] = None,
    need_lse: bool = True,
    **attention_kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    import flashinfer.decode

    rows, _, heads, _ = query.shape
    out_dtype = torch.bfloat16
    if rows == 0:
        return (
            torch.empty(
                (0, 1, heads, attention_kwargs["kv_lora_rank"]),
                dtype=out_dtype,
                device=query.device,
            ),
            torch.empty((0, heads), dtype=torch.float32, device=query.device),
        )
    lse = (
        torch.empty((rows, heads), dtype=torch.float32, device=query.device)
        if need_lse
        else None
    )
    workspace = attention_kwargs["workspace_buffer"]
    step = trtllm_rows_per_call(
        workspace_bytes=workspace.numel() * workspace.element_size(), heads=heads
    )
    if row_tiles:
        return _k31ht_tiled_trtllm_sparse_attention(
            query=query,
            block_tables=block_tables,
            seq_lens=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            need_lse=need_lse,
            lse=lse,
            step=step,
            row_tiles=row_tiles,
            attention_kwargs=attention_kwargs,
        )
    if rows > step:
        attention_kwargs["enable_pdl"] = False
    top_k_lens = attention_kwargs.pop("sparse_mla_top_k_lens", None)
    outputs = []
    for start in range(0, rows, step):
        end = min(start + step, rows)
        result = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query[start:end],
            block_tables=block_tables[start:end],
            seq_lens=seq_lens[start:end],
            return_lse=need_lse,
            lse=lse[start:end] if need_lse else None,
            sparse_mla_top_k_lens=(
                None if top_k_lens is None else top_k_lens[start:end]
            ),
            **attention_kwargs,
        )
        output = result[0] if need_lse else result
        output = output.view(end - start, heads, -1)
        if need_lse:
            fixup_zero_kv_rows(
                output,
                lse[start:end],
                seq_lens[start:end],
                cu_seqlens_q[: end - start + 1],
                1,
            )
        outputs.append(output)
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output.view(rows, 1, heads, -1), lse


def dcp_split_rows(rows: int) -> Tuple[int, int, int, int]:
    """``(per, slot, lo, hi)``: every rank owns a stride of ``per`` rows starting
    at ``slot``, and scores the rows ``[lo, hi)`` of it that exist.

    The uniform stride makes the rank-major concatenation of the strides land
    row ``i`` back at row ``i``. ``slot`` is not clamped to ``rows``: the
    in-place all-gather needs this rank's own slot even when it has no rows.
    """
    group = get_parallel().dcp_group
    per = -(-rows // group.world_size)
    slot = group.rank_in_group * per
    return per, slot, min(slot, rows), min(slot + per, rows)


def dcp_all_gather_rows_(buf: torch.Tensor, per: int, slot: int) -> None:
    """In-place inverse of ``dcp_split_rows`` on ``world_size * per`` rows of
    ``buf``. Rows at and past the real row count are left unspecified."""
    group = get_parallel().dcp_group
    out = buf[: group.world_size * per]
    group.all_gather_into_tensor(out, out[slot : slot + per])
