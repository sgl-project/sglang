"""QSA sparse attention over the K/V host offload (``--enable-qsa-kv-offload``).

Under the offload, the device K/V pool of the QSA layers holds only the hot-cache
and scratch rows of ``mem_cache/qsa_kv_offload.py``. These functions are the
backend's offload paths: store the new K/V into device rows and through to the
host, gather a prefill's cached context from the host, and run decode / target
verify attention through the offload's per-forward device-row table. In the
debug verify mode the full pool stays on the GPU and every offloaded read is
compared with the full pool's.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import msgspec
import torch

from sglang.srt.layers.attention.qsa.sparse_attn import (
    pack_qsa_prefill_kv,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
)


class QSAKVGatherSource(msgspec.Struct, frozen=True):
    """Where sparse attention reads K/V: row ``i``, column ``j`` reads
    ``k_buffer[slot_table[row_requests[i], indices[i, j]]]`` for
    ``0 <= indices[i, j] < sequence_lengths[i]``."""

    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    slot_table: torch.Tensor
    row_requests: torch.Tensor
    sequence_lengths: torch.Tensor


def _pack_gathered_kv(source: QSAKVGatherSource, indices: torch.Tensor):
    """(valid counts, packed K, packed V) of the decode gather, for verification."""
    rows, width = indices.shape
    device = indices.device
    counts = torch.empty(rows, dtype=torch.int32, device=device)
    cu = torch.arange(rows + 1, dtype=torch.int32, device=device) * width
    packed_k = torch.empty(
        (rows * width, *source.k_buffer.shape[1:]),
        dtype=source.k_buffer.dtype,
        device=device,
    )
    packed_v = torch.empty_like(packed_k)
    qwen_sparse_valid_counts_triton(
        source.sequence_lengths, indices, counts, rows, width
    )
    qwen_sparse_kv_extraction_compact_triton(
        source.k_buffer,
        source.v_buffer,
        source.slot_table,
        source.row_requests,
        indices,
        source.sequence_lengths,
        cu,
        packed_k,
        packed_v,
        rows,
        width,
        zero_fill_cols=width,
    )
    return counts, packed_k, packed_v


def store_offloaded_kv(*, backend, layer, forward_batch, k, v) -> None:
    """Store a forward's new K/V into the offload's device rows (decode: the
    tail ring, extend: the scratch) and write it through to the host."""
    offload = backend._kv_offload
    mode = forward_batch.forward_mode
    if mode.is_decode() or backend._is_speculative_paged_mode(mode):
        metadata = backend._resolve_metadata(forward_batch)
        rows = offload.new_token_rows(
            row_seq_lens=metadata.sequence_lengths,
            row_req=backend._row_requests(
                forward_batch=forward_batch, metadata=metadata
            ),
        )
    else:
        rows = offload.extend_rows(forward_batch.out_cache_loc.numel())
    pool = backend.token_to_kv_pool
    layer_idx = pool.full_attention_layer_index(layer.layer_id)
    if offload.verify:
        if layer_idx == 0:
            offload.snapshot_mapping()
        backend.kv_layers.layer(layer.layer_id).store(
            layer, forward_batch.out_cache_loc, k, v
        )
        offload.mirror(
            layer_idx=layer_idx,
            rows=rows,
            loc=forward_batch.out_cache_loc,
            pool_k=pool.full_kv_pool.k_buffer[layer_idx],
            pool_v=pool.full_kv_pool.v_buffer[layer_idx],
        )
        return
    backend.kv_layers.layer(layer.layer_id).store(layer, rows, k, v)
    offload.write_through(
        layer_idx=layer_idx, rows=rows, loc=forward_batch.out_cache_loc
    )


def gather_offloaded_context(
    *,
    backend,
    layer,
    forward_batch,
    sequence_lens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_k: int,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Every context token of each request, in ``cu_seqlens_k`` order, read
    from the host (the layout ``pack_qsa_prefill_kv`` builds from the pool)."""
    device = sequence_lens.device
    owner = torch.repeat_interleave(
        torch.arange(sequence_lens.numel(), device=device),
        sequence_lens.long(),
        output_size=total_k,
    )
    positions = torch.arange(total_k, device=device) - cu_seqlens_k[:-1].long()[owner]
    offload = backend._kv_offload
    slots = offload.req_to_token[
        forward_batch.req_pool_indices.long()[owner], positions
    ]
    pool = backend.token_to_kv_pool
    layer_idx = pool.full_attention_layer_index(layer.layer_id)
    k, v = offload.gather_context(layer_idx=layer_idx, slots=slots)
    pool_k = pool.layer_router.read(layer.layer_id).key()
    pool_v = pool.layer_router.read(layer.layer_id).value()
    packed = (
        k.view(pool_k.dtype).to(output_dtype),
        v.view(pool_k.dtype).to(output_dtype),
    )
    if offload.verify:
        expected = pack_qsa_prefill_kv(
            k=pool_k,
            v=pool_v,
            req_to_token=offload.req_to_token,
            req_indices=forward_batch.req_pool_indices,
            cu_k=cu_seqlens_k,
            total_k=total_k,
            max_k=int(sequence_lens.max()),
            output_dtype=output_dtype,
        )
        offload.check(
            what="prefill context",
            layer_idx=layer_idx,
            actual=packed,
            expected=expected,
        )
    return packed


def forward_offloaded_attention(
    *,
    backend,
    q: torch.Tensor,
    layer,
    forward_batch,
    topk_indices: torch.Tensor,
    trtllm_decode,
) -> torch.Tensor:
    """Decode, target verify and draft extend over the offload's per-forward
    device-row table."""
    pool = backend.token_to_kv_pool
    offload = backend._kv_offload
    layer_idx = pool.full_attention_layer_index(layer.layer_id)
    metadata = backend._resolve_metadata(forward_batch)
    positions = backend._expand_block_indices(topk_indices, metadata)
    row_requests = backend._row_requests(forward_batch=forward_batch, metadata=metadata)
    num_rows, width = positions.shape
    rows_per_request = num_rows // forward_batch.batch_size
    rows = offload.resolve_rows(
        layer_idx=layer_idx,
        positions=positions,
        row_seq_lens=metadata.sequence_lengths,
        row_req=row_requests,
        rows_per_request=rows_per_request,
    )
    # Valid positions lead each row, so column ids keep the packing order.
    columns = torch.arange(width, dtype=torch.int32, device=rows.device)
    indices = torch.where(rows >= 0, columns, -1)
    pool_k = pool.layer_router.read(layer.layer_id).key()
    pool_v = pool.layer_router.read(layer.layer_id).value()
    source = QSAKVGatherSource(
        k_buffer=offload.dev_k[layer_idx].view(pool_k.dtype),
        v_buffer=offload.dev_v[layer_idx].view(pool_v.dtype),
        slot_table=rows,
        row_requests=torch.arange(num_rows, device=rows.device),
        sequence_lengths=torch.full(
            (num_rows,), width, dtype=torch.int32, device=rows.device
        ),
    )
    if offload.verify:
        reference = QSAKVGatherSource(
            k_buffer=pool_k,
            v_buffer=pool_v,
            slot_table=offload.req_to_token,
            row_requests=row_requests,
            sequence_lengths=metadata.sequence_lengths,
        )
        # CUDA-graph padding rows resolve to no device row by design.
        real_rows = offload.real_rows(
            num_rows=num_rows, rows_per_request=rows_per_request
        )
        reference_positions = torch.where(real_rows[:, None], positions, -1)
        offload.check(
            what="decode gather",
            layer_idx=layer_idx,
            actual=_pack_gathered_kv(source=source, indices=indices),
            expected=_pack_gathered_kv(
                source=reference,
                indices=reference_positions.to(torch.int32).contiguous(),
            ),
        )
    indices = indices.to(torch.int32).contiguous()
    attention = (
        backend._forward_varlen_sparse
        if trtllm_decode is None
        else partial(backend._forward_trtllm_sparse, trtllm_decode=trtllm_decode)
    )
    return attention(
        q=q,
        k_buffer=source.k_buffer,
        v_buffer=source.v_buffer,
        layer=layer,
        forward_batch=forward_batch,
        metadata=metadata,
        topk_indices=indices,
        source=source,
    )
