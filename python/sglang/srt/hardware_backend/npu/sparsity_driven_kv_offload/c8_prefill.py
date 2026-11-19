"""Packed prefill attention for colocated A5 serving."""

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8_manager import (
    _copy_packed,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    C8_ROW_BYTES,
)


def forward_packed_prefill(backend, q, q_rope, layer, batch, topk_indices):
    manager = backend.sparse_kv_manager
    page = manager.page_size
    requests = batch.req_pool_indices.long()
    lengths = batch.seq_lens.to(torch.int32)
    lengths_cpu = batch.seq_lens_cpu.tolist()
    page_counts = [(length + page - 1) // page for length in lengths_cpu]
    total_pages = sum(page_counts)
    counts = torch.tensor(page_counts, dtype=torch.int64, device=q.device)
    starts = counts.cumsum(0) - counts
    row_sizes = counts * page
    packed = torch.zeros(
        (max(1, total_pages), page, 1, C8_ROW_BYTES), dtype=torch.int8, device=q.device
    )
    destinations = torch.arange(total_pages * page, dtype=torch.int64, device=q.device)
    offsets = torch.repeat_interleave(
        starts * page, row_sizes, output_size=total_pages * page
    )
    positions = destinations - offsets
    rows = torch.repeat_interleave(requests, row_sizes, output_size=total_pages * page)
    limits = torch.repeat_interleave(lengths, row_sizes, output_size=total_pages * page)
    if total_pages:
        _copy_packed(
            manager.host_kv_buffer[layer.layer_id - manager.start_layer],
            packed,
            manager.host_source_indices(rows, positions),
            destinations,
            positions < limits,
        )
    pages = torch.arange(max(1, max(page_counts, default=0)), device=q.device)
    table = torch.where(pages[None, :] < counts[:, None], starts[:, None] + pages, 0)
    heads = layer.tp_q_head_num
    query = torch.cat(
        (q.reshape(-1, heads, 512), q_rope.reshape(-1, heads, 64)), dim=-1
    )
    padded_heads = manager.budget.copy_sfa_query_heads
    if heads < padded_heads:
        query = torch.nn.functional.pad(query, (0, 0, 0, padded_heads - heads))
    ends = backend.forward_metadata.actual_seq_lengths_q
    if ends is None:
        ends = batch.extend_seq_lens.cumsum(0)
    cache = packed.view(torch.float8_e4m3fn)
    result = torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query.contiguous(),
        key=cache,
        value=cache,
        sparse_indices=topk_indices.reshape(-1, 1, manager.sparse_topk),
        scale_value=layer.scaling,
        key_quant_mode=2,
        value_quant_mode=2,
        key_dequant_scale=None,
        value_dequant_scale=None,
        actual_seq_lengths_query=ends.to(device=q.device, dtype=torch.int32),
        actual_seq_lengths_kv=lengths,
        block_table=table.to(torch.int32),
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    out = result[0] if isinstance(result, tuple) else result
    return out[:, :heads].contiguous()
