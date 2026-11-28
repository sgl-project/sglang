# SPDX-License-Identifier: Apache-2.0
"""Paged BF16 GQA attention with an Ascend DCP group."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.npu.attention.dcp_metadata import (
    build_gqa_dcp_extend_layout,
)
from sglang.srt.hardware_backend.npu.dcp_extend_gather import (
    DCP_EXTEND_GATHER_PIECE_ROWS,
    DcpExtendGatherPlan,
    dcp_extend_gather_buffer,
    plan_dcp_extend_gather,
)
from sglang.srt.layers.dcp.comm import cp_lse_ag_out_rs_mla_npu
from sglang.srt.layers.dcp.layout import dcp_local_row
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class GQADCPMetadata:
    write_rows: torch.Tensor
    write_locs: torch.Tensor
    block_table: torch.Tensor | None = None
    local_lens: list[int] | None = None
    empty_shards: torch.Tensor | None = None
    extend_plan: DcpExtendGatherPlan | None = None
    prefix_rows: torch.Tensor | None = None
    extend_cu_lens: list[int] | None = None
    full_lens: list[int] | None = None
    extend_page_rows: torch.Tensor | None = None
    extend_capacity: int = 0


def prepare_gqa_dcp_metadata(
    backend: AscendAttnBackend, batch: ForwardBatch
) -> GQADCPMetadata:
    mode = batch.forward_mode
    cache_locs = batch.out_cache_loc
    if mode.is_extend():
        # MLP sync appends zero slots, but attention only sees the real rows.
        cache_locs = cache_locs[: sum(batch.extend_seq_lens_cpu)]
    # RunnerIds already localizes write slots and marks non-owners with -1.
    rows = (cache_locs >= 0).nonzero().flatten()
    meta = GQADCPMetadata(rows, cache_locs.index_select(0, rows))
    if mode.is_decode():
        lens = (
            batch.seq_lens_cpu
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.cpu()
        )
        local_lens, meta.block_table = backend._get_kv_lens_and_block_tables(
            lens, batch.req_pool_indices
        )
        meta.local_lens = local_lens.tolist()
        meta.empty_shards = (local_lens == 0).to(
            batch.seq_lens.device, non_blocking=True
        )
    elif mode.is_extend():
        prefix_lens = batch.extend_prefix_lens_cpu
        extend_lens = batch.extend_seq_lens_cpu
        device = batch.req_pool_indices.device
        page = backend.page_size
        meta.extend_cu_lens = list(accumulate(extend_lens))
        meta.full_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        block_table, rows, meta.extend_capacity = build_gqa_dcp_extend_layout(
            meta.full_lens, page
        )
        meta.block_table = block_table.to(device, non_blocking=True)
        meta.extend_page_rows = rows.to(device, non_blocking=True)
        if any(prefix_lens):
            plan = plan_dcp_extend_gather(
                prefix_lens,
                extend_lens,
                backend.dcp_size,
                backend.dcp_rank,
                DCP_EXTEND_GATHER_PIECE_ROWS,
                page,
            )
            meta.extend_plan = plan._replace(
                pieces=[
                    piece._replace(index=piece.index.to(batch.seq_lens.device))
                    for piece in plan.pieces
                ]
            )
            # Choose the owned positions in each request, then translate its
            # allocator slots into physical token indices in the rank's KV pool.
            parts = []
            for req, local_len in enumerate(plan.local_lens):
                local = torch.arange(local_len, dtype=torch.long)
                positions = (
                    local // page * backend.dcp_size + backend.dcp_rank
                ) * page + local % page
                parts.append(torch.stack((torch.full_like(positions, req), positions)))
            reqs, positions = torch.cat(parts, dim=1).to(device, non_blocking=True)
            slots = backend.req_to_token[batch.req_pool_indices[reqs], positions]
            meta.prefix_rows = dcp_local_row(slots, backend.dcp_size, page).long()
    return meta


def _gather_paged_extend_cache(cache, current, meta, name, page_size):
    """Rebuild an ephemeral paged KV table for the ordinary causal prefill kernel."""
    plan = meta.extend_plan
    out = dcp_extend_gather_buffer(name, current, meta.extend_capacity).zero_()
    if plan is None:
        out.index_copy_(0, meta.extend_page_rows, current)
        return out.view(-1, page_size, current.shape[1] * current.shape[2])
    group = get_parallel().dcp_group
    scratch = dcp_extend_gather_buffer(f"{name}_scratch", current, plan.scratch_rows)
    shards = cache.view(-1, *current.shape[1:]).index_select(0, meta.prefix_rows)
    send = shards
    if plan.local_lens != plan.padded_lens:
        send = shards.new_zeros((plan.send_rows, *shards.shape[1:]))
        src = dst = 0
        for local_len, padded_len in zip(plan.local_lens, plan.padded_lens):
            send[dst : dst + local_len] = shards[src : src + local_len]
            src += local_len
            dst += padded_len
    for piece in plan.pieces:
        send_rows = piece.send_end - piece.send_start
        gathered = send_rows * group.world_size
        if gathered:
            group.all_gather_into_tensor(
                scratch[:gathered], send[piece.send_start : piece.send_end]
            )
        scratch[gathered : piece.scratch_rows] = current[
            piece.extend_start : piece.extend_end
        ]
        out.index_copy_(
            0,
            meta.extend_page_rows[piece.out_start : piece.out_end],
            scratch[: piece.scratch_rows].index_select(0, piece.index),
        )
    return out.view(-1, page_size, current.shape[1] * current.shape[2])


def forward_gqa_dcp(
    backend: AscendAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    batch: ForwardBatch,
    save_kv_cache: bool,
) -> torch.Tensor:
    import torch_npu

    meta = backend.forward_metadata.gqa_dcp
    q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    k = k.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim)
    v = v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim)
    if save_kv_cache and meta.write_rows.numel():
        # NPU MHA storage does not accept the -1 non-owner slots.
        backend.token_to_kv_pool.set_kv_buffer(
            layer,
            meta.write_locs,
            k.index_select(0, meta.write_rows),
            v.index_select(0, meta.write_rows),
        )

    k_cache = backend.token_to_kv_pool.get_key_buffer(layer.layer_id)
    v_cache = backend.token_to_kv_pool.get_value_buffer(layer.layer_id)
    if batch.forward_mode.is_decode():
        group = get_parallel().dcp_group
        q_all = group.all_gather(q.contiguous(), dim=1).contiguous()
        if any(meta.local_lens):
            out, lse = torch_npu.npu_fused_infer_attention_score(
                q_all.unsqueeze(1),
                k_cache.view(
                    -1, backend.page_size, layer.tp_k_head_num * layer.qk_head_dim
                ),
                v_cache.view(
                    -1, backend.page_size, layer.tp_v_head_num * layer.v_head_dim
                ),
                num_heads=q_all.shape[1],
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSND",
                block_table=meta.block_table,
                block_size=backend.page_size,
                actual_seq_lengths_kv=meta.local_lens,
                scale=layer.scaling,
                softmax_lse_flag=True,
            )
            out = out.squeeze(1)
            lse = lse.reshape(q_all.shape[:2]).masked_fill(
                meta.empty_shards[:, None], -torch.inf
            )
        else:
            # Short prompts can live entirely on one rank. Every rank still
            # sends a partial state, with zero mass for an empty shard.
            out = q_all.new_zeros((*q_all.shape[:2], layer.v_head_dim))
            lse = q_all.new_full(q_all.shape[:2], -torch.inf, dtype=torch.float32)
        out = cp_lse_ag_out_rs_mla_npu(out, lse, group)
    else:
        k = _gather_paged_extend_cache(k_cache, k, meta, "gqa_key", backend.page_size)
        v = _gather_paged_extend_cache(v_cache, v, meta, "gqa_value", backend.page_size)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            block_table=meta.block_table,
            block_size=backend.page_size,
            atten_mask=backend.fia_mask,
            actual_seq_lengths=meta.extend_cu_lens,
            actual_seq_lengths_kv=meta.full_lens,
            scale=layer.scaling,
            sparse_mode=3,
        )
    return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
