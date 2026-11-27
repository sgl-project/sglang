import torch


def plan_index_k_prefill_gather(
    local_block_table: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    key_lens_cpu: torch.Tensor,
    page_size: int,
    dcp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan the page gather and the valid rows for this rank's prefill queries."""
    span = page_size * dcp_size
    cycles = (seq_lens_cpu + span - 1) // span
    cycle_ids = torch.arange(local_block_table.shape[1], device=seq_lens_cpu.device)
    device = local_block_table.device
    req_rows, req_cycles = (cycle_ids < cycles[:, None]).nonzero().to(device).unbind(1)
    pages = local_block_table[req_rows, req_cycles]

    gathered_starts = (cycles.cumsum(0) - cycles) * span
    packed_starts = key_lens_cpu.cumsum(0) - key_lens_cpu
    num_rows = int(key_lens_cpu.sum())
    rows = torch.arange(num_rows, device=key_lens_cpu.device) + torch.repeat_interleave(
        gathered_starts - packed_starts, key_lens_cpu, output_size=num_rows
    )
    return pages, rows.to(device=device)


def score_sharded_index_k(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    group,
) -> torch.Tensor:
    if query.shape[0] == 0 or block_table.shape[1] == 0:
        return query.new_full((query.shape[0], topk), -1, dtype=torch.int32)

    from sgl_kernel_npu.operators.indexer import lightning_indexer_v2

    # Decode and Verify both supply one KV frontier per query. Treat each
    # query as a separate request so a page boundary cannot change causality.
    indices, scores = lightning_indexer_v2(
        query.contiguous(),
        key,
        weights.reshape(query.shape[:-1]).float().contiguous(),
        topk,
        cu_seqlens_q=torch.arange(
            query.shape[0] + 1, dtype=torch.int32, device=query.device
        ),
        seqused_k=seq_lens,
        block_table=block_table.contiguous(),
        layout_q="TND",
        layout_k="PA_BBND",
        return_value=1,
    )
    indices, scores = indices.squeeze(1), scores.squeeze(1)
    page_size = key.shape[1]
    valid = (indices >= 0) & (indices < seq_lens[:, None])
    global_indices = (
        (indices // page_size * group.world_size + group.rank_in_group) * page_size
        + indices % page_size
    ).masked_fill_(~valid, -1)
    # One HCCL gather preserves the FP32 score bits and int32 positions.
    scores = scores.masked_fill(~valid, -torch.inf)
    candidates = group.all_gather(
        torch.stack((scores.view(torch.int32), global_indices), dim=1), dim=-1
    )
    best = candidates[:, 0].view(torch.float32).topk(topk, dim=-1).indices
    return candidates[:, 1].gather(-1, best)
