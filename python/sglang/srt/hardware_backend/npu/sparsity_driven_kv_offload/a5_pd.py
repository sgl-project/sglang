"""Move and retract packed C8 payload alongside the rank-local indexer cache."""

import torch

from .fused_budget import C8_ROW_BYTES


def move_packed_kv(pool, target, source):
    """Read all sources for a layer before writing overlapping destinations."""
    if source.numel() == 0:
        return
    host_pool = pool.pd_host_pool
    if host_pool.rank == 0:
        rows = torch.arange(source.numel(), device=pool.device)
        valid = torch.ones_like(rows, dtype=torch.bool)
        scratch = torch.empty(
            (source.numel(), 1, C8_ROW_BYTES), dtype=torch.int8, device=pool.device
        )
        for host in host_pool.buffers[0]:
            host_pool.copy_rows((host,), (scratch,), source, rows, valid)
            host_pool.copy_rows((scratch,), (host,), rows, target, valid)
    host_pool.publish()
    for layer in range(pool.layer_num):
        for state in pool._get_index_cache_layer_buffers(layer):
            state[target] = state[source]


def snapshot_packed_kv(pool, indices):
    """Include the local index keys and scales only for layers that own them."""
    torch.npu.synchronize()
    host_pool = pool.pd_host_pool
    result = []
    for layer, host in enumerate(host_pool.buffers[0]):
        states = pool._get_index_cache_layer_buffers(layer)
        chunks = []
        for start in range(0, len(indices), pool.cpu_offloading_chunk_size):
            selected = indices[start : start + pool.cpu_offloading_chunk_size].to(
                pool.device, dtype=torch.int64
            )
            rows = torch.arange(len(selected), device=pool.device)
            packed = torch.empty(
                (len(selected), 1, C8_ROW_BYTES), dtype=torch.int8, device=pool.device
            )
            host_pool.copy_rows(
                (host,),
                (packed,),
                selected,
                rows,
                torch.ones_like(rows, dtype=torch.bool),
            )
            chunks.append([packed.cpu()] + [state[selected].cpu() for state in states])
        result.append(chunks)
    torch.npu.synchronize()
    return result


def restore_packed_kv(pool, snapshot, indices):
    """Publish the shared payload once and restore indexer state on each rank."""
    torch.npu.synchronize()
    host_pool = pool.pd_host_pool
    for layer, (host, chunks) in enumerate(
        zip(host_pool.buffers[0], snapshot, strict=True)
    ):
        states = pool._get_index_cache_layer_buffers(layer)
        for start in range(0, len(indices), pool.cpu_offloading_chunk_size):
            selected = indices[start : start + pool.cpu_offloading_chunk_size].to(
                pool.device, dtype=torch.int64
            )
            packed, *saved_states = chunks[start // pool.cpu_offloading_chunk_size]
            if len(saved_states) != len(states):
                raise ValueError(
                    "A5 retraction snapshot has unexpected indexer components"
                )
            if host_pool.rank == 0:
                rows = torch.arange(len(selected), device=pool.device)
                host_pool.copy_rows(
                    (packed.to(pool.device),),
                    (host,),
                    rows,
                    selected,
                    torch.ones_like(rows, dtype=torch.bool),
                )
            for state, saved in zip(states, saved_states, strict=True):
                state[selected] = saved.to(state.device)
    host_pool.publish()
    torch.npu.synchronize()
