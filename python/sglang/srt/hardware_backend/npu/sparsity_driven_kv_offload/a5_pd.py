"""Opaque packed-C8 relocation and retraction in native physical token space."""

import torch


def _copy_host_rows(
    manager,
    layer,
    source,
    destination,
    source_rows,
    destination_rows,
    valid,
    *,
    to_host=False,
):
    mf = manager.mf_host_pool
    if mf is not None:
        return mf.copy_rows(
            [source], [destination], source_rows, destination_rows, valid
        )
    from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace

    return unidex_copy_inplace(
        source.view(-1, 656),
        destination.view(-1, 656),
        source_rows.contiguous(),
        destination_rows.contiguous(),
        valid.contiguous(),
        1,
        1,
        block_dim=48,
        src_ptr=None if to_host else manager.dev_ptr_list[layer],
        dst_ptr=manager.dev_ptr_list[layer] if to_host else None,
    )


def _state_buffers(pool, local_layer):
    slot = pool.indexer_layer_id_to_slot.get(local_layer + pool.start_layer)
    if slot is None:
        return []
    return [
        pool.index_k_buffer[slot].flatten(0, 1).view(torch.uint8),
        pool.index_k_scale_buffer[slot].flatten(0, 1),
    ]


def move_packed_kv(manager, target, source):
    """Snapshot all source rows before an overlapping accepted-token move."""
    if source.numel() == 0:
        return
    mf = manager.mf_host_pool
    if mf is None or mf.rank == 0:
        scratch = torch.empty(
            (source.numel(), 1, 656), dtype=torch.int8, device=manager.device
        )
        rows = torch.arange(source.numel(), device=manager.device)
        valid = torch.ones_like(rows, dtype=torch.bool)
        for layer, host in enumerate(manager.host_kv_buffer):
            _copy_host_rows(manager, layer, host, scratch, source, rows, valid)
            _copy_host_rows(
                manager, layer, scratch, host, rows, target, valid, to_host=True
            )
    if mf is not None:
        mf.publish()


def snapshot_packed_kv(pool, indices):
    """Back up registered Host payload plus rank-local LI state."""
    torch.npu.synchronize()
    manager = pool.sparse_kv_manager
    result = []
    for layer, host in enumerate(manager.host_kv_buffer):
        chunks = []
        states = _state_buffers(pool, layer)
        for start in range(0, len(indices), pool.cpu_offloading_chunk_size):
            selected = indices[start : start + pool.cpu_offloading_chunk_size].to(
                pool.device, dtype=torch.int64
            )
            rows = torch.arange(len(selected), device=pool.device)
            scratch = torch.empty(
                (len(selected), 1, 656), dtype=torch.int8, device=pool.device
            )
            _copy_host_rows(
                manager,
                layer,
                host,
                scratch,
                selected,
                rows,
                torch.ones_like(rows, dtype=torch.bool),
            )
            chunks.append(
                [scratch.cpu().clone()]
                + [state[selected].cpu().clone() for state in states]
            )
        result.append(chunks)
    torch.npu.synchronize()
    return result


def restore_packed_kv(pool, snapshot, indices, request_id):
    """Restore shared bytes once, local LI state on every rank, then reset LIM."""
    torch.npu.synchronize()
    manager = pool.sparse_kv_manager
    mf = manager.mf_host_pool
    for layer, host in enumerate(manager.host_kv_buffer):
        states = _state_buffers(pool, layer)
        for start in range(0, len(indices), pool.cpu_offloading_chunk_size):
            selected = indices[start : start + pool.cpu_offloading_chunk_size].to(
                pool.device, dtype=torch.int64
            )
            payload = snapshot[layer][start // pool.cpu_offloading_chunk_size]
            if len(payload) != len(states) + 1:
                raise ValueError("A5 MF retraction snapshot component mismatch")
            if mf is None or mf.rank == 0:
                rows = torch.arange(len(selected), device=pool.device)
                _copy_host_rows(
                    manager,
                    layer,
                    payload[0].to(pool.device),
                    host,
                    rows,
                    selected,
                    torch.ones_like(rows, dtype=torch.bool),
                    to_host=True,
                )
            for state, saved in zip(states, payload[1:]):
                state[selected] = saved.to(state.device)
    if mf is not None:
        mf.publish()
    if request_id is not None:
        manager.invalidate_packed_request(int(request_id))
    torch.npu.synchronize()
