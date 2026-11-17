"""TP-shared, page-aligned MF storage for fused NPU sparse KV."""

from __future__ import annotations

import ctypes
import socket

import torch


class _NodeLocalGroup:
    """The members of one attention TP group sharing a host memory system."""

    def __init__(self, ranks, global_rank, parent):
        self.ranks = ranks
        self.world_size = len(ranks)
        self.rank_in_group = ranks.index(global_rank)
        self.cpu_group = self.device_group = None
        if self.world_size > 1:
            self.cpu_group = torch.distributed.new_group(
                ranks=ranks, backend="gloo", use_local_synchronization=True
            )
            self.device_group = torch.distributed.new_group(
                ranks=ranks,
                backend=torch.distributed.get_backend(parent.device_group),
                use_local_synchronization=True,
            )

    def broadcast_object(self, obj, src=0):
        if self.world_size > 1:
            values = [obj]
            torch.distributed.broadcast_object_list(
                values, src=self.ranks[src], group=self.cpu_group
            )
            return values[0]
        return obj

    def barrier(self):
        if self.world_size > 1:
            torch.distributed.barrier(group=self.cpu_group)

    def broadcast(self, tensor, src=0):
        if self.world_size > 1:
            torch.distributed.broadcast(
                tensor, src=self.ranks[src], group=self.device_group
            )
        return tensor


def _host_group(parent):
    if parent.world_size == 1:
        return parent

    from sglang.srt.distributed.parallel_state import in_the_same_node_as

    pending = set(range(parent.world_size))
    local_ranks = None
    while pending:
        source = min(pending)
        same_node = in_the_same_node_as(parent.cpu_group, source_rank=source)
        if not same_node[source]:
            raise RuntimeError("Failed to identify the local shared memory node")
        members = [i for i in sorted(pending) if same_node[i]]
        if len(members) == parent.world_size:
            return parent
        if parent.rank_in_group in members:
            local_ranks = [parent.ranks[i] for i in members]
        pending.difference_update(members)
    return _NodeLocalGroup(local_ranks, parent.ranks[parent.rank_in_group], parent)


class MemFabricKVPool:
    """Own the MF allocations; non-owner ranks only wrap broadcast GVAs.

    CKV and KPE are separate allocations. Every exported base is 2 MiB aligned
    so PD registration cannot include pages belonging to another allocation.
    """

    alignment = 2 * 1024 * 1024

    def __init__(
        self, *, tokens, layers, page_size, head_num, widths, dtype, device, group
    ):
        from memfabric_hybrid import offload

        self.offload = offload
        self.group = _host_group(group)
        self.rank = self.group.rank_in_group
        self._allocations = []
        sizes = [tokens * head_num * width * dtype.itemsize for width in widths]
        reserved = layers * sum(
            ((size + self.alignment - 1) // self.alignment + 1) * self.alignment
            for size in sizes
        )
        port = None
        if self.rank == 0:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
        port = self.group.broadcast_object(port)
        config = offload.OffloadConfig()
        config.device_id = torch.npu.current_device()
        config.reserve_size = reserved
        config.alloc_size = reserved if self.rank == 0 else 0
        config.world_size = self.group.world_size
        config.rank_id = self.rank
        config.scene = offload.Scene.SHARED
        config.store_url = f"tcp://127.0.0.1:{port}"
        result = offload.initialize(config)
        if result != 0:
            raise RuntimeError(f"Sparse KV MF initialization failed: {result}")
        self.group.barrier()
        self.buffers = []
        for width, size in zip(widths, sizes):
            component = []
            for _ in range(layers):
                if self.rank == 0:
                    chunks = (size + self.alignment - 1) // self.alignment
                    raw = offload.empty(
                        [(chunks + 1) * self.alignment],
                        dtype=torch.int8,
                        pin_memory=True,
                    )
                    self._allocations.append(raw)
                    offset = (-raw.data_ptr()) % self.alignment
                    tensor = (
                        raw[offset : offset + size]
                        .view(dtype)
                        .view(-1, page_size, head_num, width)
                    )
                    self.group.broadcast_object(tensor.data_ptr())
                else:
                    address = self.group.broadcast_object(None)
                    # MF SHARED provides the same GVA on every participant.
                    storage = (ctypes.c_uint8 * size).from_address(address)
                    tensor = torch.frombuffer(storage, dtype=dtype).view(
                        -1, page_size, head_num, width
                    )
                component.append(tensor)
            self.buffers.append(component)
        self._publish_fence = torch.zeros(1, dtype=torch.int32, device=device)

    def publish(self):
        # A device collective establishes stream ordering after TP0's D2H,
        # including graph replay. A CPU barrier would not order device writes.
        self.group.broadcast(self._publish_fence, src=0)

    def copy_rows(self, sources, destinations, source_rows, destination_rows, valid):
        """Enqueue byte-addressed MF sparse copies on the current NPU stream."""
        if source_rows.numel() == 0:
            return
        from sgl_kernel_npu.sparsity_driven_kv_offload.row_metadata import (
            copy_rows_metadata,
        )

        src, dst, lengths, count = copy_rows_metadata(
            sources, destinations, source_rows, destination_rows, valid
        )
        result = self.offload.sparse_copy(src, dst, lengths, count, source_rows.device)
        if result not in (None, 0):
            raise RuntimeError(f"Sparse KV MF sparse_copy failed: {result}")
