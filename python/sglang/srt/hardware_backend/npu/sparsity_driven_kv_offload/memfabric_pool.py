"""TP-shared, page-aligned MF storage for fused NPU sparse KV."""

from __future__ import annotations

import ctypes
import socket

import torch


class MemFabricKVPool:
    """Own the MF allocations; non-owner ranks only wrap broadcast GVAs.

    CKV and KPE are separate allocations. Every exported base is 2 MiB aligned
    so PD registration cannot include pages belonging to another allocation.
    """

    alignment = 2 * 1024 * 1024
    supports_pd_export = True

    def __init__(self, *, tokens, layers, widths, dtype, device, group):
        if tokens <= 0 or tokens % 128 or layers <= 0 or tuple(widths) != (512, 64):
            raise ValueError("MF split KV requires page-aligned 512/64 MLA geometry")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MF split KV requires FP16 or BF16 storage")
        from memfabric_hybrid import offload

        self.offload = offload
        self.group = group
        self.rank = self.group.rank_in_group
        self._allocations = []
        sizes = [tokens * width * dtype.itemsize for width in widths]
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
                tensor = None
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
                        raw[offset : offset + size].view(dtype).view(-1, 128, 1, width)
                    )
                address = self.group.broadcast_object(
                    tensor.data_ptr() if tensor is not None else None
                )
                if tensor is None:
                    # MF SHARED provides the same GVA on every participant.
                    storage = (ctypes.c_uint16 * (size // 2)).from_address(address)
                    tensor = torch.frombuffer(storage, dtype=dtype).view(
                        -1, 128, 1, width
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
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.row_metadata import (
            copy_rows_metadata,
        )

        src, dst, lengths, count = copy_rows_metadata(
            sources, destinations, source_rows, destination_rows, valid
        )
        result = self.offload.sparse_copy(src, dst, lengths, count, source_rows.device)
        if result not in (None, 0):
            raise RuntimeError(f"Sparse KV MF sparse_copy failed: {result}")
