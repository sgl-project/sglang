"""Initialization and lifetime of single-node Engram peer mappings."""

import torch
import torch.distributed as dist

from sglang.kernels.ops.embeddings.engram_peer_gather import engram_peer_gather


class EngramPeerTable:
    def __init__(self, ranges, dim, block_size, group):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.dim = dim
        self.block_size = block_size
        rows = ranges[group.rank_in_group][1] - ranges[group.rank_in_group][0]
        # Rendezvous requires equal allocation sizes, including uneven shards.
        capacity = max(1, max(end - start for start, end in ranges))
        self.scale_offset = capacity * dim
        self.storage = symm_mem.empty(
            (capacity * (dim + dim // block_size),),
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.uint8,
        )
        self.weight = (
            self.storage[: self.scale_offset]
            .view(torch.float8_e4m3fn)
            .view(capacity, dim)[:rows]
        )
        self.scale = (
            self.storage[self.scale_offset :]
            .view(torch.float8_e8m0fnu)
            .view(capacity, dim // block_size)[:rows]
        )
        self.bounds = torch.tensor(
            [start for start, _ in ranges] + [ranges[-1][1]],
            device=self.storage.device,
            dtype=torch.int64,
        )
        self.handle = None

    def finish_load(self):
        import torch.distributed._symmetric_memory as symm_mem

        # Also covers in-place weight reloads: all writes complete before any
        # rank starts reading a peer's immutable table.
        torch.cuda.synchronize(self.storage.device)
        dist.barrier(group=self.group.cpu_group)
        if self.handle is None:
            self.handle = symm_mem.rendezvous(self.storage, self.group.device_group)
            self.weight_ptrs = torch.tensor(
                self.handle.buffer_ptrs,
                device=self.storage.device,
                dtype=torch.int64,
            )
            self.scale_ptrs = self.weight_ptrs + self.scale_offset

    def lookup(self, ids, output):
        assert self.handle is not None, "Engram peer table must finish loading first"
        return engram_peer_gather(
            self.weight_ptrs,
            self.scale_ptrs,
            self.bounds,
            ids,
            output,
            self.block_size,
        )


def peer_table_supported(group) -> bool:
    """Use the existing topology/P2P check; other layouts keep the normal lookup."""
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except ImportError:
        return False

    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        can_use_custom_all_reduce_with_nvlink,
    )
    from sglang.srt.runtime_context import get_exec, get_platform

    execution = get_exec()
    if (
        not get_platform().is_sm100
        or torch.empty(0).device.type != "cuda"
        or group.world_size not in (2, 4, 8)
        or not hasattr(symm_mem, "empty")
        or not hasattr(symm_mem, "rendezvous")
        or execution.features.enable_memory_saver
        or execution.features.enable_weights_cpu_backup
        or execution.offload.cpu_offload_gb > 0
        or execution.offload.offload_group_size > 0
    ):
        return False
    return (
        can_use_custom_all_reduce_with_nvlink(
            group=group.cpu_group,
            device=torch.device("cuda", torch.cuda.current_device()),
            supported_world_size=[2, 4, 8],
            cls_name="EngramPeerTable",
        )
        is True
    )
