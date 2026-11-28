"""Dynamic guest experts of the fused W4A16 MoE (EP 8 x 32, use_fused_w4a16): SP batches from DG_MIN tokens pull
the experts of busy ranks into staging slots over CUDA IPC."""

import ctypes
import functools
import logging
from typing import Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_EP_SIZE = 8


def maybe_prepare_dynamic_guests(layers: Sequence[torch.nn.Module]) -> None:
    from sglang.srt.runtime_context import get_parallel

    layers = [m for m in layers if m.use_fused_w4a16]
    group = get_parallel().tp_group
    if (
        not layers
        or group.world_size != _EP_SIZE
        or layers[0].global_num_experts != _EP_SIZE * layers[0].num_local_experts
        or not tp_group_on_one_host(group)
    ):
        return
    if "expandable_segments:True" in torch._C._accelerator_getAllocatorSettings():
        logger.warning(
            "MoE dynamic guests off: CUDA IPC cannot map expandable segments"
        )
        return
    if not _all_peers_accessible(group):
        logger.warning(
            "MoE dynamic guests off: some GPU pair of the TP group has no peer access"
        )
        return
    setup = _Setup(layers, group.cpu_group, group.rank_in_group)
    for m in layers:
        m.dynamic_guest_setup = setup


@functools.cache
def tp_group_on_one_host(group) -> bool:
    """Whether CUDA IPC handles and symmetric memory can map every rank of ``group``.

    They open only on the host that created them, so a TP group split over hosts
    (e.g. TP8 on two 4-GPU nodes) keeps the NCCL paths. A collective on the first
    call: every rank must reach it in the same order.
    """
    from sglang.srt.distributed.parallel_state import in_the_same_node_as
    from sglang.srt.runtime_context import get_parallel

    if group.world_size == 1 or get_parallel().nnodes == 1:
        return True
    return all(in_the_same_node_as(group.cpu_group, source_rank=0))


def _all_peers_accessible(group) -> bool:
    """Whether every rank of ``group`` can access every peer GPU (CUDA IPC maps peer
    memory only then), agreed over the group so that all ranks pull guests or none.

    Peers are matched by GPU UUID: with SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS every
    rank is cuda:0 and sees only its own GPU, so a peer it cannot see cannot be
    checked and counts as inaccessible. Under --enable-p2p-check the cached P2P
    probe must pass too. A collective: every rank must reach it in the same order.
    """
    from sglang.srt.distributed.device_communicators import custom_all_reduce_utils

    me = torch.cuda.current_device()
    visible = {_gpu_uuid(i): i for i in range(torch.cuda.device_count())}
    mine = _gpu_uuid(me)
    peers = [None] * group.world_size
    dist.all_gather_object(peers, mine, group=group.cpu_group)
    ok = all(
        uuid == mine
        or (
            uuid in visible
            and torch.cuda.can_device_access_peer(me, visible[uuid])
            # True unless --enable-p2p-check (bootstrap patches it otherwise)
            and custom_all_reduce_utils.gpu_p2p_access_check(me, visible[uuid])
        )
        for uuid in peers
    )
    agreed = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group.cpu_group)
    return bool(agreed.item())


def _gpu_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


class _Setup:
    def __init__(self, layers: Sequence[torch.nn.Module], cpu_group, rank: int):
        self._layers, self._cpu_group, self._rank, self._pending = (
            layers,
            cpu_group,
            rank,
            len(layers),
        )

    def on_repacked(self) -> None:
        self._pending -= 1
        if self._pending:
            return
        from sglang.kernels.ops.moe.muse_spark_nvfp4_w4a16_moe import (
            DYNAMIC_GUESTS,
            DynamicGuests,
        )
        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            CudaRTLibrary,
        )

        torch.cuda.synchronize()
        lib = CudaRTLibrary()
        lib.cudaSetDevice(torch.cuda.current_device())
        table, opened = _peer_weight_table(
            self._layers, self._cpu_group, self._rank, lib
        )
        w13, w2, g = (
            self._layers[0].w13_weight,
            self._layers[0].w2_weight,
            DYNAMIC_GUESTS,
        )
        stage = (
            w13.new_empty((g,) + w13.shape[1:]),
            w2.new_empty((g,) + w2.shape[1:]),
            w13.new_empty(2 * g, dtype=torch.float32),
        )
        gsrc = w13.new_full((g,), -1, dtype=torch.int32)
        table = table.to(w13.device)
        for li, m in enumerate(self._layers):
            m.dynamic_guests = DynamicGuests(stage, gsrc, table[li])
        logger.info(
            "MoE dynamic guests: %d layers, %d peer segments mapped",
            len(self._layers),
            opened,
        )


def _ipc_handle(t: torch.Tensor, lib) -> Tuple[bytes, int]:
    base, size = ctypes.c_uint64(), ctypes.c_size_t()
    rc = ctypes.CDLL("libcuda.so.1").cuMemGetAddressRange_v2(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(t.data_ptr())
    )
    if rc != 0:
        raise RuntimeError(f"cuMemGetAddressRange failed ({rc})")
    return bytes(
        lib.cudaIpcGetMemHandle(ctypes.c_void_p(base.value)).internal
    ), t.data_ptr() - base.value


def _peer_weight_table(
    layers: Sequence[torch.nn.Module], cpu_group, rank: int, lib
) -> Tuple[torch.Tensor, int]:
    """[layers, ranks, 4] int64 {w13, w2, gs13, gs2} addresses, peers' opened here."""
    from sglang.srt.distributed.device_communicators.cuda_wrapper import (
        cudaIpcMemHandle_t,
    )

    tensors = [
        (m.w13_weight, m.w2_weight, m.w13_weight_scale_2, m.w2_weight_scale_2)
        for m in layers
    ]
    everyone = [None] * _EP_SIZE
    dist.all_gather_object(
        everyone, [[_ipc_handle(t, lib) for t in ts] for ts in tensors], group=cpu_group
    )
    opened, table = {}, torch.zeros((len(layers), _EP_SIZE, 4), dtype=torch.int64)
    for p, rows in enumerate(everyone):
        for li, row in enumerate(rows):
            for k, (handle, offset) in enumerate(row):
                if p != rank and handle not in opened:
                    h = cudaIpcMemHandle_t()
                    ctypes.memmove(ctypes.addressof(h), handle, len(handle))
                    opened[handle] = lib.cudaIpcOpenMemHandle(h).value
                table[li, p, k] = (
                    tensors[li][k].data_ptr() if p == rank else opened[handle] + offset
                )
    return table, len(opened)
