# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared assets for all-reduce host wrappers."""

import ctypes
import logging
from pathlib import Path

import torch
import torch.distributed as dist

from .allreduce_policy import resolve_link
from .kernels.comm.allreduce_shared import (
    MAX_PAYLOAD_BYTES,
    PAYLOAD_ALIGN_BYTES,
    SUPPORTED_ARCHS,
    SUPPORTED_WORLDS,
)
from .kernels.tensor_shim import _preload_compiled
from .quick_allreduce_int4_ipc import UncachedIpcHeap

logger = logging.getLogger("aiter")

# How the IPC inbox is allocated. The wire protocol is identical in every
# mode; only the memory type changes.
INBOX_MEMORY_MODES = ("auto", "uncached", "finegrained")


def _cuda_index(device, *, owner: str) -> int:
    """The CUDA device index of *device*. *owner* names the caller in errors."""
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"{owner} requires a CUDA device, got {device}")
        if device.index is None:
            return int(torch.cuda.current_device())
        return int(device.index)
    return int(device)


def _resolve_inbox_flags(
    mode: str, world_size: int, link: str | None = None
) -> tuple[int, str]:
    """(hipExtMallocWithFlags mode, resolved name) for an ``inbox_memory``.

    ``"auto"`` is ``uncached`` on xGMI and ``finegrained`` on PCIe, except at
    TP2, where it is ``uncached`` on PCIe too. The PCIe rule exists because
    uncached peer writes serialize per destination and collapse as the fanout
    widens. At TP2 every schedule writes to a single remote peer,
    so there is nothing to collapse, and an uncached inbox skips the L2 writeback.
    """
    if mode not in INBOX_MEMORY_MODES:
        raise ValueError(
            f"inbox_memory must be one of {INBOX_MEMORY_MODES}, got {mode!r}"
        )
    if mode == "auto":
        single_peer = int(world_size) == 2
        mode = (
            "uncached" if single_peer or resolve_link(link) == "xgmi" else "finegrained"
        )
    flags = {
        "uncached": UncachedIpcHeap._HIP_DEVICE_MALLOC_UNCACHED,
        "finegrained": UncachedIpcHeap._HIP_DEVICE_MALLOC_FINEGRAINED,
    }[mode]
    return flags, mode


class _StEngine:
    """One compile-time SUPER inbox + launch."""

    def __init__(
        self,
        *,
        spec,
        group,
        rank: int,
        world_size: int,
        inbox_flags: int,
    ):
        self.spec = spec
        self.launch = spec["launch"]
        self.super_tile = spec["super_tile"]
        self.grid = spec["grid"]
        self.buf_bytes = spec["flags_bytes"] + spec["data_bytes"]
        self.lds_bytes = spec["lds_bytes"]
        self.tile_bytes = spec["tile_bytes"]
        self.tile_fp16 = spec["tile_fp16"]
        self.rank_tile_bytes = spec["rank_tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self.block = spec["block"]
        self._peer_bases = [None] * world_size
        self._buf_ptr = None
        self._meta_ptr = None
        self._gpu_peer_ptrs = None
        self._colors = None
        # Setup has two collectives, the handle exchange and a final agreement.
        # A rank that fails must still take part in both, and report the
        # failure through them. If it raised at once, its peers would wait in a
        # collective that it never joins, and they would hang. The error is
        # raised on every rank after the agreement.
        error = None
        try:
            # The inbox is the only allocation peers write into, so it is the
            # only one whose memory type matters for fabric throughput.
            self._buf_ptr = UncachedIpcHeap.alloc(
                self.buf_bytes, inbox_flags, fill=spec.get("inbox_fill", 0)
            )
            my_meta = (UncachedIpcHeap.get_mem_handle_bytes(self._buf_ptr), 0)
        except Exception as e:  # noqa: BLE001
            error = e
            my_meta = (None, 0)
        try:
            all_meta = UncachedIpcHeap.gather_object_list_via_broadcast(group, my_meta)
        except Exception:
            self.close()
            raise

        if error is None:
            try:
                if any(handle is None for handle, _off in all_meta):
                    raise RuntimeError(
                        "a peer rank failed to allocate its inbox, which has "
                        f"{self.buf_bytes} B"
                    )
                self._open_peers(all_meta, rank=rank, world_size=world_size)
            except Exception as e:  # noqa: BLE001
                error = e

        ok = torch.tensor([0 if error is not None else 1], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if error is not None or not ok.item():
            self.close()
            raise error or RuntimeError("a peer rank failed to set up its engine")

    def _open_peers(self, all_meta, *, rank: int, world_size: int):
        """Open the peers' inboxes and upload the pointer table and the colours."""
        peer_ptrs = [0] * world_size
        for r in range(world_size):
            handle, off = all_meta[r]
            if r == rank:
                peer_ptrs[r] = self._buf_ptr + off
            else:
                base = int(UncachedIpcHeap.open_mem_handle(bytes(handle)))
                self._peer_bases[r] = base
                peer_ptrs[r] = base + off

        peer_bytes = world_size * 8
        color_bytes = self.grid * 4
        # Peer-pointer table and per-block colours: written by the host once
        # and by this rank's own kernel, never by a peer. Stays uncached in
        # every mode -- no cross-GPU visibility question, and it is a few
        # KiB.
        self._meta_ptr = UncachedIpcHeap.alloc_uncached(peer_bytes + color_bytes)
        self._gpu_peer_ptrs = self._meta_ptr
        self._colors = self._meta_ptr + peer_bytes
        UncachedIpcHeap.copy_host_to_device(
            self._gpu_peer_ptrs,
            (ctypes.c_int64 * world_size)(*peer_ptrs),
            peer_bytes,
        )
        UncachedIpcHeap.copy_host_to_device(
            self._colors,
            (ctypes.c_int32 * self.grid)(*([1] * self.grid)),
            color_bytes,
        )

    def close(self):
        for b in self._peer_bases:
            if b is not None:
                try:
                    UncachedIpcHeap.close_mem_handle(int(b))
                except RuntimeError:
                    pass
        self._peer_bases = []
        if self._meta_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._meta_ptr)
            except RuntimeError:
                pass
            self._meta_ptr = None
            self._gpu_peer_ptrs = None
            self._colors = None
        if self._buf_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._buf_ptr)
            except RuntimeError:
                pass
            self._buf_ptr = None


def require_arch(*, owner: str) -> str:
    """This device's ``gfx`` arch. Raises if the kernels do not support it."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    arch = get_gfx_runtime()
    if arch not in SUPPORTED_ARCHS:
        raise RuntimeError(f"{owner} supports {', '.join(SUPPORTED_ARCHS)}, got {arch}")
    return arch


def check_world_size(world_size: int) -> None:
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )


def check_tensors(inp, out, *, device_index: int, owner: str) -> tuple[int, int]:
    """The data pointers of *inp* and *out*.

    Raises unless both are contiguous, aligned bf16 CUDA tensors on
    *device_index*. Their sizes are the caller's to check.
    """
    align = PAYLOAD_ALIGN_BYTES
    if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
        raise TypeError(f"{owner} requires torch.Tensor input/output")
    if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError(f"{owner} supports bf16 input/output")
    if not inp.is_cuda or not out.is_cuda:
        raise ValueError(f"{owner} requires CUDA tensors")
    if inp.device.index != device_index or out.device.index != device_index:
        raise ValueError(
            f"inp/out must be on cuda:{device_index}, got {inp.device} / {out.device}"
        )
    if not inp.is_contiguous() or not out.is_contiguous():
        raise ValueError(f"{owner} requires contiguous input/output")
    inp_ptr = int(inp.data_ptr())
    out_ptr = int(out.data_ptr())
    if inp_ptr % align != 0 or out_ptr % align != 0:
        raise ValueError(f"{owner} requires {align}-byte-aligned input/output")
    return inp_ptr, out_ptr


def check_payload(inp, out, *, device_index: int, owner: str) -> int:
    """The payload size of an all-reduce of *inp* into *out*, in bytes.

    Raises unless both are contiguous bf16 CUDA tensors on *device_index*, of
    the same size, aligned, not overlapping, and within ``MAX_PAYLOAD_BYTES``.
    """
    align = PAYLOAD_ALIGN_BYTES
    inp_ptr, out_ptr = check_tensors(inp, out, device_index=device_index, owner=owner)
    live_bytes = int(inp.numel()) * int(inp.element_size())
    if live_bytes > MAX_PAYLOAD_BYTES:
        raise ValueError(f"{owner} payload must not exceed the 4 GiB buffer window")
    if live_bytes % align != 0:
        raise ValueError(f"byte size must be a multiple of {align} ({align // 2} bf16)")
    if int(out.numel()) * int(out.element_size()) != live_bytes:
        raise ValueError("inp/out byte size mismatch")
    if max(inp_ptr, out_ptr) < min(inp_ptr + live_bytes, out_ptr + live_bytes):
        raise ValueError(f"{owner} requires non-overlapping input/output")
    return live_bytes


def _validate_ipc_process_group(
    group, *, rank: int, world_size: int, owner: str
) -> None:
    """Reject groups that do not match *rank* and *world_size*, or that cannot
    exchange HIP IPC handles or CPU-side metadata."""
    # Keep parallel_state lazy: this module is imported while aiter's AOT setup
    # is still initializing the top-level package.
    from aiter.dist.parallel_state import in_the_same_node_as

    group_world = dist.get_world_size(group=group)
    group_rank = dist.get_rank(group=group)
    if group_world != int(world_size):
        raise ValueError(
            f"world_size={world_size} does not match group size {group_world}"
        )
    if group_rank != int(rank):
        raise ValueError(f"rank={rank} does not match group rank {group_rank}")

    backend = dist.get_backend(group)
    if backend == dist.Backend.NCCL:
        raise ValueError(
            f"{owner} does not support NCCL process groups (got "
            f"{backend!r} on group rank {rank}): IPC handle exchange requires "
            "CPU-side broadcast_object_list."
        )

    same_node = in_the_same_node_as(group, source_rank=0)
    if not all(same_node):
        off_node = [r for r, ok in enumerate(same_node) if not ok]
        raise RuntimeError(
            f"{owner} does not support multi-node process groups: HIP "
            f"IPC handles are node-local (ranks not on rank 0's node: {off_node})."
        )


# KFD io-link type for xGMI, from include/uapi/linux/kfd_sysfs.h. PCIe is 2.
_HSA_IOLINK_TYPE_XGMI = 11
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def probe_xgmi_peer_links() -> bool | None:
    """Whether any GPU-to-GPU link on this host is xGMI rather than PCIe, from
    the KFD topology. None when the topology cannot be read.

    Arch is not enough to make this call: an MI350X (xGMI) and an MI350P
    (PCIe-only) both report ``gfx950``, and the right inbox memory type is
    opposite on the two. KFD exposes the real link type per peer pair, so read
    that instead of guessing from the SKU.

    Both ``io_links`` and ``p2p_links`` have to be scanned. KFD only populates
    ``p2p_links`` for peers reachable indirectly (through a host bridge), so on
    a directly-connected mesh it holds nothing but the PCIe links to the CPU
    nodes and the xGMI peers appear solely under ``io_links``. Reading
    ``p2p_links`` alone reports "PCIe" on an 8-GPU all-xGMI MI350X, which flips
    ``inbox_memory="auto"`` to the fine-grained heap and silently costs the
    uncached fanout the kernel was designed around.

    The function returns None, not False, when it finds no link ``properties``
    file at all or cannot read one.

    The answer is per host, not per process group: one xGMI link anywhere
    classifies every group on the node as xGMI. That assumes a node is uniformly
    xGMI or uniformly PCIe, which holds for the single-node systems this targets.
    """
    try:
        found = False
        for subdir in ("io_links", "p2p_links"):
            for props in _KFD_NODES.glob(f"*/{subdir}/*/properties"):
                found = True
                for line in props.read_text().splitlines():
                    field, _, value = line.partition(" ")
                    if field == "type" and int(value) == _HSA_IOLINK_TYPE_XGMI:
                        return True
        return False if found else None
    except (OSError, ValueError):
        logger.debug("FlyDSL all-reduce: cannot read KFD topology", exc_info=True)
        return None


def kernel_symbol(launch) -> str:
    """The JIT symbol a kernel factory stamped on its launch wrapper."""
    name = getattr(getattr(launch, "func", None), "__name__", None)
    if not name:
        return "?"
    return name.removeprefix("launch_")


def _launch_artifacts(launch) -> list:
    """The compiled artifacts of a ``@flyc.jit`` launch in this process."""
    cache = getattr(launch, "_mem_cache", None)
    if not isinstance(cache, dict):
        raise TypeError(
            f"cannot reach the compiled artifacts of {kernel_symbol(launch)}: "
            "this FlyDSL version has no JitFunction._mem_cache"
        )
    return list(cache.values())


def preload_launcher(launch, *args) -> None:
    """Compile launch for args and load it onto the current device,
    without launching it.

    The module loads into the current device, so the caller must make the
    engine's device current.
    """
    _preload_compiled(launch, *args)
    artifacts = _launch_artifacts(launch)
    if not artifacts:
        raise RuntimeError(f"FlyDSL compiled no artifact for {kernel_symbol(launch)}")
    for artifact in artifacts:
        # Builds the ExecutionEngine, which loads the module, on first use.
        artifact._get_func_exe()


def launcher_loaded(launch) -> bool:
    """Whether every compiled artifact of *launch* is loaded, so that its
    next launch does no JIT and no module load. False if none is compiled."""
    artifacts = _launch_artifacts(launch)
    return bool(artifacts) and all(a._engine is not None for a in artifacts)


def payload_probes(floors, lo: int, hi: int) -> tuple[int, ...]:
    """Sample payload sizes that select all the configs a ladder can use for a
    payload of ``lo..hi`` bytes (inclusive).

    *floors* are the ``min_bytes`` values of the ladder rungs. The function
    returns ``()`` if no payload size in the range is valid.
    """
    step = 16
    lo = max(step, -(-int(lo) // step) * step)
    hi = int(hi) // step * step
    if lo > hi:
        return ()
    probes = {lo, hi}
    for floor in floors:
        first = -(-int(floor) // step) * step
        if lo < first <= hi:
            probes.update((first - step, first))
    return tuple(sorted(probes))
