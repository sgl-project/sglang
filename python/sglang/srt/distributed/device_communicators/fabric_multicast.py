"""FABRIC multicast registration for regular VMM tensors (NVLS zero-copy AR).

Binds a regular (``expandable_segments``/VMM) tensor into a CUDA multicast object so
the two-shot ``multimem.ld_reduce``/``.st`` all-reduce reduces it in place -- no
stage-in/out through the pull workspace. The multicast analog of
``VmmGraphInputManager``'s peer mapping (which ``cuMemMap``s a peer's chunk for the
unicast graph-pull path): rank 0 ``cuMulticastCreate``s the object + shares its
FABRIC handle, then every rank ``cuMulticastBindMem``s its own chunk and maps a
local multicast VA.

Requires VMM + FABRIC (one NVLS clique). The bind is a driver call, so run it
outside CUDA-graph capture (register the captured graph-input bases post-capture).
"""

from __future__ import annotations

from typing import Dict, Tuple

import cuda.bindings.driver as drv
import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.vmm_utils import (
    check_drv,
    is_vmm_pointer,
    make_rw_access_desc,
)

_FABRIC = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC


def _align_up(x: int, a: int) -> int:
    return -(-x // a) * a


def _addr_range(addr: int) -> Tuple[int, int]:
    """(base, size) of the VMM allocation containing ``addr``."""
    r = drv.cuMemGetAddressRange(addr)
    if r[0] != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemGetAddressRange(0x{addr:x}): {r[0]}")
    return int(r[1]), int(r[2])


class FabricMulticastRegistry:
    """Per-process registry: regular VMM tensor -> its NVLS multicast address.

    ``register(t)`` binds ``t``'s local allocation into a per-allocation multicast
    object shared across the clique and returns the multicast VA for ``t``. The
    result is stable for the allocation's lifetime and cached by ``data_ptr``
    (revalidated against the live allocation, since the caching allocator recycles
    pointers). Collective: all ranks must call ``register`` in the same order over
    ``group`` -- the custom-AR's non-NCCL, CPU-capable process group (same group the
    graph-input handle exchange uses).
    """

    def __init__(self, group) -> None:
        self.group = group
        self.rank = dist.get_rank(group=self.group)
        self.world = dist.get_world_size(group=self.group)
        self.device_id = torch.cuda.current_device()
        # data_ptr -> (mc_addr, mc_handle, va, bind_size, base, base_size)
        self._cache: Dict[int, Tuple] = {}

    def register(self, t: torch.Tensor) -> int:
        """Return the NVLS multicast address of ``t`` (cached). Collective."""
        return self.register_ptr(int(t.data_ptr()), t.numel() * t.element_size())

    def register_ptr(self, data_ptr: int, nbytes: int) -> int:
        """Return the NVLS multicast address of ``[data_ptr, data_ptr+nbytes)``
        (cached). Collective. Used by the post-capture hook, which only has the
        recorded pointer (not the tensor) of the graph's captured AR buffer.

        Raises on a non-VMM (cudaMalloc) pointer -- multicast binding needs a VMM
        allocation handle to bind. Must run outside CUDA-graph capture.
        """
        key = int(data_ptr)
        if not is_vmm_pointer(key):
            raise RuntimeError(
                "FABRIC multicast register requires a VMM (expandable_segments) "
                "pointer; got a cudaMalloc pointer with no allocation handle to bind."
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FabricMulticastRegistry.register must run outside capture"
            )
        base, base_size = _addr_range(key)
        offset = key - base
        cached = self._cache.get(key)
        if cached is not None:
            mc_addr, _mc, _va, _bs, c_base, c_bsize = cached
            if c_base == base and c_bsize == base_size:
                return mc_addr
            self._release_entry(cached)  # allocator recycled this data_ptr
        # A live activation (unlike a symm arena) isn't guaranteed to sit at the
        # same offset on every rank, which multicast reduction needs -- verify.
        self._assert_symmetric_offset(offset)
        entry = self._register_alloc(base, base_size, offset, int(nbytes))
        self._cache[key] = entry
        return entry[0]

    def _assert_symmetric_offset(self, offset: int) -> None:
        """Raise unless ``offset`` (activation position within its VMM allocation)
        is identical on every rank -- the precondition for an in-place multicast
        all-reduce to combine matching elements. Collective."""
        buf = torch.tensor([int(offset)], dtype=torch.int64)
        gathered = [torch.empty_like(buf) for _ in range(self.world)]
        dist.all_gather(gathered, buf, group=self.group)
        offs = [int(g.item()) for g in gathered]
        if any(o != offs[0] for o in offs):
            raise RuntimeError(
                "in-place multicast all-reduce requires a symmetric activation "
                f"offset across ranks, got {offs}; the per-rank allocation diverged "
                "(the staged path is offset-agnostic -- drop the registered path)."
            )

    def _register_alloc(
        self, base: int, base_size: int, offset: int, nbytes: int
    ) -> Tuple:
        """Bind ``[base, base+bind_size)`` on every rank into one multicast object,
        map a local multicast VA, and return the cache entry. A captured activation
        can span several expandable_segments chunks (each its own cuMemCreate
        handle at a contiguous VA); bind every chunk in the range at its matching
        multicast offset. Collective."""
        prop = drv.CUmulticastObjectProp()
        prop.numDevices = self.world
        prop.handleTypes = int(_FABRIC)
        prop.flags = 0
        prop.size = max(1, offset + nbytes)
        # MINIMUM (not RECOMMENDED): RECOMMENDED scales with the size and can be
        # ~512 MB, so aligning a small activation's bind up to it overflows its chunk.
        gran = int(
            check_drv(
                drv.cuMulticastGetGranularity(
                    prop,
                    drv.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_MINIMUM,
                ),
                "cuMulticastGetGranularity",
            )
        )
        bind_size = _align_up(offset + nbytes, gran)
        prop.size = bind_size

        # Rank 0 creates the object + shares its FABRIC handle; peers import.
        mc_handle = self._make_shared_mc(prop)
        check_drv(
            drv.cuMulticastAddDevice(mc_handle, self.device_id), "cuMulticastAddDevice"
        )
        # Bind each physical chunk covering [base, base+bind_size) at its offset from
        # base; symmetric offsets (verified above) make mcOffset k every rank's elem k.
        bound = 0
        while bound < bind_size:
            c_base, c_size = _addr_range(base + bound)
            n = min(c_size - (base + bound - c_base), bind_size - bound)
            mem_h = check_drv(
                drv.cuMemRetainAllocationHandle(c_base), "cuMemRetainAllocationHandle"
            )
            try:
                check_drv(
                    drv.cuMulticastBindMem(
                        mc_handle, bound, mem_h, base + bound - c_base, n, 0
                    ),
                    "cuMulticastBindMem",
                )
            finally:
                check_drv(drv.cuMemRelease(mem_h), "cuMemRelease(local mem)")
            bound += n

        # Reserve + map a local VA over the multicast object -> the multicast alias.
        va = int(
            check_drv(
                drv.cuMemAddressReserve(bind_size, gran, 0, 0),
                "cuMemAddressReserve(mc)",
            )
        )
        check_drv(drv.cuMemMap(va, bind_size, 0, mc_handle, 0), "cuMemMap(mc)")
        check_drv(
            drv.cuMemSetAccess(va, bind_size, [make_rw_access_desc(self.device_id)], 1),
            "cuMemSetAccess(mc)",
        )
        return (va + offset, mc_handle, va, bind_size, base, base_size)

    def _make_shared_mc(self, prop) -> int:
        """Create the multicast object on rank 0, share its FABRIC handle over the
        clique, and return this rank's handle to the SAME object."""
        handle_bytes = b""
        if self.rank == 0:
            mc = check_drv(drv.cuMulticastCreate(prop), "cuMulticastCreate")
            fabric_h = check_drv(
                drv.cuMemExportToShareableHandle(mc, _FABRIC, 0),
                "cuMemExportToShareableHandle(mc, FABRIC)",
            )
            handle_bytes = bytes(fabric_h.data)
        # all_gather the 64-byte handle; every rank imports rank 0's object.
        buf = torch.frombuffer(
            bytearray(handle_bytes.ljust(64, b"\0")), dtype=torch.uint8
        ).clone()
        gathered = [torch.empty_like(buf) for _ in range(self.world)]
        dist.all_gather(gathered, buf, group=self.group)
        if self.rank == 0:
            return mc
        return check_drv(
            drv.cuMemImportFromShareableHandle(
                bytes(gathered[0].numpy().tobytes()), _FABRIC
            ),
            "cuMemImportFromShareableHandle(mc)",
        )

    def _release_entry(self, entry: Tuple) -> None:
        _mc_addr, mc_handle, va, bind_size, _base, _bsize = entry
        check_drv(drv.cuMemUnmap(va, bind_size), "cuMemUnmap(mc)")
        check_drv(drv.cuMemAddressFree(va, bind_size), "cuMemAddressFree(mc)")
        check_drv(drv.cuMemRelease(mc_handle), "cuMemRelease(mc)")

    def release(self) -> None:
        """Unmap + free all multicast VAs and drop the cache."""
        for entry in self._cache.values():
            self._release_entry(entry)
        self._cache.clear()
