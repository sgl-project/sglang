# SPDX-License-Identifier: Apache-2.0
"""Read-only CUDA FABRIC imports for externally owned Engram tables.

The wire layout matches the experimental vLLM shared Engram owner. The owner
alone allocates HOST_NUMA pages; SGLang maps the same pages into each local GPU
VA without allocating a table replica.
"""

import logging
from dataclasses import dataclass

import torch

from sglang.srt.utils.cuda_vmm_utils import _get_cuda_driver, check_drv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableHandles:
    size_bytes: int
    padded_bytes: int
    chunk_bytes: int
    fabric_handles: tuple[bytes, ...]

    @classmethod
    def from_wire(cls, value: dict) -> "TableHandles":
        size = int(value["size_bytes"])
        padded = int(value["padded_bytes"])
        chunk = int(value["chunk_bytes"])
        handles = tuple(bytes.fromhex(item) for item in value["fabric_handles"])
        expected = (padded + chunk - 1) // chunk if chunk > 0 else 0
        nodes = tuple(value.get("chunk_numa_nodes", ()))
        if (
            size <= 0
            or padded < size
            or chunk <= 0
            or not handles
            or len(handles) != expected
            or any(len(handle) != 64 for handle in handles)
            or (nodes and len(nodes) != len(handles))
        ):
            raise ValueError("invalid external Engram table extent")
        return cls(size, padded, chunk, handles)


class TableMapping:
    """A local VA that remains valid until GPU work and graphs have stopped."""

    def __init__(
        self,
        address: int,
        table: TableHandles,
        mapped: list[tuple[int, int]],
        device: int,
    ) -> None:
        self.address = address
        self.table = table
        self._mapped = mapped
        self.device = device
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        # A failed synchronization leaves the mapping and owner lease intact.
        try:
            torch.cuda.synchronize(self.device)
        except Exception:
            raise RuntimeError("cannot quiesce external Engram GPU mapping") from None
        drv = _get_cuda_driver()
        while self._mapped:
            offset, size = self._mapped[-1]
            check_drv(drv.cuMemUnmap(self.address + offset, size), "cuMemUnmap")
            self._mapped.pop()
        check_drv(
            drv.cuMemAddressFree(self.address, self.table.padded_bytes),
            "cuMemAddressFree",
        )
        self._closed = True


def import_table(table: TableHandles, device: int) -> TableMapping:
    """Import FABRIC handles with driver-level GPU read-only access."""
    drv = _get_cuda_driver()
    handle_type = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    handles = []
    address = None
    mapped: list[tuple[int, int]] = []
    try:
        for exported in table.fabric_handles:
            handles.append(
                check_drv(
                    drv.cuMemImportFromShareableHandle(exported, handle_type),
                    "cuMemImportFromShareableHandle",
                )
            )
        address = int(
            check_drv(
                drv.cuMemAddressReserve(table.padded_bytes, 0, 0, 0),
                "cuMemAddressReserve",
            )
        )
        access = drv.CUmemAccessDesc()
        access.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device
        access.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
        for index, handle in enumerate(handles):
            offset = index * table.chunk_bytes
            size = min(table.chunk_bytes, table.padded_bytes - offset)
            check_drv(drv.cuMemMap(address + offset, size, 0, handle, 0), "cuMemMap")
            mapped.append((offset, size))
            check_drv(
                drv.cuMemSetAccess(address + offset, size, [access], 1),
                "cuMemSetAccess",
            )
        return TableMapping(address, table, mapped, device)
    except BaseException:
        if address is not None:
            for offset, size in reversed(mapped):
                drv.cuMemUnmap(address + offset, size)
            drv.cuMemAddressFree(address, table.padded_bytes)
        raise
    finally:
        for handle in handles:
            result = drv.cuMemRelease(handle)
            error = result[0] if isinstance(result, tuple) else result
            if error != drv.CUresult.CUDA_SUCCESS:
                # Once mapped, propagating this error would lose the local VA
                # before the caller can keep its owner lease alive.
                logger.error("cuMemRelease(imported handle) failed: %s", error)
