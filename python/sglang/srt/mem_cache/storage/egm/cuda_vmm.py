# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""CUDA Driver API calls of the EGM storage tier, through ``cuda.bindings.driver``.

Covers the provider's host allocations (Virtual Memory Management with
``CU_MEM_LOCATION_TYPE_HOST_NUMA`` and ``CU_MEM_HANDLE_TYPE_FABRIC``), their
import and mapping on a backend rank, and the async copies between them. Every
call is checked; a failing call raises CudaDriverError with the driver's error
name.
"""

from __future__ import annotations

import ctypes
import mmap
import threading
import time
from typing import Dict, List, Sequence, Tuple

from cuda.bindings import driver as drv

from sglang.srt.mem_cache.storage.egm.store import coalesce

_GRANULARITY_MINIMUM = (
    drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
)
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


class CudaDriverError(RuntimeError):
    pass


def _check(result, api: str):
    if not isinstance(result, tuple):
        result = (result,)
    err = result[0]
    if err != drv.CUresult.CUDA_SUCCESS:
        name_err, name = drv.cuGetErrorName(err)
        label = name.decode() if name_err == drv.CUresult.CUDA_SUCCESS else "?"
        raise CudaDriverError("%s -> %s (%d)" % (api, label, int(err)))
    return result[1] if len(result) > 1 else None


def _init_driver() -> None:
    global _INITIALIZED
    with _INIT_LOCK:
        if not _INITIALIZED:
            _check(drv.cuInit(0), "cuInit")
            _INITIALIZED = True


class Context:
    """The device's primary context; enter() makes it current on this thread."""

    def __init__(self, ordinal: int):
        _init_driver()
        self.ordinal = int(ordinal)
        self.dev = _check(drv.cuDeviceGet(self.ordinal), "cuDeviceGet")
        self.ctx = _check(
            drv.cuDevicePrimaryCtxRetain(self.dev), "cuDevicePrimaryCtxRetain"
        )
        self.released = False

    def enter(self) -> None:
        _check(drv.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    def release(self) -> None:
        if not self.released:
            self.released = True
            _check(drv.cuDevicePrimaryCtxRelease(self.dev), "cuDevicePrimaryCtxRelease")


def _new_stream():
    return _check(
        drv.cuStreamCreate(drv.CUstream_flags.CU_STREAM_NON_BLOCKING),
        "cuStreamCreate",
    )


class Copier:
    """Async copies over several streams; run() waits for all of them or raises."""

    def __init__(self, ctx: Context, streams: int = 2):
        self.ctx = ctx
        self.lock = threading.Lock()
        ctx.enter()
        self.streams = [_new_stream() for _ in range(max(1, int(streams)))]
        self.issue_ms: List[float] = []

    def run(self, copies: Sequence[Tuple[int, int, int]], to_host: bool) -> dict:
        runs = coalesce(copies)
        with self.lock:
            self.ctx.enter()
            t0 = time.perf_counter()
            for i, (dst, src, n) in enumerate(runs):
                stream = self.streams[i % len(self.streams)]
                _check(drv.cuMemcpyAsync(dst, src, n, stream), "cuMemcpyAsync")
            t1 = time.perf_counter()
            for s in self.streams:
                _check(drv.cuStreamSynchronize(s), "cuStreamSynchronize")
            t2 = time.perf_counter()
        self.issue_ms.append((t1 - t0) * 1e3)
        if len(self.issue_ms) > 4096:
            del self.issue_ms[:2048]
        return {
            "ms": (t2 - t0) * 1e3,
            "issue_ms": (t1 - t0) * 1e3,
            "issued": len(runs),
        }


class HostBuf:
    """Anonymous host memory, page-locked as portable and device-mapped."""

    def __init__(self, nbytes: int, ctx: Context):
        self.n = nbytes
        self.mm = mmap.mmap(-1, nbytes)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        ctx.enter()
        flags = drv.CU_MEMHOSTREGISTER_PORTABLE | drv.CU_MEMHOSTREGISTER_DEVICEMAP
        _check(drv.cuMemHostRegister(self.addr, nbytes, flags), "cuMemHostRegister")
        self.registered = True

    def free(self) -> None:
        if self.registered:
            self.registered = False
            _check(drv.cuMemHostUnregister(self.addr), "cuMemHostUnregister")


def _access_desc(ordinal: int):
    ad = drv.CUmemAccessDesc()
    ad.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    ad.location.id = ordinal
    ad.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return ad


def _unmap_all(
    ctx: Context, mapped: List[Tuple[int, int]], va: int, total: int
) -> List[str]:
    errs = []
    try:
        ctx.enter()
    except Exception as e:
        errs.append("ctx: %s" % e)
    for addr, size in mapped:
        try:
            _check(drv.cuMemUnmap(addr, size), "cuMemUnmap")
        except Exception as e:
            errs.append("unmap: %s" % e)
    if va:
        try:
            _check(drv.cuMemAddressFree(va, total), "cuMemAddressFree")
        except Exception as e:
            errs.append("addrfree: %s" % e)
    return errs


def _map_back_to_back(
    ctx: Context, allocs: Sequence[Tuple[int, int, object]], total: int, gran: int
) -> Tuple[int, List[Tuple[int, int]]]:
    """Reserve one range, map each (off, size, handle) at its offset, grant RW access.

    On a failure everything mapped here is unmapped again before the error propagates.
    """
    va = int(_check(drv.cuMemAddressReserve(total, gran, 0, 0), "cuMemAddressReserve"))
    mapped: List[Tuple[int, int]] = []
    try:
        for off, size, handle in allocs:
            _check(drv.cuMemMap(va + off, size, 0, handle, 0), "cuMemMap")
            mapped.append((va + off, size))
        ad = _access_desc(ctx.ordinal)
        for off, size, _ in allocs:
            _check(drv.cuMemSetAccess(va + off, size, [ad], 1), "cuMemSetAccess")
    except Exception:
        _unmap_all(ctx, mapped, va, total)
        raise
    return va, mapped


class ImportedSegment:
    """A provider's allocations imported from fabric handles, mapped back to back."""

    def __init__(self, ctx: Context, handle: dict):
        self.ctx = ctx
        self.total = int(handle["total_size"])
        self.handles: List[object] = []
        self.mapped: List[Tuple[int, int]] = []
        self.va = 0
        self.times: Dict[str, float] = {}
        ctx.enter()
        fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        t0 = time.perf_counter()
        try:
            for a in handle["allocs"]:
                h = _check(
                    drv.cuMemImportFromShareableHandle(
                        bytes.fromhex(a["fabric"]), fabric
                    ),
                    "cuMemImportFromShareableHandle",
                )
                self.handles.append(h)
            t1 = time.perf_counter()
            allocs = [
                (int(a["off"]), int(a["size"]), h)
                for a, h in zip(handle["allocs"], self.handles)
            ]
            self.va, self.mapped = _map_back_to_back(
                ctx, allocs, self.total, int(handle["gran"])
            )
            t2 = time.perf_counter()
        except Exception:
            self.close()
            raise
        self.times = {"import_s": t1 - t0, "map_s": t2 - t1, "total_s": t2 - t0}

    def close(self) -> List[str]:
        errs = _unmap_all(self.ctx, self.mapped, self.va, self.total)
        self.mapped = []
        self.va = 0
        for h in self.handles:
            try:
                _check(drv.cuMemRelease(h), "cuMemRelease")
            except Exception as e:
                errs.append("release: %s" % e)
        self.handles = []
        return errs


class ExportedSegment:
    """Provider side: one host-NUMA allocation with a fabric handle per NUMA node,
    mapped back to back for the provider's device, which writes the control block."""

    def __init__(self, ctx: Context, numas: Sequence[int], bytes_per_numa: int):
        self.ctx = ctx
        self.allocs: List[dict] = []
        self.va = 0
        self.total = 0
        self.mapped: List[Tuple[int, int]] = []
        ctx.enter()
        fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        props = []
        g = 1
        for k in numas:
            p = drv.CUmemAllocationProp()
            p.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            p.requestedHandleTypes = fabric
            p.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
            p.location.id = int(k)
            gr = _check(
                drv.cuMemGetAllocationGranularity(p, _GRANULARITY_MINIMUM),
                "cuMemGetAllocationGranularity",
            )
            g = max(g, int(gr))
            props.append((int(k), p))
        self.gran = g
        size = ((int(bytes_per_numa) + g - 1) // g) * g
        try:
            off = 0
            for k, p in props:
                t = time.perf_counter()
                h = _check(drv.cuMemCreate(size, p, 0), "cuMemCreate")
                self.allocs.append(
                    {
                        "numa": k,
                        "handle": h,
                        "size": size,
                        "off": off,
                        "create_s": time.perf_counter() - t,
                    }
                )
                off += size
            self.total = off
            self.va, self.mapped = _map_back_to_back(
                ctx,
                [(a["off"], a["size"], a["handle"]) for a in self.allocs],
                self.total,
                g,
            )
            for a in self.allocs:
                fh = _check(
                    drv.cuMemExportToShareableHandle(a["handle"], fabric, 0),
                    "cuMemExportToShareableHandle",
                )
                a["fabric"] = bytes(fh.data).hex()
        except Exception:
            self.close()
            raise

    def close(self) -> List[str]:
        errs = _unmap_all(self.ctx, self.mapped, self.va, self.total)
        self.mapped = []
        self.va = 0
        for a in self.allocs:
            if a["handle"] is not None:
                try:
                    _check(drv.cuMemRelease(a["handle"]), "cuMemRelease")
                except Exception as e:
                    errs.append("release: %s" % e)
                a["handle"] = None
        return errs


class SmallIO:
    """One page-locked host page and one stream for control-block reads and writes,
    separate from the data copies."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.buf = HostBuf(4096, ctx)
        self.stream = _new_stream()
        self.lock = threading.Lock()

    def read(self, src: int, n: int) -> bytes:
        with self.lock:
            self.ctx.enter()
            _check(
                drv.cuMemcpyAsync(self.buf.addr, src, n, self.stream), "cuMemcpyAsync"
            )
            _check(drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize")
            return bytes(self.buf.mm[:n])

    def write(self, dst: int, data: bytes) -> None:
        with self.lock:
            self.ctx.enter()
            self.buf.mm[: len(data)] = data
            _check(
                drv.cuMemcpyAsync(dst, self.buf.addr, len(data), self.stream),
                "cuMemcpyAsync",
            )
            _check(drv.cuStreamSynchronize(self.stream), "cuStreamSynchronize")

    def close(self) -> None:
        try:
            self.ctx.enter()
            _check(drv.cuStreamDestroy(self.stream), "cuStreamDestroy")
        finally:
            self.buf.free()
