import array
import ctypes
import logging
import os
import socket
import struct
import tempfile
import threading
import time
from functools import cache
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

_FD_HEADER_BYTES = 24
_FD_SEND_TIMEOUT_S = 120.0

try:
    from cuda.bindings import driver as _drv
except ImportError:
    _drv = None

if _drv is None:
    _RECOMMENDED_GRANULARITY = 1
else:
    _RECOMMENDED_GRANULARITY = (
        _drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    )

try:
    import pynvml
except ImportError:
    pynvml = None

_NVML_GPU_FABRIC_INFO_V3_TYPE = None
_NVML_GPU_FABRIC_INFO_V3_VERSION = None
if pynvml is not None:
    try:
        _NVML_GPU_FABRIC_INFO_V3_TYPE = pynvml.c_nvmlGpuFabricInfo_v3_t
        _NVML_GPU_FABRIC_INFO_V3_VERSION = pynvml.nvmlGpuFabricInfo_v3
    except AttributeError:
        pass

# NVML_GPU_FABRIC_STATE_COMPLETED: the GPU has joined its NVLink fabric clique.
_NVML_GPU_FABRIC_STATE_COMPLETED = 3


def _get_cuda_driver():
    """Return the imported CUDA driver bindings."""
    if _drv is None:
        raise ImportError("cuda.bindings.driver is required for CUDA VMM operations")
    return _drv


def check_drv(result_tuple, label):
    """Check a cuda.bindings driver call result and return the value."""
    if not isinstance(result_tuple, tuple):
        result_tuple = (result_tuple,)
    err = result_tuple[0]
    drv = _get_cuda_driver()
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label}: {err}")
    return result_tuple[1] if len(result_tuple) > 1 else None


def tensor_from_pointer(
    pointer: int,
    nbytes: int,
    *,
    shape=None,
    dtype: torch.dtype = torch.uint8,
    device_id: int,
) -> torch.Tensor:
    """Use non-owning storage; the caller controls the underlying pages' lifetime."""
    device = torch.device("cuda", device_id)
    storage = torch._C._construct_storage_from_data_pointer(pointer, device, nbytes)
    if shape is None:
        shape = (nbytes,)
    return torch.empty(0, dtype=dtype, device=device).set_(storage, 0, shape)


def is_vmm_pointer(ptr: int) -> bool:
    """Check if a device pointer is VMM-backed (cuMemCreate/cuMemMap).

    cuMemRetainAllocationHandle succeeds only on pointers from cuMemCreate;
    it fails on cudaMalloc pointers.
    """
    drv = _get_cuda_driver()
    err, handle = drv.cuMemRetainAllocationHandle(ptr)
    if err == drv.CUresult.CUDA_SUCCESS:
        drv.cuMemRelease(handle)
        return True
    return False


def compute_graph_capture_bases(graph_inputs: List[tuple]):
    """Map graph-capture inputs onto their VMM base allocations.

    ``graph_inputs`` is a list of ``(device_ptr, nbytes)`` pairs. A captured
    tensor can cross expandable-segment allocation boundaries, so each input
    is walked with ``cuMemGetAddressRange`` until its byte span is covered.

    Returns ``(bases_info, input_chunk_indices, input_offsets)``:
      - ``bases_info[i] = (base_ptr, alloc_size)`` per unique allocation
      - ``input_chunk_indices[j]`` = indices of allocations covering input j
      - ``input_offsets[j]`` = byte offset of input j from its first base
    """
    drv = _get_cuda_driver()
    base_to_idx = {}
    bases_info: List[tuple] = []
    input_chunk_indices: List[List[int]] = []
    input_offsets: List[int] = []
    for ptr, nbytes in graph_inputs:
        ptr, remaining = int(ptr), int(nbytes)
        if remaining <= 0:
            raise RuntimeError(f"Invalid graph capture input size: {nbytes}")
        cursor = ptr
        first_base = None
        chunks: List[int] = []
        while remaining > 0:
            err, base, size = drv.cuMemGetAddressRange(cursor)
            if err != drv.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemGetAddressRange: {err}")
            base, size = int(base), int(size)
            if first_base is None:
                first_base = base
            byte_offset = cursor - base
            if not 0 <= byte_offset < size:
                raise RuntimeError(
                    f"graph capture input at {ptr} is outside VMM allocation "
                    f"[base={base}, size={size}]"
                )
            idx = base_to_idx.setdefault(base, len(bases_info))
            if idx == len(bases_info):
                bases_info.append((base, size))
            chunks.append(idx)
            advance = min(remaining, size - byte_offset)
            assert advance > 0, "Failed to advance VMM graph capture span"
            remaining -= advance
            cursor += advance
        input_chunk_indices.append(chunks)
        input_offsets.append(ptr - first_base)
    return bases_info, input_chunk_indices, input_offsets


def make_rw_access_desc(device_id: int):
    """A read-write, device-local ``CUmemAccessDesc`` for ``device_id``."""
    drv = _get_cuda_driver()
    desc = drv.CUmemAccessDesc()
    desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device_id
    desc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return desc


def _gpu_fabric_clique(device: torch.device):
    """Return this GPU's NVLink fabric clique, or ``None`` if not joined."""
    if pynvml is None:
        return None
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices:
        device_ids = list(map(int, cuda_visible_devices.split(",")))
    else:
        device_ids = list(range(torch.cuda.device_count()))
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_ids[device.index])
    if (
        _NVML_GPU_FABRIC_INFO_V3_TYPE is not None
        and _NVML_GPU_FABRIC_INFO_V3_VERSION is not None
    ):
        fabric = _NVML_GPU_FABRIC_INFO_V3_TYPE()
        fabric.version = _NVML_GPU_FABRIC_INFO_V3_VERSION
        pynvml.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(fabric))
        clique_id = fabric.cliqueId
    else:
        fabric = pynvml.c_nvmlGpuFabricInfo_t()
        pynvml.nvmlDeviceGetGpuFabricInfo(handle, ctypes.byref(fabric))
        clique_id = fabric.partitionId
    if fabric.state != _NVML_GPU_FABRIC_STATE_COMPLETED:
        return None
    return (bytes(fabric.clusterUuid), int(clique_id))


def is_gpu_fabric_ready(device: torch.device) -> bool:
    """Whether one CUDA GPU has completed NVLink fabric initialization."""
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        try:
            return _gpu_fabric_clique(device) is not None
        finally:
            pynvml.nvmlShutdown()
    except Exception as error:
        logger.warning("GPU fabric readiness query failed: %r", error)
        return False


def allocation_handle_type_name(handle_type: int) -> str:
    """Return a stable display name for a CUDA allocation handle type."""
    drv = _get_cuda_driver()
    fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    posix_fd = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    if handle_type == fabric:
        return "FABRIC"
    if handle_type == posix_fd:
        return "POSIX_FD"
    if handle_type == 0:
        return "NONE"
    return str(handle_type)


@cache
def get_device_allocation_handle_type(device_id: int) -> int:
    """Probe and cache the best supported VMM handle type for one device."""
    device_id = int(device_id)
    drv = _get_cuda_driver()
    if not is_gpu_fabric_ready(torch.device("cuda", device_id)):
        logger.info(
            "GPU %d has not joined an NVLink fabric clique; probing local "
            "FABRIC allocation support",
            device_id,
        )

    fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    posix_fd = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    candidates = (fabric, posix_fd, 0)
    last_error = None
    for handle_type in candidates:
        name = allocation_handle_type_name(handle_type)
        prop = make_device_allocation_prop(
            device_id,
            handle_types=handle_type,
            gpu_direct_rdma=False,
        )
        try:
            granularity = get_allocation_granularity(prop)
            probe_handle = check_drv(
                drv.cuMemCreate(granularity, prop, 0),
                f"cuMemCreate({name} probe)",
            )
            check_drv(
                drv.cuMemRelease(probe_handle),
                f"cuMemRelease({name} probe)",
            )
        except RuntimeError as error:
            last_error = error
            logger.warning(
                "CUDA VMM %s backing unavailable on device %d; trying fallback: %s",
                name,
                device_id,
                error,
            )
            continue
        logger.info(
            "CUDA VMM selected %s backing for device %d",
            name,
            device_id,
        )
        return handle_type
    raise RuntimeError("no supported CUDA VMM allocation handle type") from last_error


def make_device_allocation_prop(
    device_id: int,
    *,
    handle_types: int | str | None = "auto",
    gpu_direct_rdma: bool = False,
):
    """Build a device allocation prop with automatic or explicit exportability."""
    drv = _get_cuda_driver()
    if handle_types == "auto":
        handle_types = get_device_allocation_handle_type(device_id)
    elif handle_types is None:
        handle_types = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    elif not isinstance(handle_types, int):
        raise ValueError("handle_types must be 'auto', an integer, or None")

    handle_type_value = int(handle_types)
    valid_handle_types = {
        int(drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE): (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        ),
        int(drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR): (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        ),
        int(drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC): (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        ),
    }
    if handle_type_value not in valid_handle_types:
        raise ValueError(f"invalid CUDA handle-type value: {handle_type_value}")

    prop = drv.CUmemAllocationProp()
    prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = int(device_id)
    # cuda-bindings 13.0.x requires the generated enum here; newer releases
    # also accept a plain int, which previously hid this compatibility issue.
    prop.requestedHandleTypes = valid_handle_types[handle_type_value]
    prop.allocFlags.gpuDirectRDMACapable = int(gpu_direct_rdma)
    return prop


def get_allocation_granularity(prop, flag=_RECOMMENDED_GRANULARITY) -> int:
    """Return allocation granularity for a CUDA policy flag."""
    drv = _get_cuda_driver()
    return int(
        check_drv(
            drv.cuMemGetAllocationGranularity(prop, flag),
            "cuMemGetAllocationGranularity",
        )
    )


@cache
def get_device_granularity(device_id: int) -> int:
    """Granularity for this device's default allocations. Cached: it is a device
    constant, and callers that size a reservation must agree with the one that
    maps into it."""
    device_id = int(device_id)
    return get_allocation_granularity(make_device_allocation_prop(device_id))


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to a positive byte ``alignment``."""
    return (int(value) + alignment - 1) // alignment * alignment


def align_down(value: int, alignment: int) -> int:
    """Round ``value`` down to a positive byte ``alignment``."""
    return int(value) // alignment * alignment


# Bump allocator over caller-provided extents: malloc first-fits an extent and
# hands back base+cursor, bounded by each extent's RESERVED size (not any
# committed watermark) so upper-bound tensors can be allocated before physical
# commit. Allocations are aligned so VMM users can commit each pointer at its
# own VA range (cuMemMap requires it; GB300 rejects partial-handle maps).
# Symbols are SUFFIXED per (process, arena instance) and each instance loads its
# own .so, so neither multiple arenas per process nor co-located engine
# processes sharing the tempdir clobber each other.
def _bump_arena_stub_source(sfx: str) -> str:
    return f"""
#include <cstddef>
#include <cstdint>
#include <mutex>
extern "C" {{
enum {{ BUMPARENA_MAX_EXTENTS = 64 }};
static uintptr_t g_bases[BUMPARENA_MAX_EXTENTS];
static size_t g_reserved[BUMPARENA_MAX_EXTENTS];
static size_t g_cursors[BUMPARENA_MAX_EXTENTS];
static size_t g_num_extents = 0;
static size_t g_freed_bytes = 0;
static size_t g_align = 512;
static int g_best_fit = 0;
static std::mutex g_mu;
static size_t align_up(size_t v, size_t a){{ return (v + a - 1) / a * a; }}
void bumparena_set_extents_{sfx}(const uintptr_t* bases, const size_t* sizes, size_t n){{
  std::lock_guard<std::mutex> lk(g_mu);
  if (n > BUMPARENA_MAX_EXTENTS) n = BUMPARENA_MAX_EXTENTS;
  g_num_extents = n;
  g_freed_bytes = 0;
  for (size_t i = 0; i < n; ++i) {{
    g_bases[i] = bases[i];
    g_reserved[i] = sizes[i];
    g_cursors[i] = 0;
  }}
}}
void bumparena_set_align_{sfx}(size_t a){{ std::lock_guard<std::mutex> lk(g_mu); if (a) g_align=a; }}
void bumparena_set_best_fit_{sfx}(int on){{ std::lock_guard<std::mutex> lk(g_mu); g_best_fit = on; }}
size_t bumparena_cursor_{sfx}(void){{
  std::lock_guard<std::mutex> lk(g_mu);
  size_t total = 0;
  for (size_t i = 0; i < g_num_extents; ++i) total += g_cursors[i];
  return total;
}}
void* bumparena_malloc_{sfx}(size_t size, int device, void* stream){{
  std::lock_guard<std::mutex> lk(g_mu);
  size_t need = align_up(size, g_align);
  size_t pick = BUMPARENA_MAX_EXTENTS;
  for (size_t i = 0; i < g_num_extents; ++i) {{
    size_t avail = g_reserved[i] - g_cursors[i];
    if (avail < need) continue;
    // First fit is for callers that map physical pages at the offsets they
    // are handed, so extent order is theirs to choose.
    if (!g_best_fit) {{ pick = i; break; }}
    if (pick == BUMPARENA_MAX_EXTENTS ||
        avail < g_reserved[pick] - g_cursors[pick]) pick = i;
  }}
  if (pick == BUMPARENA_MAX_EXTENTS) return 0;   // no extent fits -- surfaces as an allocator OOM
  void* p = reinterpret_cast<void*>(g_bases[pick] + g_cursors[pick]);
  g_cursors[pick] += need;
  return p;
}}
size_t bumparena_freed_{sfx}(void){{ std::lock_guard<std::mutex> lk(g_mu); return g_freed_bytes; }}
void bumparena_free_{sfx}(void* ptr, size_t size, int device, void* stream){{
  std::lock_guard<std::mutex> lk(g_mu);
  g_freed_bytes += size;
}}
}}
"""


class BumpArenaStub:
    """JIT-built pluggable bump allocator over caller-provided device VA extents.

    ``malloc`` picks an extent and hands out ``base + cursor``; ``free`` is a
    no-op. Plain ``torch.empty`` can thus be placed on externally managed
    storage by wrapping ``allocator`` in a ``torch.cuda.MemPool``.
    ``set_extents`` re-points the arena and resets every cursor, letting one
    stub serve successive region sets.
    """

    MAX_EXTENTS = 64  # mirrors BUMPARENA_MAX_EXTENTS in the stub source

    # Per-instance suffix -> isolated allocator symbols/state (see _bump_arena_stub_source).
    _instance_count = 0

    def __init__(self):
        # Unique per (process, instance): the stub .so lives in a host-shared
        # tempdir, so co-located engine processes must not build the same-named
        # .so (they race and one loads a half-relinked copy -> undefined symbol
        # crash).
        self.sfx = f"{os.getpid()}_{BumpArenaStub._instance_count}"
        BumpArenaStub._instance_count += 1
        self._lib = self._build()
        from torch.cuda.memory import CUDAPluggableAllocator

        self.allocator = CUDAPluggableAllocator(
            self._so_path,
            f"bumparena_malloc_{self.sfx}",
            f"bumparena_free_{self.sfx}",
        ).allocator()

    def _build(self) -> ctypes.CDLL:
        import torch.utils.cpp_extension

        # Per-stub build dir: load_inline writes every caller's source to the
        # same main.cpp inside build_directory, so any sharing (across
        # co-located engine processes under the host tempdir, or across arenas
        # within one process) can compile another stub's source and link a .so
        # missing this stub's symbols. One dir per stub means no shared ninja
        # scratch or .so, ever.
        out_dir = os.path.join(tempfile.gettempdir(), "sgl_bump_arena", self.sfx)
        os.makedirs(out_dir, exist_ok=True)
        libname = f"sgl_bump_arena_stub_{self.sfx}"
        torch.utils.cpp_extension.load_inline(
            name=libname,
            cpp_sources=_bump_arena_stub_source(self.sfx),
            with_cuda=False,  # pure arithmetic -- no nvcc, no CUDA headers
            is_python_module=False,
            verbose=False,
            build_directory=out_dir,
            no_implicit_headers=True,
        )
        self._so_path = f"{out_dir}/{libname}.so"
        lib = ctypes.CDLL(self._so_path)
        self._fn_set_extents = lib[f"bumparena_set_extents_{self.sfx}"]
        self._fn_set_extents.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
        ]
        self._fn_set_extents.restype = None
        self._fn_set_align = lib[f"bumparena_set_align_{self.sfx}"]
        self._fn_set_align.argtypes = [ctypes.c_size_t]
        self._fn_set_align.restype = None
        self._fn_set_best_fit = lib[f"bumparena_set_best_fit_{self.sfx}"]
        self._fn_set_best_fit.argtypes = [ctypes.c_int]
        self._fn_set_best_fit.restype = None
        self._fn_cursor = lib[f"bumparena_cursor_{self.sfx}"]
        self._fn_cursor.argtypes = []
        self._fn_cursor.restype = ctypes.c_size_t
        self._fn_freed = lib[f"bumparena_freed_{self.sfx}"]
        self._fn_freed.argtypes = []
        self._fn_freed.restype = ctypes.c_size_t
        return lib

    def set_extents(self, extents: List[tuple]) -> None:
        """Register ``(base, nbytes)`` extents (first-fit order) and reset
        every bump cursor."""
        if len(extents) > BumpArenaStub.MAX_EXTENTS:
            raise ValueError(
                f"{len(extents)} extents exceed BUMPARENA_MAX_EXTENTS "
                f"({BumpArenaStub.MAX_EXTENTS})"
            )
        n = len(extents)
        bases = (ctypes.c_void_p * n)(*(base for base, _ in extents))
        sizes = (ctypes.c_size_t * n)(*(nbytes for _, nbytes in extents))
        self._fn_set_extents(bases, sizes, ctypes.c_size_t(n))

    def set_align(self, nbytes: int) -> None:
        self._fn_set_align(ctypes.c_size_t(nbytes))

    def set_best_fit(self, on: bool) -> None:
        self._fn_set_best_fit(ctypes.c_int(1 if on else 0))

    @property
    def cursor_bytes(self) -> int:
        return int(self._fn_cursor())

    @property
    def freed_bytes(self) -> int:
        """Bytes handed back through ``free`` since the last ``set_extents``
        -- nonzero means someone (empty_cache) released this arena's segments."""
        return int(self._fn_freed())


class VmmReservation:
    """Own a VA reservation, its mappings, and their teardown order."""

    def __init__(
        self,
        size: int,
        prop,
        device_id: int,
        *,
        alignment: int = 0,
        requested_address: int = 0,
    ) -> None:
        drv = _get_cuda_driver()
        self.size = int(size)
        self._prop = prop
        self._access_descs = [make_rw_access_desc(int(device_id))]
        self.base = int(
            check_drv(
                drv.cuMemAddressReserve(
                    self.size,
                    int(alignment),
                    int(requested_address),
                    0,
                ),
                "cuMemAddressReserve(local)",
            )
        )
        self._mappings = []
        self._closed = False

    def map(
        self,
        offset: int,
        size: int,
        *,
        retain_handle: bool,
    ):
        """Create and map local memory at ``base + offset``."""
        if self._closed:
            raise RuntimeError("VmmReservation.map after close")
        offset, size = int(offset), int(size)
        if offset < 0 or size <= 0 or offset + size > self.size:
            raise ValueError(
                f"mapping [{offset}, {offset + size}) is outside reservation "
                f"[0, {self.size})"
            )

        drv = _get_cuda_driver()
        address = self.base + offset
        handle = check_drv(drv.cuMemCreate(size, self._prop, 0), "cuMemCreate(local)")
        mapped = False
        try:
            check_drv(
                drv.cuMemMap(address, size, 0, handle, 0),
                "cuMemMap(local)",
            )
            mapped = True
            check_drv(
                drv.cuMemSetAccess(
                    address,
                    size,
                    self._access_descs,
                    len(self._access_descs),
                ),
                "cuMemSetAccess(local)",
            )
            if not retain_handle:
                check_drv(drv.cuMemRelease(handle), "cuMemRelease(local)")
                handle = None
        except BaseException as error:
            cleanup_errors = []
            if mapped:
                try:
                    check_drv(
                        drv.cuMemUnmap(address, size), "cuMemUnmap(local rollback)"
                    )
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if handle is not None:
                try:
                    check_drv(drv.cuMemRelease(handle), "cuMemRelease(local rollback)")
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                error.add_note(
                    f"{len(cleanup_errors)} CUDA VMM rollback operation(s) also failed"
                )
                raise error from cleanup_errors[0]
            raise

        self._mappings.append((address, size, handle))
        return handle

    def map_existing(self, offset: int, size: int, handle) -> None:
        """Map a caller-owned physical allocation into this reservation."""
        if self._closed:
            raise RuntimeError("VmmReservation.map_existing after close")
        offset, size = int(offset), int(size)
        drv = _get_cuda_driver()
        address = self.base + offset
        mapped = False
        try:
            check_drv(
                drv.cuMemMap(address, size, 0, handle, 0),
                "cuMemMap(existing)",
            )
            mapped = True
            check_drv(
                drv.cuMemSetAccess(
                    address,
                    size,
                    self._access_descs,
                    len(self._access_descs),
                ),
                "cuMemSetAccess(existing)",
            )
        except BaseException as error:
            if mapped:
                try:
                    check_drv(
                        drv.cuMemUnmap(address, size),
                        "cuMemUnmap(existing rollback)",
                    )
                except BaseException as cleanup_error:
                    error.add_note("CUDA VMM alias rollback also failed")
                    raise error from cleanup_error
            raise

        self._mappings.append((address, size, None))

    def close(self, *, release_handles: bool = True) -> None:
        """Unmap allocations, optionally release retained handles, and free VA."""
        if self._closed:
            return
        self._closed = True
        drv = _get_cuda_driver()
        while self._mappings:
            address, size, handle = self._mappings.pop()
            err = drv.cuMemUnmap(address, size)
            err = err[0] if isinstance(err, tuple) else err
            if err != drv.CUresult.CUDA_SUCCESS:
                logger.warning("cuMemUnmap(local) -> %s", err)
            if release_handles and handle is not None:
                err = drv.cuMemRelease(handle)
                err = err[0] if isinstance(err, tuple) else err
                if err != drv.CUresult.CUDA_SUCCESS:
                    logger.warning("cuMemRelease(local) -> %s", err)
        err = drv.cuMemAddressFree(self.base, self.size)
        err = err[0] if isinstance(err, tuple) else err
        if err != drv.CUresult.CUDA_SUCCESS:
            logger.warning("cuMemAddressFree(local) -> %s", err)


def all_ranks_ok(group: ProcessGroup, ok: bool) -> bool:
    """True iff ``ok`` holds on every rank in ``group`` (BAND all-reduce)."""
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)
    return flag.item() == 1


def release_mappings(mappings) -> None:
    """Unmap + address-free each ``(va, span_size, [(rel, size), ...])`` mapping.

    Pops from ``mappings`` so a partially-released list is safe to retry.
    """
    drv = _get_cuda_driver()
    while mappings:
        va, span_size, mapped_chunks = mappings.pop()
        for rel, size in mapped_chunks:
            check_drv(drv.cuMemUnmap(int(va) + int(rel), int(size)), "cuMemUnmap")
        check_drv(drv.cuMemAddressFree(int(va), int(span_size)), "cuMemAddressFree")


def _send_fd(sock, fd: int, src_rank: int, base_idx: int) -> None:
    fds = array.array("i", [int(fd)])
    header = struct.pack("<QQQ", int(src_rank), int(base_idx), 1)
    sent = sock.sendmsg(
        [header],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())],
    )
    if sent != len(header):
        raise RuntimeError(f"sendmsg sent {sent} bytes, expected {len(header)}")


def _recv_fd(sock):
    fd_item_size = array.array("i").itemsize
    data, ancdata, _, _ = sock.recvmsg(
        _FD_HEADER_BYTES, socket.CMSG_SPACE(fd_item_size)
    )
    if not data:
        return None
    if len(data) != _FD_HEADER_BYTES:
        raise RuntimeError(
            f"received truncated fd header: {len(data)} < {_FD_HEADER_BYTES}"
        )
    src_rank, base_idx, fd_count = struct.unpack("<QQQ", data)
    fds = array.array("i")
    for level, cmsg_type, cmsg_data in ancdata:
        if level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fd_item_size)])
    if fd_count != 1 or len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise RuntimeError(
            f"expected one fd, got header={fd_count}, ancillary={len(fds)}"
        )
    return int(src_rank), int(base_idx), int(fds[0])


def export_shareable_handles(retained_handles, group: ProcessGroup, rank: int):
    """Export retained VMM handles, preferring FABRIC and falling back to POSIX fds.

    FABRIC is used only if every rank can export it; otherwise all ranks use POSIX
    fds. Returns ``(fabric_handles, posix_fds, use_fabric)`` (one list populated);
    raises if both fail on any rank. Caller owns the returned ``posix_fds``.
    """
    drv = _get_cuda_driver()
    FABRIC = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    POSIX_FD = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    fabric_handles: List[bytes] = []
    fabric_error: Optional[Exception] = None
    try:
        for alloc_h in retained_handles:
            fabric_h = check_drv(
                drv.cuMemExportToShareableHandle(alloc_h, FABRIC, 0),
                "cuMemExportToShareableHandle(FABRIC)",
            )
            fabric_handles.append(bytes(fabric_h.data))
        fabric_ok = True
    except Exception as e:
        fabric_error = e
        fabric_ok = False
        fabric_handles = []
        logger.info(
            "FABRIC handle export failed on rank %s; falling back to "
            "POSIX fd transport: %s",
            rank,
            e,
        )

    if all_ranks_ok(group, fabric_ok):
        return fabric_handles, [], True

    posix_fds: List[int] = []
    posix_error: Optional[Exception] = None
    try:
        for alloc_h in retained_handles:
            fd = check_drv(
                drv.cuMemExportToShareableHandle(alloc_h, POSIX_FD, 0),
                "cuMemExportToShareableHandle(POSIX_FD)",
            )
            posix_fds.append(int(fd))
        posix_ok = True
    except Exception as e:
        posix_error = e
        posix_ok = False
        for fd in posix_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        posix_fds = []

    if not all_ranks_ok(group, posix_ok):
        cause = posix_error or fabric_error
        message = (
            "VMM handle export failed: FABRIC export failed on at least one "
            "rank and POSIX fd export failed on at least one rank"
        )
        if cause is not None:
            message += f"; local rank {rank} error: {cause}"
        raise RuntimeError(message) from posix_error

    return [], posix_fds, False


def exchange_posix_fds(
    group: ProcessGroup,
    rank: int,
    world_size: int,
    local_fds: List[int],
    peer_base_counts: List[int],
):
    """Exchange POSIX file descriptors across ranks via SCM_RIGHTS over a UNIX
    socket. Returns ``{(src_rank, base_idx): fd}`` for every peer. The caller
    owns the received fds and must close them.
    """
    sock_kind = socket.SOCK_SEQPACKET
    sock_dir = tempfile.mkdtemp(prefix="sgl_ar_fd_")
    sock_path = os.path.join(sock_dir, f"rank_{rank}.sock")
    server = socket.socket(socket.AF_UNIX, sock_kind)
    server.settimeout(_FD_SEND_TIMEOUT_S)
    received_fds = {}
    errors = []

    def recv_loop():
        try:
            for _ in range(world_size - 1):
                conn, _ = server.accept()
                with conn:
                    conn.settimeout(_FD_SEND_TIMEOUT_S)
                    while True:
                        packet = _recv_fd(conn)
                        if packet is None:
                            break
                        src_rank, base_idx, fd = packet
                        key = (src_rank, base_idx)
                        if key in received_fds:
                            os.close(fd)
                            raise RuntimeError(f"duplicate fd for {key}")
                        received_fds[key] = fd
        except BaseException as e:
            errors.append(e)

    try:
        server.bind(sock_path)
        server.listen(world_size)
        paths = [None] * world_size
        dist.all_gather_object(paths, sock_path, group=group)

        thread = threading.Thread(target=recv_loop, daemon=True)
        thread.start()
        try:
            for peer_rank, peer_path in enumerate(paths):
                if peer_rank == rank:
                    continue
                with socket.socket(socket.AF_UNIX, sock_kind) as sock:
                    sock.settimeout(_FD_SEND_TIMEOUT_S)
                    sock.connect(peer_path)
                    for base_idx, fd in enumerate(local_fds):
                        _send_fd(sock, fd, rank, base_idx)
        finally:
            thread.join(_FD_SEND_TIMEOUT_S)

        if thread.is_alive():
            raise RuntimeError("timed out waiting for POSIX fd exchange")
        if errors:
            raise RuntimeError("POSIX fd exchange receive failed") from errors[0]

        expected = {
            (src_rank, base_idx)
            for src_rank, count in enumerate(peer_base_counts)
            if src_rank != rank
            for base_idx in range(count)
        }
        missing = expected.difference(received_fds)
        extra = set(received_fds).difference(expected)
        if missing or extra:
            for fd in received_fds.values():
                os.close(fd)
            raise RuntimeError(
                "POSIX fd exchange mismatch: "
                f"missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}"
            )
        return received_fds
    finally:
        server.close()
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(sock_dir)
        except OSError:
            pass


def import_peer_handle(fabric_handle, fd, *, use_fabric: bool, peer_rank: int):
    """Import a peer allocation handle (FABRIC or POSIX fd). Returns the handle.

    For POSIX the fd is duped before import so the caller keeps ownership of the
    original.
    """
    drv = _get_cuda_driver()
    if use_fabric:
        FABRIC = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        return check_drv(
            drv.cuMemImportFromShareableHandle(fabric_handle, FABRIC),
            f"cuMemImportFromShareableHandle(rank={peer_rank})",
        )
    POSIX_FD = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    dup_fd = os.dup(fd)
    try:
        return check_drv(
            drv.cuMemImportFromShareableHandle(dup_fd, POSIX_FD),
            f"cuMemImportFromShareableHandle(rank={peer_rank}, POSIX_FD)",
        )
    finally:
        try:
            os.close(dup_fd)
        except OSError:
            pass


def import_and_map_alloc(
    fabric_handle,
    fd,
    alloc_size: int,
    device_id: int,
    *,
    use_fabric: bool,
    peer_rank: int,
) -> int:
    """Import a peer allocation, map it at a freshly reserved VA, return the VA."""
    drv = _get_cuda_driver()
    imp_h = import_peer_handle(
        fabric_handle, fd, use_fabric=use_fabric, peer_rank=peer_rank
    )
    prop = check_drv(
        drv.cuMemGetAllocationPropertiesFromHandle(imp_h),
        "cuMemGetAllocationPropertiesFromHandle",
    )
    gran = get_allocation_granularity(prop)
    va = check_drv(
        drv.cuMemAddressReserve(alloc_size, int(gran), 0, 0), "cuMemAddressReserve"
    )
    check_drv(drv.cuMemMap(int(va), alloc_size, 0, imp_h, 0), "cuMemMap")
    access = make_rw_access_desc(device_id)
    check_drv(drv.cuMemSetAccess(int(va), alloc_size, [access], 1), "cuMemSetAccess")
    check_drv(drv.cuMemRelease(imp_h), "cuMemRelease(peer)")
    return int(va)


def map_chunk_into_span(
    fabric_handle,
    fd,
    span_va: int,
    rel: int,
    alloc_size: int,
    device_id: int,
    *,
    use_fabric: bool,
    peer_rank: int,
) -> None:
    """Import + map a peer chunk into a caller-reserved span at ``span_va + rel``."""
    drv = _get_cuda_driver()
    imp_h = import_peer_handle(
        fabric_handle, fd, use_fabric=use_fabric, peer_rank=peer_rank
    )
    check_drv(
        drv.cuMemMap(int(span_va) + rel, int(alloc_size), 0, imp_h, 0),
        "cuMemMap(span)",
    )
    access = make_rw_access_desc(device_id)
    check_drv(
        drv.cuMemSetAccess(int(span_va) + rel, int(alloc_size), [access], 1),
        "cuMemSetAccess(span)",
    )
    check_drv(drv.cuMemRelease(imp_h), "cuMemRelease(span)")


class VmmGraphInputManager:
    def __init__(
        self,
        obj: Any,
        group: ProcessGroup,
        rank: int,
        world_size: int,
    ) -> None:
        self.obj = obj
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self._peer_mappings = []

    def register_graph_inputs(self):
        """Register graph capture inputs via VMM handle exchange.

        VMM-compatible path for expandable_segments. The C++ side deduplicates
        graph capture pointers into unique base allocations via cuMemGetAddressRange.
        Python exports handles for each unique base, imports + cuMemMaps peer
        allocations, then registers the peer virtual addresses. FABRIC handles are
        preferred; POSIX file descriptors are used when FABRIC is unavailable.
        """
        FABRIC_HANDLE_BYTES = 64
        MAX_VMM_BASES = 4096
        MAX_CHUNKS_PER_INPUT = 16

        t0 = time.perf_counter()

        bases_info, input_chunk_indices, input_offsets = (
            self.obj.get_graph_capture_bases()
        )
        if not bases_info:
            return
        new_count = len(input_chunk_indices)
        num_bases = len(bases_info)
        device_id = torch.cuda.current_device()

        if num_bases > MAX_VMM_BASES:
            raise RuntimeError(
                f"Too many VMM bases to share: {num_bases} > {MAX_VMM_BASES}"
            )

        drv = _get_cuda_driver()
        local_posix_fds: List[int] = []
        retained_handles = []
        try:
            for base_ptr, _ in bases_info:
                alloc_h = check_drv(
                    drv.cuMemRetainAllocationHandle(base_ptr),
                    "cuMemRetainAllocationHandle",
                )
                retained_handles.append(alloc_h)

            local_fabric_handles, local_posix_fds, use_fabric = (
                export_shareable_handles(retained_handles, self.group, self.rank)
            )

            local_input_chunks = [
                [int(idx) for idx in indices] for indices in input_chunk_indices
            ]
            for chunks in local_input_chunks:
                if len(chunks) > MAX_CHUNKS_PER_INPUT:
                    raise RuntimeError(
                        "Too many VMM chunks for graph input: "
                        f"{len(chunks)} > {MAX_CHUNKS_PER_INPUT}"
                    )

            # All-gather base metadata and per-input VMM spans. A captured tensor
            # can cross expandable-segment allocation boundaries, so peer mappings
            # must preserve each input's contiguous virtual-address span. FABRIC
            # handles are inline metadata; POSIX fds are exchanged separately via
            # SCM_RIGHTS because fd integers are process-local.
            header_struct = struct.Struct("<QQ")
            base_struct = struct.Struct(
                f"<QQ{FABRIC_HANDLE_BYTES}s" if use_fabric else "<QQ"
            )
            input_struct = struct.Struct(f"<QQ{MAX_CHUNKS_PER_INPUT}Q")
            base_offset = header_struct.size
            input_offset = base_offset + MAX_VMM_BASES * base_struct.size
            payload_size = input_offset + new_count * input_struct.size
            local_payload = bytearray(payload_size)

            header_struct.pack_into(local_payload, 0, num_bases, new_count)
            for i, (base_ptr, alloc_size) in enumerate(bases_info):
                if use_fabric:
                    base_struct.pack_into(
                        local_payload,
                        base_offset + i * base_struct.size,
                        int(base_ptr),
                        int(alloc_size),
                        local_fabric_handles[i],
                    )
                else:
                    base_struct.pack_into(
                        local_payload,
                        base_offset + i * base_struct.size,
                        int(base_ptr),
                        int(alloc_size),
                    )
            for i, (chunks, offset) in enumerate(
                zip(local_input_chunks, input_offsets)
            ):
                padded_chunks = chunks + [0] * (MAX_CHUNKS_PER_INPUT - len(chunks))
                input_struct.pack_into(
                    local_payload,
                    input_offset + i * input_struct.size,
                    int(offset),
                    len(chunks),
                    *padded_chunks,
                )

            in_buf = torch.frombuffer(local_payload, dtype=torch.uint8).clone()
            gather_list = [torch.empty_like(in_buf) for _ in range(self.world_size)]
            dist.all_gather(gather_list, in_buf, group=self.group)

            all_base_payload = []
            all_input_chunks = []
            all_input_offsets = []
            for rank, gathered in enumerate(gather_list):
                payload = gathered.numpy().tobytes()
                peer_num_bases, peer_new_count = header_struct.unpack_from(payload, 0)
                if peer_new_count != new_count:
                    raise RuntimeError(
                        "Mismatched graph input count across ranks: "
                        f"rank {rank} has {peer_new_count}, expected {new_count}"
                    )

                peer_bases = []
                for i in range(peer_num_bases):
                    if use_fabric:
                        base_ptr, alloc_size, fabric_handle = base_struct.unpack_from(
                            payload, base_offset + i * base_struct.size
                        )
                    else:
                        base_ptr, alloc_size = base_struct.unpack_from(
                            payload, base_offset + i * base_struct.size
                        )
                        fabric_handle = None
                    peer_bases.append((base_ptr, fabric_handle, alloc_size))

                peer_chunks = []
                peer_offsets = []
                for i in range(new_count):
                    unpacked = input_struct.unpack_from(
                        payload, input_offset + i * input_struct.size
                    )
                    offset, chunk_count, *chunks = unpacked
                    peer_offsets.append(offset)
                    peer_chunks.append(list(chunks[:chunk_count]))

                all_base_payload.append(peer_bases)
                all_input_chunks.append(peer_chunks)
                all_input_offsets.append(peer_offsets)

            posix_peer_fds = {}
            if not use_fabric:
                posix_peer_fds = exchange_posix_fds(
                    self.group,
                    self.rank,
                    self.world_size,
                    local_posix_fds,
                    [len(peer_bases) for peer_bases in all_base_payload],
                )

            # Import + map peer allocations. Individual base mappings are kept for
            # single-chunk inputs; span mappings reserve a contiguous VA range and
            # map each chunk at its original relative offset.
            peer_base_va = {}  # (rank, base_idx) -> local VA
            peer_span_va = {}  # (rank, chunk_indices...) -> (local VA, peer base)
            new_mappings = []

            try:
                for peer_rank in range(self.world_size):
                    if peer_rank == self.rank:
                        for idx, (bp, _) in enumerate(bases_info):
                            peer_base_va[(peer_rank, idx)] = int(bp)
                        continue

                    peer_bases = all_base_payload[peer_rank]
                    for idx, (_, fb, alloc_size) in enumerate(peer_bases):
                        fd = None if use_fabric else posix_peer_fds[(peer_rank, idx)]
                        va = import_and_map_alloc(
                            fb,
                            fd,
                            alloc_size,
                            device_id,
                            use_fabric=use_fabric,
                            peer_rank=peer_rank,
                        )
                        peer_base_va[(peer_rank, idx)] = va
                        new_mappings.append((va, alloc_size, [(0, alloc_size)]))

                # Build per-input peer VA lists and register.
                peer_ptrs = []
                for j in range(new_count):
                    ptrs_j = []
                    for rank in range(self.world_size):
                        chunks = all_input_chunks[rank][j]
                        off = all_input_offsets[rank][j]
                        if len(chunks) == 1:
                            ptrs_j.append(peer_base_va[(rank, chunks[0])] + off)
                            continue

                        span_key = (rank, *chunks)
                        if span_key not in peer_span_va:
                            peer_bases = all_base_payload[rank]
                            first_base = peer_bases[chunks[0]][0]
                            last_base, _, last_size = peer_bases[chunks[-1]]
                            span_size = (
                                int(last_base) + int(last_size) - int(first_base)
                            )
                            if rank == self.rank:
                                span_va = int(first_base)
                            else:
                                span_va = check_drv(
                                    drv.cuMemAddressReserve(span_size, 0, 0, 0),
                                    "cuMemAddressReserve(span)",
                                )
                                mapped_chunks = []
                                for chunk_idx in chunks:
                                    base_ptr, fb, alloc_size = peer_bases[chunk_idx]
                                    rel = int(base_ptr) - int(first_base)
                                    fd = (
                                        None
                                        if use_fabric
                                        else posix_peer_fds[(rank, chunk_idx)]
                                    )
                                    map_chunk_into_span(
                                        fb,
                                        fd,
                                        span_va,
                                        rel,
                                        int(alloc_size),
                                        device_id,
                                        use_fabric=use_fabric,
                                        peer_rank=rank,
                                    )
                                    mapped_chunks.append((rel, int(alloc_size)))
                                new_mappings.append(
                                    (int(span_va), span_size, mapped_chunks)
                                )
                            peer_span_va[span_key] = (int(span_va), int(first_base))

                        span_va, _ = peer_span_va[span_key]
                        ptrs_j.append(span_va + off)
                    peer_ptrs.append(ptrs_j)

                self.obj.register_peer_mapped_inputs(peer_ptrs)
                self._peer_mappings.extend(new_mappings)
            except Exception:
                release_mappings(new_mappings)
                raise
            finally:
                for fd in posix_peer_fds.values():
                    os.close(fd)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            transport = "FABRIC" if use_fabric else "POSIX fd"
            log_info_on_rank0(
                logger,
                f"Registered {new_count} cuda graph addresses via "
                f"{transport} handles ({num_bases} unique allocations) "
                f"in {elapsed_ms:.1f} ms",
            )
        finally:
            for fd in local_posix_fds:
                os.close(fd)
            for h in retained_handles:
                check_drv(drv.cuMemRelease(h), "cuMemRelease(retained)")

    def close(self):
        if not self._peer_mappings:
            return
        release_mappings(self._peer_mappings)
