from __future__ import annotations

import ctypes
import logging
import os
import tempfile
from math import prod
from typing import TYPE_CHECKING, List, Optional, Sequence

import torch
import torch.utils.cpp_extension
from torch.cuda.memory import CUDAPluggableAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KvBufferDesc

logger = logging.getLogger(__name__)

_drv = None


def _driver():
    global _drv
    if _drv is None:
        from cuda.bindings import driver

        _drv = driver
    return _drv


def _check(result, label: str):
    drv = _driver()
    err = result[0] if isinstance(result, tuple) else result
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label} failed: {err}")
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def query_granularity(device_id: int) -> int:
    """Minimum CUDA virtual-memory allocation granularity (bytes) for ``device_id``."""
    drv = _driver()
    prop = drv.CUmemAllocationProp()
    prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = int(device_id)
    return int(
        _check(
            drv.cuMemGetAllocationGranularity(
                prop,
                drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "cuMemGetAllocationGranularity",
        )
    )


# Bump allocator: hands back base+cursor, bounded by the RESERVED size (not the
# committed watermark) so upper-bound tensors can be allocated before physical
# commit. Allocations are granularity-aligned so each pointer can be committed at
# its own VA range (cuMemMap requires it; GB300 rejects partial-handle maps).
# Symbols are SUFFIXED per arena instance and each instance loads its own .so, so
# multiple arenas per process (hybrid-SWA: full + swa) don't clobber each other.
def _stub_source(sfx: str) -> str:
    return f"""
#include <cstddef>
#include <cstdint>
#include <mutex>
extern "C" {{
static uintptr_t g_base = 0;
static size_t g_cursor = 0;
static size_t g_reserved = 0;
static size_t g_align = 512;
static std::mutex g_mu;
static size_t align_up(size_t v, size_t a){{ return (v + a - 1) / a * a; }}
void kvarena_set_base_{sfx}(uintptr_t b){{ std::lock_guard<std::mutex> lk(g_mu); g_base=b; g_cursor=0; }}
void kvarena_set_reserved_{sfx}(size_t r){{ std::lock_guard<std::mutex> lk(g_mu); g_reserved=r; }}
void kvarena_set_align_{sfx}(size_t a){{ std::lock_guard<std::mutex> lk(g_mu); if (a) g_align=a; }}
size_t kvarena_cursor_{sfx}(void){{ std::lock_guard<std::mutex> lk(g_mu); return g_cursor; }}
void* kvarena_malloc_{sfx}(size_t size, int device, void* stream){{
  std::lock_guard<std::mutex> lk(g_mu);
  size_t need = g_cursor + align_up(size, g_align);
  if (need > g_reserved) return 0;   // never exceed the reserved VA range
  void* p = reinterpret_cast<void*>(g_base + g_cursor);
  g_cursor = need;
  return p;
}}
void kvarena_free_{sfx}(void* ptr, size_t size, int device, void* stream){{}}
}}
"""


_DEFAULT_RESERVE_BYTES = 256 * (1024**3)  # 256 GiB virtual; free until committed


class KvVmmArena:
    """One device's CUDA virtual-memory reservation exposed as a ``torch.cuda.MemPool``."""

    # Per-instance suffix source -> isolated allocator symbols/state (see _stub_source).
    _instance_count = 0

    def __init__(self, device_id: int, reserve_bytes: int = _DEFAULT_RESERVE_BYTES):
        self.device_id = int(device_id)
        self._sfx = str(KvVmmArena._instance_count)
        KvVmmArena._instance_count += 1
        drv = _driver()
        with torch.cuda.device(self.device_id):
            _check(drv.cuInit(0), "cuInit")
            self._prop = drv.CUmemAllocationProp()
            self._prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            self._prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            self._prop.location.id = self.device_id
            self.granularity = query_granularity(self.device_id)
            self._access = drv.CUmemAccessDesc()
            self._access.location.type = (
                drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            )
            self._access.location.id = self.device_id
            self._access.flags = (
                drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            )

            self.reserved = self._align(reserve_bytes)
            # Align the base to granularity so base + (granularity-aligned cursor) is
            # always a valid cuMemMap address for per-buffer commit_range().
            self.base = int(
                _check(
                    drv.cuMemAddressReserve(self.reserved, self.granularity, 0, 0),
                    "cuMemAddressReserve",
                )
            )
            # commit_range bookkeeping: mapped VA -> (size, handle); committed bytes per offset.
            self._ranges = {}
            self._committed_by_offset = {}
            self._range_backed = 0
            self._closed = False

        self._lib = self._build_stub()
        self._fn_set_base(ctypes.c_void_p(self.base))
        self._fn_set_reserved(ctypes.c_size_t(self.reserved))
        self._fn_set_align(ctypes.c_size_t(self.granularity))
        self._allocator = CUDAPluggableAllocator(
            self._so_path, f"kvarena_malloc_{self._sfx}", f"kvarena_free_{self._sfx}"
        ).allocator()
        # no_split so the caching allocator hands our bump pointers back verbatim.
        self.pool = torch.cuda.MemPool(self._allocator, no_split=True)
        logger.info(
            "KvVmmArena[%s] ready: device=%d reserved=%.1f GiB granularity=%d KiB",
            self._sfx,
            self.device_id,
            self.reserved / (1024**3),
            self.granularity // 1024,
        )

    def _align(self, v: int) -> int:
        return align_up(v, self.granularity)

    def _build_stub(self) -> ctypes.CDLL:
        out_dir = os.path.join(tempfile.gettempdir(), "sgl_kv_vmm_arena")
        os.makedirs(out_dir, exist_ok=True)
        libname = f"sgl_kv_vmm_arena_stub_{self._sfx}"
        torch.utils.cpp_extension.load_inline(
            name=libname,
            cpp_sources=_stub_source(self._sfx),
            with_cuda=False,  # pure arithmetic — no nvcc, no CUDA headers
            is_python_module=False,
            verbose=False,
            build_directory=out_dir,
        )
        self._so_path = f"{out_dir}/{libname}.so"
        lib = ctypes.CDLL(self._so_path)
        self._fn_set_base = getattr(lib, f"kvarena_set_base_{self._sfx}")
        self._fn_set_base.argtypes = [ctypes.c_void_p]
        self._fn_set_base.restype = None
        self._fn_set_reserved = getattr(lib, f"kvarena_set_reserved_{self._sfx}")
        self._fn_set_reserved.argtypes = [ctypes.c_size_t]
        self._fn_set_reserved.restype = None
        self._fn_set_align = getattr(lib, f"kvarena_set_align_{self._sfx}")
        self._fn_set_align.argtypes = [ctypes.c_size_t]
        self._fn_set_align.restype = None
        self._fn_cursor = getattr(lib, f"kvarena_cursor_{self._sfx}")
        self._fn_cursor.argtypes = []
        self._fn_cursor.restype = ctypes.c_size_t
        return lib

    def commit_range(self, offset: int, want_bytes: int) -> None:
        """Back ``[base+offset, base+offset+want_bytes)`` (monotonic per offset).
        ``offset`` must be granularity-aligned (the bump allocator guarantees it).
        Maps one full handle per extension -- GB300 rejects partial-handle maps."""
        if self._closed:
            raise RuntimeError("KvVmmArena.commit_range after close")
        if offset % self.granularity != 0:
            raise ValueError(
                f"commit_range offset {offset} not granularity-aligned "
                f"({self.granularity})"
            )
        want = self._align(int(want_bytes))
        prev = self._committed_by_offset.get(offset, 0)
        if want <= prev:
            return
        if offset + want > self.reserved:
            raise RuntimeError(
                f"commit_range [{offset}, {offset + want}) exceeds reservation "
                f"{self.reserved}"
            )
        drv = _driver()
        add = want - prev
        addr = self.base + offset + prev
        with torch.cuda.device(self.device_id):
            handle = _check(drv.cuMemCreate(add, self._prop, 0), "cuMemCreate")
            try:
                _check(drv.cuMemMap(addr, add, 0, handle, 0), "cuMemMap")
                _check(
                    drv.cuMemSetAccess(addr, add, [self._access], 1), "cuMemSetAccess"
                )
            except Exception:
                # Roll back this failed extension; leave already-mapped ranges intact.
                unmap = drv.cuMemUnmap(addr, add)
                unmap = unmap[0] if isinstance(unmap, tuple) else unmap
                rel = drv.cuMemRelease(handle)
                rel = rel[0] if isinstance(rel, tuple) else rel
                raise
        self._ranges[addr] = (add, handle)
        self._committed_by_offset[offset] = want
        self._range_backed += add

    @property
    def backed_bytes(self) -> int:
        """Total physically-backed bytes (sum of scattered per-buffer ranges)."""
        return self._range_backed

    @property
    def cursor_bytes(self) -> int:
        return int(self._fn_cursor())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        drv = _driver()
        try:
            torch.cuda.synchronize()
        except Exception as e:  # pragma: no cover
            logger.warning("KvVmmArena.close synchronize failed: %s", e)
        for addr, (size, handle) in self._ranges.items():
            err = drv.cuMemUnmap(addr, size)
            err = err[0] if isinstance(err, tuple) else err
            if err != drv.CUresult.CUDA_SUCCESS:
                logger.warning("cuMemUnmap range -> %s", err)
            err = drv.cuMemRelease(handle)
            err = err[0] if isinstance(err, tuple) else err
            if err != drv.CUresult.CUDA_SUCCESS:
                logger.warning("cuMemRelease range -> %s", err)
        self._ranges.clear()
        err = drv.cuMemAddressFree(self.base, self.reserved)
        err = err[0] if isinstance(err, tuple) else err
        if err != drv.CUresult.CUDA_SUCCESS:
            logger.warning("cuMemAddressFree -> %s", err)


# torch's caching allocator hands the pluggable allocator whole large-pool segments
# (rounded up to >= ~20 MiB) per tensor, so reserve slack beyond the tight tensor sum.
# VA is free until committed, so this costs only address space, not GPU memory.
_PER_BUFFER_VA_SLACK = 32 << 20


class _BufferSpec:
    """Per-buffer placement + backing state inside the shared VA reservation."""

    __slots__ = ("desc", "offset", "reserved_span", "aligned_reserved", "backed_to")

    def __init__(
        self,
        desc: KvBufferDesc,
        offset: int,
        reserved_span: int,
        aligned_reserved: int,
    ):
        self.desc = desc
        self.offset = offset  # granularity-aligned arena offset of this buffer
        self.reserved_span = reserved_span  # logical (unaligned) tensor bytes
        self.aligned_reserved = aligned_reserved  # reserved span rounded to granularity
        self.backed_to = 0  # bytes from offset currently backed


class KvVmmBufferOwner:
    """Owns one ``KvVmmArena`` plus its incrementally-backed KV buffers.

    ``buffer_descs`` is an ordered list of ``KvBufferDesc``; the created ``torch.empty``
    tensors are exposed in the same order as ``self.tensors``.
    """

    def __init__(
        self,
        *,
        device: str,
        device_id: int,
        store_dtype: torch.dtype,
        page_size: int,
        reserved_num_tokens: int,
        buffer_descs: Sequence[KvBufferDesc],
    ):
        self.device = device
        self.device_id = int(device_id)
        self.store_dtype = store_dtype
        self.page_size = int(page_size)
        self._reserved_num_tokens = int(reserved_num_tokens)
        self._final_num_tokens: Optional[int] = None
        self._arena: Optional[KvVmmArena] = None
        self._specs: List[_BufferSpec] = []
        self.tensors: List[torch.Tensor] = []

        itemsize = store_dtype.itemsize
        with torch.cuda.device(self.device_id):
            gran = query_granularity(self.device_id)
            reserved_spans = [d.reserved_span_bytes(itemsize) for d in buffer_descs]
            aligned = [align_up(s, gran) for s in reserved_spans]
            reserve_bytes = sum(a + _PER_BUFFER_VA_SLACK for a in aligned) + gran
            self._arena = KvVmmArena(self.device_id, reserve_bytes=reserve_bytes)
            assert self._arena.granularity == gran, (self._arena.granularity, gran)

            # NORMAL torch tensors through the arena MemPool; torch.empty never touches
            # the unbacked tail.
            with torch.cuda.use_mem_pool(self._arena.pool):
                self.tensors = [
                    torch.empty(d.shape, dtype=store_dtype, device=self.device)
                    for d in buffer_descs
                ]

            specs: List[_BufferSpec] = []
            for desc, tensor, reserved_span, aligned_reserved in zip(
                buffer_descs, self.tensors, reserved_spans, aligned
            ):
                if prod(tensor.shape) * itemsize != reserved_span:
                    raise RuntimeError(
                        f"buffer {desc.name!r} tensor bytes "
                        f"{prod(tensor.shape) * itemsize} != reserved span {reserved_span}"
                    )
                offset = tensor.data_ptr() - self._arena.base
                if offset < 0 or offset % gran != 0:
                    raise RuntimeError(
                        f"buffer {desc.name!r} arena offset {offset} not "
                        f"granularity-aligned ({gran})"
                    )
                if offset + aligned_reserved > self._arena.reserved:
                    raise RuntimeError(
                        f"buffer {desc.name!r} [{offset}, {offset + aligned_reserved}) "
                        f"exceeds reservation {self._arena.reserved}"
                    )
                specs.append(_BufferSpec(desc, offset, reserved_span, aligned_reserved))
            self._specs = specs

            # Back one page so slot 0 is resident before capture: capture routes every
            # dummy KV write to slot 0 (out_cache_loc is zeros). finalize() backs the rest.
            self.ensure_prefix(self.page_size)

        for t in self.tensors:
            assert (
                t.is_cuda and t.device.index == self.device_id
            ), f"post-capture KV buffer landed on {t.device}, expected cuda:{self.device_id}"

    # -- backing --------------------------------------------------------------

    @staticmethod
    def _check_span(spec: _BufferSpec, span: int) -> int:
        """Return ``span`` if it fits ``[0, reserved_span]``; raise otherwise."""
        span = int(span)
        if not (0 <= span <= spec.reserved_span):
            raise ValueError(
                f"buffer {spec.desc.name!r}: span {span} outside "
                f"[0, {spec.reserved_span}] (reserved tensor bytes)"
            )
        return span

    def _back_spans(self, span_bytes: Sequence[int]) -> None:
        """Back each buffer to (at least) ``span_bytes[i]``. An out-of-reservation
        span is a descriptor bug: raise before committing anything, never clamp."""
        if self._arena is None:
            raise RuntimeError("backing after close / before construction")
        for spec, span in zip(self._specs, span_bytes):
            self._check_span(spec, span)
        gran = self._arena.granularity
        for spec, span in zip(self._specs, span_bytes):
            want = align_up(
                int(span), gran
            )  # <= aligned_reserved since span <= reserved
            if want > spec.backed_to:
                self._arena.commit_range(spec.offset, want)
                spec.backed_to = want

    def ensure_prefix(self, num_tokens: int) -> None:
        """Ensure the first ``num_tokens`` slots of every buffer are physically backed."""
        self._back_spans(
            [s.desc.prefix_span_bytes(num_tokens, self.page_size) for s in self._specs]
        )

    def finalize(self, final_num_tokens: int) -> None:
        """Back each buffer's final advertised span; set the final serving capacity."""
        final = int(final_num_tokens)
        if not (self.page_size <= final <= self._reserved_num_tokens):
            raise ValueError(
                f"final_num_tokens={final} must satisfy page_size="
                f"{self.page_size} <= final <= reserved={self._reserved_num_tokens}"
            )
        self._back_spans(
            [s.desc.final_span_bytes(final, self.page_size) for s in self._specs]
        )
        self._final_num_tokens = final

    # -- accessors / teardown -------------------------------------------------

    @property
    def backed_bytes(self) -> int:
        return self._arena.backed_bytes if self._arena is not None else 0

    def close(self) -> None:
        self.tensors = []
        self._specs = []
        if self._arena is not None:
            self._arena.close()
            self._arena = None
