"""Host storage for offloaded PLE tables.

``pinned`` (default)
    ``torch.empty(..., pin_memory=True)``. On a discrete GPU this frees VRAM.

``file``
    A shared mmap of a sparse file under ``--ple-offload-dir``. GPUs with
    host-page-table access read it directly; other CUDA GPUs stage rows
    through pinned buffers. ``PleFileRssTrimmer`` bounds mapping residency.

This module has no Triton or CUDA-kernel imports so that its allocator and
prefetcher can be unit-tested on CPU.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import numpy as np
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LIBC: Optional[ctypes.CDLL] = None
_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ")
_SMAPS_RSS = re.compile(r"^Rss:\s+(\d+) kB")

PLE_OFFLOAD_BACKENDS = ("pinned", "file")

# cudaDeviceAttr enum values (cuda_runtime_api.h).
_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
_MADV_RANDOM = 1
_MADV_DONTNEED = 4
_PAGE_SHIFT = 12
# One MADV_DONTNEED call takes mmap_lock for its whole range; over the full
# 47.7 GiB table that is ~3.5 s during which every fault in the process --
# including the ones the gather kernel takes -- stalls. Trim in slices.
PLE_FILE_RSS_TRIM_CHUNK_BYTES = 1 << 30
# Below this many rows a gather is decode-sized (16 rows per token): the page
# faults are cheap and the host-side hint would cost more than it saves.
PLE_FILE_PREFETCH_MIN_ROWS = 2048


class PleFilePageHints:
    """Issue page-cache hints on the calling thread."""

    def __init__(self, path: str, row_bytes: int):
        self._fd = os.open(path, os.O_RDONLY)
        self._row_bytes = int(row_bytes)
        self._advise_failed = False

    @staticmethod
    def pages_for_rows(row_ids: torch.Tensor | np.ndarray, row_bytes: int) -> list[int]:
        ids = np.asarray(row_ids, dtype=np.int64)
        starts = ids * row_bytes
        page_size = 1 << _PAGE_SHIFT
        return np.unique(
            np.concatenate(
                [
                    (starts + offset) >> _PAGE_SHIFT
                    for offset in range(0, row_bytes, page_size)
                ]
                + [(starts + row_bytes - 1) >> _PAGE_SHIFT]
            )
        ).tolist()

    def advise_rows(self, row_ids) -> None:
        self._advise(self.pages_for_rows(row_ids, self._row_bytes))

    def _advise(self, pages: list[int]) -> None:
        if self._advise_failed:
            return
        for p in pages:
            try:
                os.posix_fadvise(
                    self._fd, p << _PAGE_SHIFT, 1 << _PAGE_SHIFT, os.POSIX_FADV_WILLNEED
                )
            except OSError as exc:
                self._advise_failed = True
                logger.warning(
                    "PLE table: posix_fadvise failed (%s); page hints off", exc
                )
                return

    def close(self) -> None:
        try:
            os.close(self._fd)
        except OSError:
            pass


class PleFilePrefetcher(PleFilePageHints):
    """Queue page hints for direct GPU prefill reads; skip decode and capture."""

    def __init__(
        self, path: str, row_bytes: int, min_rows: int = PLE_FILE_PREFETCH_MIN_ROWS
    ):
        super().__init__(path, row_bytes)
        self._min_rows = int(min_rows)
        self._pool = ThreadPoolExecutor(max_workers=1)

    def enqueue(
        self,
        flat_ids: torch.Tensor,
        *,
        vocab_start: int = 0,
        vocab_end: Optional[int] = None,
    ) -> bool:
        """Queue the hint for ``flat_ids``. Returns whether anything was queued."""
        if flat_ids.numel() < self._min_rows:
            return False
        if flat_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            return False
        # The .cpu() syncs the stream; acceptable for prefill chunks (~1 s) and
        # it is what lets the page set be computed without touching the kernel.
        row_ids = flat_ids.detach().cpu()
        if vocab_end is not None:
            # The file contains only this rank's vocabulary shard.
            row_ids = row_ids[(row_ids >= vocab_start) & (row_ids < vocab_end)]
        row_ids = row_ids - vocab_start
        if row_ids.numel() == 0:
            return False
        pages = self.pages_for_rows(row_ids, self._row_bytes)
        self._pool.submit(self._advise, pages)
        return True

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        super().close()


class PleFileRssTrimmer:
    """Keep the mapped table's resident set under a budget.

    Every random row fault maps in a whole page-cache folio, so with large
    folios (Linux 6.x) the mapping's Rss climbs towards the table's full size
    while a generated token only reads a few KB of it: measured ~45 KB of Rss
    growth per token on a GB10. On a unified-memory part that is not a slow
    leak, it is a countdown -- the free-memory readings that size the KV pool
    come from the same pool the folios are accumulating in.

    ``MADV_RANDOM`` does not prevent it (it limits readahead I/O, not the
    mapping-in of folios already in cache) and ``posix_fadvise(DONTNEED)`` does
    not release them either. ``MADV_DONTNEED`` over the mapping does: the page
    table entries go, the pages stay in the page cache, and hot rows come back
    at minor-fault cost.

    Dropping entries under a running gather is the state this backend already
    handles: the file starts out entirely unfaulted and every cold row is
    faulted in from inside the kernel through the same host page tables. What
    must not happen is one ``madvise`` call over the whole table, so the trim
    is chunked (see ``PLE_FILE_RSS_TRIM_CHUNK_BYTES``) and runs on its own
    daemon thread -- decode replays a CUDA graph and executes no Python, so a
    hook in the gather would never fire in the phase that grows the table.
    """

    def __init__(
        self,
        addr: int,
        nbytes: int,
        budget_bytes: int,
        interval_s: float,
        chunk_bytes: int = PLE_FILE_RSS_TRIM_CHUNK_BYTES,
    ) -> None:
        self._addr = int(addr)
        self._nbytes = int(nbytes)
        self._budget = int(budget_bytes)
        self._interval = float(interval_s)
        self._chunk = int(chunk_bytes)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="ple-file-rss-trim", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def mapping_rss_bytes(self) -> Optional[int]:
        """Resident bytes of the VMAs backing the table, or None off Linux."""
        return _mapping_rss_bytes(self._addr, self._nbytes)

    def trim_once(self) -> int:
        """Drop the mapping's resident pages if over budget. Returns bytes freed."""
        before = self.mapping_rss_bytes()
        if before is None or before <= self._budget:
            return 0
        for offset in range(0, self._nbytes, self._chunk):
            if self._stop.is_set():
                break
            length = min(self._chunk, self._nbytes - offset)
            if not _madvise(self._addr + offset, length, _MADV_DONTNEED):
                return 0
            # Let the faults that queued behind mmap_lock through.
            self._stop.wait(0.005)
        after = self.mapping_rss_bytes()
        freed = before - after if after is not None else 0
        logger.info(
            "PLE table: trimmed resident set %.1f -> %.1f GiB (budget %.1f GiB)",
            before / 2**30,
            (after if after is not None else 0) / 2**30,
            self._budget / 2**30,
        )
        return max(freed, 0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.trim_once()
            except Exception as exc:  # advisory only; never fail a request
                logger.warning("PLE table: resident-set trim skipped (%s)", exc)

    def close(self) -> None:
        self._stop.set()


def allocate_ple_host_table(
    shape: Sequence[int],
    dtype: torch.dtype,
    backend: str = "pinned",
    table_dir: Optional[str] = None,
    tag: Optional[str] = None,
) -> torch.Tensor:
    """Return a host tensor of ``shape``/``dtype`` for the PLE table.

    For the file backend, ``table_dir`` should be private to one checkpoint
    (the server defaults it to ``$SGLANG_CACHE_DIR/ple/<model path>``): the
    file name only encodes shape, dtype and ``tag``, and every boot rewrites
    the whole table through the weight loader.
    """
    if backend not in PLE_OFFLOAD_BACKENDS:
        raise ValueError(
            f"unknown PLE offload backend {backend!r}; choose from {PLE_OFFLOAD_BACKENDS}"
        )
    if backend == "pinned":
        return torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=True)

    numel = 1
    for d in shape:
        numel *= int(d)
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    table_dir = os.path.expanduser(table_dir or envs.SGLANG_QWEN4_PLE_FILE_DIR.get())
    os.makedirs(table_dir, exist_ok=True)
    path = os.path.join(table_dir, ple_table_file_name(shape, dtype, tag))
    if not os.path.exists(path) or os.path.getsize(path) != nbytes:
        # Sparse: only pages that get written take disk space.
        with open(path, "wb") as f:
            f.truncate(nbytes)
    logger.info(
        "PLE table: file-backed mmap %s (%.1f GiB, %s)", path, nbytes / 2**30, dtype
    )
    storage = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    _madvise_random(storage, nbytes)
    table = storage.view(dtype).view(*[int(d) for d in shape])
    table._sglang_ple_file_path = path
    return table


def make_ple_file_prefetcher(table: torch.Tensor) -> Optional[PleFilePrefetcher]:
    """A prefetcher for a table returned by ``allocate_ple_host_table(..., "file")``."""
    path = getattr(table, "_sglang_ple_file_path", None)
    if path is None or not envs.SGLANG_QWEN4_PLE_FILE_PREFETCH.get():
        return None
    row_bytes = (
        int(table.shape[-1]) * table.element_size()
        if table.dim() >= 2
        else table.element_size()
    )
    prefetcher = PleFilePrefetcher(path=path, row_bytes=row_bytes)
    logger.info(
        "PLE table: WILLNEED prefetch on for gathers of >= %d rows (row = %d B)",
        PLE_FILE_PREFETCH_MIN_ROWS,
        row_bytes,
    )
    return prefetcher


def make_ple_file_rss_trimmer(table: torch.Tensor) -> Optional[PleFileRssTrimmer]:
    """A started trimmer for a table from ``allocate_ple_host_table(..., "file")``.

    ``SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=0`` turns it off; it is also absent
    where the resident set cannot be read (no ``/proc/self/smaps``).
    """
    path = getattr(table, "_sglang_ple_file_path", None)
    if path is None:
        return None
    budget_gb = float(envs.SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB.get())
    if budget_gb <= 0:
        return None
    nbytes = table.numel() * table.element_size()
    if _mapping_rss_bytes(table.data_ptr(), nbytes) is None:
        logger.warning(
            "PLE table: resident-set trim off, /proc/self/smaps is not readable; "
            "the mapping will creep towards %.1f GiB resident",
            nbytes / 2**30,
        )
        return None
    trimmer = PleFileRssTrimmer(
        addr=table.data_ptr(),
        nbytes=nbytes,
        budget_bytes=int(budget_gb * 2**30),
        interval_s=float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    trimmer.start()
    logger.info(
        "PLE table: resident set capped at %.1f GiB, checked every %.0f s",
        budget_gb,
        float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    return trimmer


def default_ple_table_dir(model_path: str) -> str:
    """``$SGLANG_QWEN4_PLE_FILE_DIR/<model path>``, one directory per checkpoint."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_path).rstrip("/")).strip("_")
    return os.path.join(envs.SGLANG_QWEN4_PLE_FILE_DIR.get(), safe or "model")


def ple_table_file_name(
    shape: Sequence[int], dtype: torch.dtype, tag: Optional[str] = None
) -> str:
    """Deterministic file name so the sparse table is reused across restarts.

    ``tag`` distinguishes tables of the same shape that must not share a file,
    e.g. the vocabulary shards of different tensor-parallel ranks.
    """
    numel = 1
    for d in shape:
        numel *= int(d)
    elem = torch.empty(0, dtype=dtype).element_size()
    dims = "x".join(str(int(d)) for d in shape)
    suffix = f"_{tag}" if tag else ""
    return f"ple_table_{dims}_{str(dtype).replace('torch.', '')}_{numel * elem}B{suffix}.bin"


_HOST_PAGE_TABLES: dict[int, Optional[bool]] = {}


def device_uses_host_page_tables(device_index: int = 0) -> Optional[bool]:
    """Whether pageable host memory is directly addressable by the GPU.

    Returns None when the CUDA runtime library cannot be queried.
    """
    if device_index in _HOST_PAGE_TABLES:
        return _HOST_PAGE_TABLES[device_index]
    candidates = [ctypes.util.find_library("cudart")]
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        candidates += sorted(
            os.path.join(torch_lib, f)
            for f in os.listdir(torch_lib)
            if f.startswith("libcudart.so")
        )
    try:
        import nvidia.cuda_runtime  # type: ignore

        nv_lib = os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib")
        if os.path.isdir(nv_lib):
            candidates += sorted(
                os.path.join(nv_lib, f)
                for f in os.listdir(nv_lib)
                if f.startswith("libcudart.so")
            )
    except Exception:
        pass
    for name in [c for c in candidates if c]:
        try:
            cudart = ctypes.CDLL(name)
            value = ctypes.c_int()
            rc = cudart.cudaDeviceGetAttribute(
                ctypes.byref(value),
                ctypes.c_int(
                    _CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
                ),
                ctypes.c_int(device_index),
            )
            if rc == 0:
                result = bool(value.value)
                _HOST_PAGE_TABLES[device_index] = result
                return result
        except OSError:
            continue
    _HOST_PAGE_TABLES[device_index] = None
    return None


def _madvise_random(storage: torch.Tensor, nbytes: int) -> None:
    """The table is pure random access (16 rows of 160 B per token). Without
    this the kernel's readahead pulls its whole window: measured 1.4 MB of disk
    per token, ~560x the bytes actually used.

    It bounds readahead I/O only. Folios that are already in the page cache are
    still mapped in whole on a fault, which is what ``PleFileRssTrimmer``
    exists for."""
    if not _madvise(storage.data_ptr(), nbytes, _MADV_RANDOM):
        logger.warning("PLE table: madvise(MADV_RANDOM) not applied")


def _libc() -> Optional[ctypes.CDLL]:
    global _LIBC
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL(
                ctypes.util.find_library("c") or "libc.so.6", use_errno=True
            )
        except OSError:
            return None
    return _LIBC


def _madvise(addr: int, length: int, advice: int) -> bool:
    """``madvise(2)`` on our own mapping. Advisory: never affects correctness."""
    libc = _libc()
    if libc is None:
        return False
    try:
        rc = libc.madvise(
            ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(advice)
        )
    except Exception:
        return False
    if rc != 0:
        logger.warning(
            "PLE table: madvise(advice=%d) failed (errno %d)",
            advice,
            ctypes.get_errno(),
        )
        return False
    return True


def _mapping_rss_bytes(
    addr: int, nbytes: int, smaps_path: str = "/proc/self/smaps"
) -> Optional[int]:
    """Resident bytes of the VMAs overlapping ``[addr, addr + nbytes)``.

    Summed per mapping rather than taken from ``statm``/``smaps_rollup``: only
    the table's own residency should drive the trim, and on a unified-memory
    box the process RSS is dominated by everything else.
    """
    lo, hi = int(addr), int(addr) + int(nbytes)
    total = 0
    overlapping = False
    try:
        with open(smaps_path, "r") as f:
            for line in f:
                header = _SMAPS_HEADER.match(line)
                if header is not None:
                    start = int(header.group(1), 16)
                    end = int(header.group(2), 16)
                    overlapping = start < hi and end > lo
                elif overlapping:
                    rss = _SMAPS_RSS.match(line)
                    if rss is not None:
                        total += int(rss.group(1)) * 1024
    except OSError:
        return None
    return total
