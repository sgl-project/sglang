"""Read the part of a mapped checkpoint tensor that a rank copies with ``pread``.

``safe_open`` yields views of a private mapping of each safetensors file, and a
host-to-device copy out of such a view runs at a few GB/s, against 28-30 GB/s
out of ordinary host memory filled with ``pread``. Under tensor parallelism a
rank copies only its slice of a tensor:

- A slice that is a contiguous range of rows (output-dim shards such as MoE w13
  and column-parallel weights) is read with ``pread``: only its own bytes.
- Any other slice (column shards such as MoE w2) stays a view of the mapping
  while the checkpoint is prefetched into the page cache, so only the slice's
  bytes are copied. Without prefetch, its whole byte span is read once with
  ``pread``, and the node's other ranks then find it in the page cache.

Only the loading thread reads: loaders that already copy weights from a pool of
threads (``maybe_executor_submit``) hide the mapping's slow copies behind each
other, and are limited by the GIL, which faster copies only contend for more.

Rows are read with ``pread`` only when the page cache holds them (``mincore`` on
the mapping): on a network filesystem, small reads of pages that are not cached
are much slower than the mapping's faults, which read ahead in large blocks.

The bytes are the file's bytes either way. A view is read only while the tensor
it was cut from is alive (so the mapping is too) and has not been written in
place since it was yielded; anything else keeps the plain copy.

Reads up to 16 MiB land in one reused (pinned, with CUDA) buffer per thread, so
loading threads do not allocate and fault in fresh host memory per tensor: a
result is valid until the same thread's next read, and callers copy it at once.
"""

import concurrent.futures
import ctypes
import mmap
import os
import threading
import weakref
from typing import Dict, NamedTuple, Optional

import torch

# Reads larger than this are split across a small shared pool of threads.
_CHUNK_BYTES = 16 << 20
_NUM_THREADS = 8


class _File:
    """An open checkpoint file, closed once no registered tensor uses it."""

    __slots__ = ("fd", "path", "prefetched", "__weakref__")

    def __init__(self, path: str, prefetched: bool):
        self.fd = os.open(path, os.O_RDONLY)
        self.path = path
        self.prefetched = prefetched

    def __del__(self):
        try:
            os.close(self.fd)
        except (AttributeError, OSError):
            pass


class _Source(NamedTuple):
    file: _File
    offset: int  # File offset of the storage's first byte.
    nbytes: int  # Bytes in the storage.
    version: int  # The tensor's version counter when it was yielded.
    ref: "weakref.ref[torch.Tensor]"


# Storage data pointer -> the mapped file region it views.
_sources: Dict[int, _Source] = {}
_lock = threading.Lock()
_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
_local = threading.local()
_PAGE = mmap.PAGESIZE
try:
    # PyDLL keeps the GIL. Neither call blocks: reads of cached pages are memory
    # copies, and handing the GIL back and forth around each one costs loaders
    # that copy weights from many threads more than the copy itself.
    _libc = ctypes.PyDLL(None)
    _mincore = _libc.mincore
    _mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    _mincore.restype = ctypes.c_int
    _cpread = _libc.pread
    _cpread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_long]
    _cpread.restype = ctypes.c_ssize_t
except (AttributeError, OSError):
    _mincore = _cpread = None


def _cached(address: int, nbytes: int) -> bool:
    """Whether the page cache holds every page of a mapped range."""
    if _mincore is None:
        return True
    start = address & ~(_PAGE - 1)
    pages = (address + nbytes - start + _PAGE - 1) // _PAGE
    vec = ctypes.create_string_buffer(pages)
    if _mincore(start, address + nbytes - start, vec) != 0:
        return True
    return b"\x00" not in vec.raw


def _staging(nbytes: int) -> torch.Tensor:
    """This thread's reused host buffer, at least ``nbytes`` long."""
    buf = getattr(_local, "buf", None)
    if buf is None or buf.numel() < nbytes:
        size = 1 << max(nbytes - 1, 1 << 20).bit_length()
        buf = torch.empty(size, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
        _local.buf = buf
    return buf[:nbytes]


def open_file(path: str, prefetched: bool) -> _File:
    """Open ``path`` for reading the tensors registered from it.

    ``prefetched``: whether the checkpoint is being read into the page cache
    while it loads, so that column shards are better copied from the mapping."""
    return _File(path, prefetched)


def register(file: _File, file_offset: int, tensor: torch.Tensor) -> None:
    """Record that ``tensor`` maps ``file`` at ``file_offset`` (its first byte)."""
    storage = tensor.untyped_storage()
    ptr, nbytes = storage.data_ptr(), storage.nbytes()
    if nbytes == 0 or tensor.device.type != "cpu":
        return
    offset = file_offset - (tensor.data_ptr() - ptr)

    def forget(ref, ptr=ptr):
        with _lock:
            source = _sources.get(ptr)
            if source is not None and source.ref is ref:
                del _sources[ptr]

    source = _Source(file, offset, nbytes, tensor._version, weakref.ref(tensor, forget))
    with _lock:
        _sources[ptr] = source


def _pread(fd: int, out: memoryview, offset: int, path: str) -> None:
    done = 0
    while done < len(out):
        n = os.preadv(fd, [out[done:]], offset + done)
        if n == 0:
            raise EOFError(f"{path} ends before byte {offset + len(out)}")
        done += n


def _read(file: _File, out: torch.Tensor, offset: int) -> None:
    """Fill the uint8 tensor ``out`` from ``file`` at ``offset``."""
    global _pool
    n = out.numel()
    if n <= _CHUNK_BYTES and _cpread is not None:
        address, done = out.data_ptr(), 0
        while done < n:
            got = _cpread(file.fd, address + done, n - done, offset + done)
            if got <= 0:
                break
            done += got
        if done == n:
            return
        out = out[done:]
        offset += done
    view = memoryview(out.numpy())
    n = len(view)
    if n <= _CHUNK_BYTES:
        _pread(file.fd, view, offset, file.path)
        return
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = concurrent.futures.ThreadPoolExecutor(
                    _NUM_THREADS, thread_name_prefix="sglang-pread"
                )
    futures = [
        _pool.submit(_pread, file.fd, view[i : i + _CHUNK_BYTES], offset + i, file.path)
        for i in range(0, n, _CHUNK_BYTES)
    ]
    for future in futures:
        future.result()


def read_mapped_slice(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Return ``tensor``'s values read with ``pread`` into host memory, or
    ``None`` if it is not a slice of a registered mapping or stays on it. The
    result may share this thread's buffer: copy it before the thread's next read."""
    if (
        not _sources
        or tensor.device.type != "cpu"
        or tensor.numel() == 0
        or threading.current_thread() is not threading.main_thread()
    ):
        return None
    ptr = tensor.untyped_storage().data_ptr()
    source = _sources.get(ptr)
    if (
        source is None
        or source.ref() is None  # The mapping may be gone.
        or tensor._version != source.version  # Written in place since.
    ):
        return None
    itemsize = tensor.element_size()
    start = tensor.data_ptr() - ptr
    if tensor.is_contiguous():
        span = tensor.numel() * itemsize
    else:
        if source.file.prefetched or any(s < 0 for s in tensor.stride()):
            return None
        last = sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
        span = (last + 1) * itemsize
    if start < 0 or start + span > source.nbytes:
        return None
    if not _cached(tensor.data_ptr(), span):
        return None
    if span <= _CHUNK_BYTES:
        data = _staging(span)
    else:
        data = torch.empty(span, dtype=torch.uint8)
    _read(source.file, data, source.offset + start)
    data = data.view(tensor.dtype)
    if tensor.is_contiguous():
        return data.view(tensor.shape)
    return data.as_strided(tensor.shape, tensor.stride())


def host_source(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor``, or its values read with ``pread`` if ``read_mapped_slice``
    reads them; copy the result before this thread's next read."""
    read = read_mapped_slice(tensor)
    return tensor if read is None else read
