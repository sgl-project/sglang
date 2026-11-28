"""Read safetensors checkpoint tensors to the GPU through pinned staging buffers.

The default loader hands each weight loader a view of the file mapping, so with
tensor parallelism every narrow, ``.contiguous()`` and pad runs on the CPU, one
tensor at a time, page-faulting the mapping, before a pageable copy to the GPU.
``GpuStagingReader`` instead reads the bytes of each tensor with ``os.preadv``
(several threads, GIL released), in merged file ranges of up to
``_CHUNK_BYTES``, into a ring of ``_NUM_BUFFERS`` pinned host buffers; copies
each range to a GPU buffer of the ring on a side stream; and yields each tensor
as a new GPU tensor with the dtype and shape ``safe_open`` would give. The
weight loaders then slice, pad and copy on the GPU.

Tensors larger than ``_MAX_TENSOR_BYTES``, empty tensors and tensors of a dtype
this module does not know are yielded as ``safe_open`` views, as before.
Within a file, tensors are yielded in file order.
"""

import concurrent.futures
import json
import logging
import math
import os
import queue
import struct
import threading
import time
from typing import Callable, Generator, List, Optional, Tuple

import safetensors
import safetensors.torch
import torch

logger = logging.getLogger(__name__)

# Adjacent tensors are read and copied as one range of up to this many bytes.
_CHUNK_BYTES = 64 << 20
# Larger tensors (e.g. some embeddings) stay mapped views.
_MAX_TENSOR_BYTES = 512 << 20
# Bytes per pread call; the threads split a range into such calls.
_READ_BYTES = 16 << 20
_NUM_READ_THREADS = 8
# Pinned host and GPU buffers each: ranges in flight between reads and loaders.
_NUM_BUFFERS = 3

_DTYPES = {
    **safetensors.torch._TYPES,
    "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
}

# (name, nbytes) -> whether the caller does not need the tensor.
SkipFn = Callable[[str, int], bool]


def _pread_into(fd: int, buf: memoryview, pos: int) -> None:
    while buf:
        n = os.preadv(fd, [buf], pos)
        if n == 0:
            raise EOFError(f"short read at file offset {pos}")
        buf, pos = buf[n:], pos + n


class _Range:
    """Consecutive tensors of a file, read and copied to the GPU together."""

    def __init__(self, start: int):
        self.start = self.end = start
        # (name, dtype, shape, offset in the range, nbytes)
        self.tensors: List[Tuple[str, torch.dtype, List[int], int, int]] = []


def _plan(path: str, skip: Optional[SkipFn]) -> List[Tuple[str, object]]:
    """The tensors of a file to yield, in file order, as ("map", name) or
    ("range", _Range) items."""
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    items: List[Tuple[str, object]] = []
    cur: Optional[_Range] = None
    for name, info in sorted(header.items(), key=lambda e: e[1]["data_offsets"]):
        start, end = (8 + size + o for o in info["data_offsets"])
        nbytes = end - start
        if skip is not None and skip(name, nbytes):
            continue
        dtype = _DTYPES.get(info["dtype"])
        if (
            dtype is None
            or not 0 < nbytes <= _MAX_TENSOR_BYTES
            or math.prod(info["shape"]) * dtype.itemsize != nbytes
        ):
            items.append(("map", name))
            cur = None
            continue
        if cur is None or start != cur.end or end - cur.start > _CHUNK_BYTES:
            cur = _Range(start)
            items.append(("range", cur))
        cur.tensors.append((name, dtype, info["shape"], start - cur.start, nbytes))
        cur.end = end
    return items


class GpuStagingReader:
    """Reads safetensors files to the current GPU for one model load.

    Allocates ``_NUM_BUFFERS`` pinned host and GPU buffers of
    max(``_CHUNK_BYTES``, largest staged tensor) bytes on first use, and keeps
    them until ``close``, which frees them and logs the load's statistics."""

    def __init__(self) -> None:
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.copy_stream = torch.cuda.Stream(self.device)
        self.pool = concurrent.futures.ThreadPoolExecutor(
            _NUM_READ_THREADS, thread_name_prefix="gpu-staging-read"
        )
        self.capacity = 0
        self.host: List[torch.Tensor] = []
        self.dev: List[torch.Tensor] = []
        self.staged_bytes = self.staged_tensors = self.mapped_tensors = 0
        self.read_s = self.wait_s = self.loader_s = 0.0

    def _reserve(self, nbytes: int) -> None:
        if nbytes <= self.capacity:
            return
        nbytes = -(-nbytes // _CHUNK_BYTES) * _CHUNK_BYTES
        torch.cuda.synchronize(self.device)  # Nothing uses the old buffers.
        self.host = self.dev = []
        self.host = [
            torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(_NUM_BUFFERS)
        ]
        self.dev = [
            torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            for _ in range(_NUM_BUFFERS)
        ]
        self.capacity = nbytes

    def _read(self, fd: int, out: torch.Tensor, pos: int) -> None:
        """Read ``out.numel()`` bytes at file offset ``pos`` into ``out``."""
        buf = memoryview(out.numpy())
        futures = [
            self.pool.submit(_pread_into, fd, buf[i : i + _READ_BYTES], pos + i)
            for i in range(0, len(buf), _READ_BYTES)
        ]
        concurrent.futures.wait(futures)  # All done with ``out`` before raising.
        for future in futures:
            future.result()

    def read_file(
        self, path: str, skip: Optional[SkipFn] = None
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Yield the tensors of ``path`` not skipped by ``skip(name, nbytes)``,
        which is called before any tensor is read."""
        items = _plan(path, skip)
        ranges = [r for kind, r in items if kind == "range"]
        if not ranges:
            with safetensors.safe_open(path, framework="pt", device="cpu") as mapped:
                for _, name in items:
                    self.mapped_tensors += 1
                    yield name, mapped.get_tensor(name)
            return

        self._reserve(max(r.end - r.start for r in ranges))
        device, copy_stream = self.device, self.copy_stream
        stream = torch.cuda.current_stream(device)
        # Buffers the reader may fill, each with the event after which the
        # loader no longer reads it (None: unused so far in this file).
        free: queue.Queue = queue.Queue()
        for k in range(_NUM_BUFFERS):
            free.put((k, None))
        ready: queue.Queue = queue.Queue()
        stop = threading.Event()

        def produce(fd: int) -> None:
            try:
                torch.cuda.set_device(device)
                for r in ranges:
                    while True:
                        if stop.is_set():
                            return
                        try:
                            k, consumed = free.get(timeout=0.1)
                            break
                        except queue.Empty:
                            pass
                    if consumed is not None:
                        # The loader's copies out of dev[k] are done, and so is
                        # the copy into dev[k] that last read host[k].
                        consumed.synchronize()
                    host = self.host[k][: r.end - r.start]
                    t0 = time.perf_counter()
                    self._read(fd, host, r.start)
                    self.read_s += time.perf_counter() - t0
                    with torch.cuda.stream(copy_stream):
                        self.dev[k][: r.end - r.start].copy_(host, non_blocking=True)
                        copied = torch.cuda.Event()
                        copied.record(copy_stream)
                    ready.put((k, copied))
            except BaseException as e:  # Raised in the loader's thread.
                ready.put(e)

        fd = os.open(path, os.O_RDONLY)
        producer = threading.Thread(
            target=produce, args=(fd,), name="gpu-staging", daemon=True
        )
        producer.start()
        try:
            with safetensors.safe_open(path, framework="pt", device="cpu") as mapped:
                for kind, item in items:
                    if kind == "map":
                        self.mapped_tensors += 1
                        yield item, mapped.get_tensor(item)
                        continue
                    t0 = time.perf_counter()
                    got = ready.get()
                    self.wait_s += time.perf_counter() - t0
                    if isinstance(got, BaseException):
                        raise got
                    k, copied = got
                    stream.wait_event(copied)
                    for name, dtype, shape, offset, nbytes in item.tensors:
                        # A new tensor per weight: the buffer is reused, and a
                        # loader may keep what it is given.
                        out = torch.empty(shape, dtype=dtype, device=device)
                        out.view(-1).view(torch.uint8).copy_(
                            self.dev[k][offset : offset + nbytes]
                        )
                        self.staged_tensors += 1
                        self.staged_bytes += nbytes
                        t1 = time.perf_counter()
                        yield name, out
                        self.loader_s += time.perf_counter() - t1
                    consumed = torch.cuda.Event()
                    consumed.record(stream)
                    free.put((k, consumed))
        finally:
            stop.set()
            producer.join()
            os.close(fd)
            # The next file's reads must not overwrite buffers still in use.
            copy_stream.synchronize()
            stream.synchronize()

    def close(self) -> None:
        """Free the buffers and log the statistics of the load."""
        torch.cuda.synchronize(self.device)
        self.host = self.dev = []
        self.capacity = 0
        self.pool.shutdown()
        # The KV cache is sized from the free memory after loading.
        torch.cuda.empty_cache()
        logger.info(
            "GPU staging: read %.2f GiB in %d tensors to the GPU, %d tensors as "
            "mapped views; reads took %.2f s, the loader waited %.2f s for them "
            "and spent %.2f s in weight loaders on staged tensors.",
            self.staged_bytes / (1 << 30),
            self.staged_tensors,
            self.mapped_tensors,
            self.read_s,
            self.wait_s,
            self.loader_s,
        )
