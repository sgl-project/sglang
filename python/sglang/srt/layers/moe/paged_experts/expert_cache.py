"""The expert cache (``--paged-experts-enable-disk``); see "The expert cache" in
docs/advanced_features/paged_experts.mdx.

A file is a 4 KiB header (magic, length, JSON), then one object per expert at a 4 KiB-aligned
offset, holding the expert's row of every paged tensor, each row 4 KiB-aligned, so that rows can
be read with direct I/O straight into a host store.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import ctypes
import errno
import functools
import glob
import hashlib
import importlib.util
import json
import logging
import math
import mmap
import os
import random
import re
import shutil
import struct
import time
import uuid
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.moe.paged_experts.formats import build_store
from sglang.srt.layers.moe.paged_experts.store import HostExpertStore, same_bytes

logger = logging.getLogger(__name__)

_VERSION = 2
_MAGIC = b"SGLPEXC\0"
_HEADER_BYTES = 4096
_OBJECT_ALIGN = 4096
_ROW_ALIGN = 4096
# Experts per read or write: bounded memory, few system calls.
_BATCH_BYTES = 64 << 20
# A staging file of a process this one cannot see (another container's), older than this, is
# left from a start that was killed.
_STALE_STAGING_S = 3600
# Free space a write leaves on the disk at least.
_FREE_MARGIN_BYTES = 8 << 30

# Files that load or repack the experts, relative to the sglang package; directories whose
# sources are hashed whole (the repack kernels, compiled at runtime).
_SOURCE_FILES = tuple(
    f"srt/layers/moe/paged_experts/{name}"
    for name in ("formats.py", "method.py", "runners.py", "store.py", "expert_cache.py")
)
_SOURCE_DIRS = ("kernels/jit/csrc/gemm/marlin",)
_SOURCE_SUFFIXES = (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp")
_PACKAGES = ("torch", "triton", "sglang-kernel", "flashinfer-python")
# What the Marlin runner's post-load steps run beyond the base method's and its scheme's
# modules: the repack helpers, the repack ops and their kernels.
_REPACK_SOURCES = (
    "sglang.srt.layers.quantization.utils",
    "sglang.srt.layers.quantization.marlin_utils",
    "sglang.srt.layers.quantization.marlin_utils_fp4",
    "sglang.srt.layers.quantization.marlin_utils_fp8",
    "sglang.srt.layers.quantization.mxfp4_marlin_moe",
    "sglang.kernels.ops.quantization.gptq_marlin_repack",
    "sglang.kernels.ops.quantization.awq_marlin_repack",
    "sglang.srt.hardware_backend.gpu.quantization.gptq_kernels",
    "sglang.srt.hardware_backend.gpu.quantization.awq_kernels",
)
# Reads of a file's batches in flight at once (they release the GIL).
_READERS = 4
# Single-expert reads of the disk tier in flight at once: a few keep an NVMe queue busy.
_READ_WORKERS = 6
# Segments one preadv takes at most (IOV_MAX).
_MAX_SEGMENTS = 1024
# Reads and writes bypass the page cache: an entry is read once per start, and caching it
# would crowd out the memory the host store pins.
_O_DIRECT = getattr(os, "O_DIRECT", 0)

_said = set()  # kinds of messages already logged during this model load


def _once(level: int, kind: str, message: str, *args) -> None:
    if kind not in _said:
        _said.add(kind)
        logger.log(level, message, *args)


_hint = {"bytes": 0, "repacks": False}  # what the cache would keep, without the flag


def hint_disk(layer, rows, num_experts: int, repacks: bool) -> None:
    """Without --paged-experts-enable-disk: count what the cache would keep of a layer it
    could keep; ``loading_done`` names the flag once, with the total."""
    if getattr(layer, "layer_name", None):
        _hint["bytes"] += ObjectLayout.of(rows).file_bytes(num_experts)
        _hint["repacks"] |= repacks


def cache_root() -> str:
    from sglang.srt.environ import envs

    return os.path.abspath(
        os.path.join(os.path.expanduser(envs.SGLANG_CACHE_DIR.get()), "paged_experts")
    )


# -- the file ---------------------------------------------------------------------------


class ObjectLayout(NamedTuple):
    """Where each paged tensor's row sits in an expert's object."""

    # name -> (row shape, dtype, offset in the object, bytes)
    tensors: Dict[str, Tuple[Tuple[int, ...], torch.dtype, int, int]]
    object_bytes: int

    @classmethod
    def of(cls, rows: Dict[str, Tuple[Tuple[int, ...], torch.dtype]]) -> ObjectLayout:
        tensors, offset = {}, 0
        for name, (shape, dtype) in rows.items():
            nbytes = math.prod(shape) * dtype.itemsize
            tensors[name] = (tuple(shape), dtype, offset, nbytes)
            offset += -(-nbytes // _ROW_ALIGN) * _ROW_ALIGN
        object_bytes = -(-max(offset, 1) // _OBJECT_ALIGN) * _OBJECT_ALIGN
        return cls(tensors, object_bytes)

    @classmethod
    def of_host(cls, host: Dict[str, torch.Tensor]) -> ObjectLayout:
        return cls.of({k: (tuple(v.shape[1:]), v.dtype) for k, v in host.items()})

    def describe(self):
        return [
            [name, list(shape), str(dtype).removeprefix("torch."), offset]
            for name, (shape, dtype, offset, _) in self.tensors.items()
        ]

    def file_bytes(self, num_experts: int) -> int:
        return _HEADER_BYTES + self.object_bytes * num_experts


def _encode_header(header: dict) -> bytes:
    body = json.dumps(header, sort_keys=True).encode()
    data = _MAGIC + struct.pack("<I", len(body)) + body
    if len(data) > _HEADER_BYTES:
        raise ValueError(f"expert cache header of {len(data)} bytes")
    return data.ljust(_HEADER_BYTES, b"\0")


def _decode_header(data: bytes) -> Optional[dict]:
    if len(data) < 12 or data[:8] != _MAGIC:
        return None
    (length,) = struct.unpack("<I", data[8:12])
    if 12 + length > len(data):
        return None
    return json.loads(data[12 : 12 + length])


def _pack(layout: ObjectLayout, rows: Dict[str, torch.Tensor], objects) -> None:
    """Rows ``[n, ...]`` of each tensor into ``n`` objects (uint8 ``[n, object_bytes]``)."""
    for name, (_, _, offset, nbytes) in layout.tensors.items():
        if nbytes:  # e.g. GPTQ without act-order keeps empty g_idx tensors
            src = rows[name].reshape(objects.shape[0], -1).view(torch.uint8)
            objects[:, offset : offset + nbytes].copy_(src)


def write_experts(fd: int, layout: ObjectLayout, host: Dict[str, torch.Tensor]) -> None:
    """Write every expert of ``host`` (``[E, ...]`` per tensor) as objects after the header."""
    num_experts = next(iter(host.values())).shape[0]
    per_batch = max(1, min(num_experts, _BATCH_BYTES // layout.object_bytes))
    buf = mmap.mmap(
        -1, per_batch * layout.object_bytes
    )  # page-aligned, as O_DIRECT needs
    objects = torch.frombuffer(buf, dtype=torch.uint8).view(-1, layout.object_bytes)
    for first in range(0, num_experts, per_batch):
        n = min(per_batch, num_experts - first)
        _pack(layout, {k: v[first : first + n] for k, v in host.items()}, objects[:n])
        offset = _HEADER_BYTES + first * layout.object_bytes
        written = os.pwrite(fd, memoryview(buf)[: n * layout.object_bytes], offset)
        if written != n * layout.object_bytes:
            raise OSError(f"short write at {offset}")


def read_experts(fd: int, layout: ObjectLayout, host: Dict[str, torch.Tensor]) -> None:
    """Read every expert into ``host`` (``[E, ...]`` per tensor): straight into rows that are
    whole, aligned blocks of direct I/O (a pinned store's large rows; the alignment is 4 KiB,
    not the page size, which is 64 KiB on some kernels), through a page-aligned bounce buffer
    otherwise; several batches at once."""
    num_experts = next(iter(host.values())).shape[0]
    width = len(layout.tensors) + 1
    per_batch = max(
        1,
        min(num_experts, _BATCH_BYTES // layout.object_bytes, _MAX_SEGMENTS // width),
    )

    def read(first: int, n: int) -> None:
        segments, bounced, copies = [], 0, []
        for e in range(first, first + n):
            used = 0
            for name, (_, _, offset, nbytes) in layout.tensors.items():
                if not nbytes:  # e.g. GPTQ without act-order keeps empty g_idx tensors
                    continue
                padded = -(-nbytes // _ROW_ALIGN) * _ROW_ALIGN
                row = host[name][e]
                if row.data_ptr() % _ROW_ALIGN == 0 and nbytes == padded:
                    segments.append((row.data_ptr(), nbytes))
                else:
                    segments.append((None, padded))
                    copies.append((row, bounced, nbytes))
                    bounced += padded
                used = offset + padded
            if layout.object_bytes > used:  # the object's padding
                segments.append((None, layout.object_bytes - used))
                bounced += layout.object_bytes - used
        bounce = mmap.mmap(-1, max(bounced, 1))
        base = ctypes.addressof(ctypes.c_char.from_buffer(bounce))
        buffers, cursor = [], 0
        for address, size in segments:
            if address is None:
                address, cursor = base + cursor, cursor + size
            buffers.append(
                memoryview((ctypes.c_char * size).from_address(address)).cast("B")
            )
        offset = _HEADER_BYTES + first * layout.object_bytes
        if os.preadv(fd, buffers, offset) != n * layout.object_bytes:
            raise OSError(f"short read at {offset}")
        staged = torch.frombuffer(bounce, dtype=torch.uint8)
        for row, at, nbytes in copies:
            row.reshape(-1).view(torch.uint8).copy_(staged[at : at + nbytes])
        del buffers, staged

    batches = [
        (s, min(per_batch, num_experts - s)) for s in range(0, num_experts, per_batch)
    ]
    with concurrent.futures.ThreadPoolExecutor(_READERS) as pool:
        for done in [pool.submit(read, *b) for b in batches]:
            done.result()


def _open_direct(path: str, flags: int, first_io):
    """``(fd, direct, result)``: the file opened with O_DIRECT when the filesystem takes it
    for ``first_io(fd)`` (an aligned read or write, whose result is returned), without it
    otherwise. A refused O_DIRECT open may have created the file already: the retry does not
    insist on creating it (``O_EXCL``)."""
    if _O_DIRECT:
        fd = None
        try:
            fd = os.open(path, flags | _O_DIRECT, 0o644)
            return fd, True, first_io(fd)
        except OSError as e:
            if fd is not None:
                os.close(fd)
            if e.errno != errno.EINVAL:
                raise
            _once(
                logging.INFO,
                "direct",
                "Paged experts: the expert cache under %s is read and written through the "
                "page cache (the filesystem refuses O_DIRECT)",
                os.path.dirname(path),
            )
            flags &= ~os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        return fd, False, first_io(fd)
    except BaseException:
        os.close(fd)
        raise


# -- the cache entry of one layer ---------------------------------------------------------


class ExpertCache:
    """One MoE layer's cache entry."""

    def __init__(self, path: str, header: dict):
        self.path = path
        self.header = header  # what the entry must declare, besides its layout
        self._building = None  # (staging path, fd, mapping) of an entry being built

    @classmethod
    def for_layer(
        cls, layer, expert_format, runner, base_method, rows, num_experts: int, repacks
    ) -> Optional[ExpertCache]:
        """The entry of ``layer``, whose loaded rows are ``rows`` (row shape, dtype per
        tensor); None when the checkpoint or the layer cannot be identified."""
        try:
            return cls._identify(
                layer, expert_format, runner, base_method, rows, num_experts, repacks
            )
        except Exception as e:  # noqa: BLE001 - the cache only saves time
            _once(logging.WARNING, "key", "Paged experts: no expert cache: %s", e)
            return None

    @classmethod
    def _identify(
        cls, layer, expert_format, runner, base_method, rows, num_experts, repacks
    ):
        layer_name = getattr(layer, "layer_name", None)
        if not layer_name:
            _once(
                logging.INFO,
                "name",
                "Paged experts: no expert cache for unnamed MoE layers",
            )
            return None
        model = _model()
        if model is None:
            _once(
                logging.INFO,
                "checkpoint",
                "Paged experts: no expert cache: the model loader did not report the "
                "checkpoint files it read (--load-format %s)",
                _load_format(),
            )
            return None
        # One directory per model (its layers may differ in format); what decides a layer's
        # bytes keys its file. Not the loader's code (the layer's, the model's): a change
        # there reaches every expert, and the checked groups catch it.
        key = _digest(f"v{_VERSION}", model.identity, _versions())
        scheme = getattr(layer, "scheme", None)
        backend = getattr(getattr(runner, "runner_backend", None), "value", "")
        layer_key = _digest(
            f"{type(expert_format).__name__}:{backend}",
            f"{type(base_method).__qualname__}:{type(scheme).__qualname__}",
            _sources(
                (type(base_method), type(scheme)), _REPACK_SOURCES if repacks else ()
            ),
            *(f"{n}:{shape}:{dtype}" for n, (shape, dtype) in rows.items()),
            str(num_experts),
        )
        directory = os.path.join(cache_root(), f"{model.label}-{key[:24]}")
        _sweep_staging(directory)
        filename = f"{_safe(layer_name)}-{layer_key[:12]}.experts"
        header = {"version": _VERSION, "key": key, "layer": layer_name, "of": layer_key}
        return cls(os.path.join(directory, filename), header)

    def _expected(self, layout: ObjectLayout, num_experts: int) -> dict:
        return {
            **self.header,
            "num_experts": num_experts,
            "object_bytes": layout.object_bytes,
            "tensors": layout.describe(),
        }

    def exists(self, num_experts: int) -> bool:
        """Whether the entry exists for this layer (its tensors are checked when read)."""
        try:
            with open(self.path, "rb") as f:
                header = _decode_header(f.read(_HEADER_BYTES))
                size = os.fstat(f.fileno()).st_size
            return (
                header is not None
                and all(header.get(k) == v for k, v in self.header.items())
                and header.get("num_experts") == num_experts
                and size == _HEADER_BYTES + header["object_bytes"] * num_experts
            )
        except Exception:  # noqa: BLE001
            return False

    def map(self, rows, num_experts: int) -> Optional[Dict[str, torch.Tensor]]:
        """The entry as ``[E, ...]`` views of each tensor of ``rows`` (row shape, dtype), on a
        private mapping of the file, for the disk tier: pages are read when touched, writes
        stay in memory. None when there is no entry or it does not fit."""
        layout = ObjectLayout.of(rows)
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "rb") as f:
                header = _decode_header(f.read(_HEADER_BYTES))
                size = os.fstat(f.fileno()).st_size
                if header != self._expected(
                    layout, num_experts
                ) or size != layout.file_bytes(num_experts):
                    self._does_not_fit()
                    return None
                mapped = mmap.mmap(f.fileno(), size, flags=mmap.MAP_PRIVATE)
        except Exception as e:  # noqa: BLE001
            self._cannot("read", e)
            return None
        return _views(mapped, layout, num_experts)

    def create(self, rows, num_experts: int) -> Dict[str, torch.Tensor]:
        """A new entry, being built, for the disk tier: ``[E, ...]`` views of each tensor of
        ``rows`` on a shared mapping of a staging file, so that what is written into them is
        the only write; ``publish`` puts it in place. The tier cannot do without it: failures
        raise."""
        layout = ObjectLayout.of(rows)
        nbytes = layout.file_bytes(num_experts)
        directory = os.path.dirname(self.path)

        def cannot(e: BaseException) -> RuntimeError:
            return RuntimeError(
                f"Paged experts: cannot build the disk tier's file under {directory} "
                f"({nbytes / (1 << 30):.1f} GiB, keeping {_FREE_MARGIN_BYTES >> 30} GiB "
                f"free): {e}; free space, set SGLANG_CACHE_DIR to a larger disk or raise "
                "--paged-experts-host-memory-gb"
            )

        try:
            os.makedirs(directory, exist_ok=True)
            # What this process never reads goes first, so that it can make the space.
            self._prune_layer_other_keys()
            _prune_other_keys(directory)
            free = shutil.disk_usage(directory).free
            if free < nbytes + _FREE_MARGIN_BYTES:
                raise OSError(errno.ENOSPC, f"{free / (1 << 30):.1f} GiB free")
            name = (
                f"{_staging_prefix()}{uuid.uuid4().hex}-{os.path.basename(self.path)}"
            )
            staging = os.path.join(directory, name)
            fd = os.open(staging, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        except OSError as e:
            raise cannot(e) from e
        try:
            # Reserve the space first: a full disk fails here, not as SIGBUS on a write.
            os.posix_fallocate(fd, 0, nbytes)
            header = _encode_header(self._expected(layout, num_experts))
            if os.pwrite(fd, header, 0) != _HEADER_BYTES:
                raise OSError("short write of the header")
            mapped = mmap.mmap(fd, nbytes)
        except OSError as e:
            os.close(fd)
            os.unlink(staging)
            raise cannot(e) from e
        except BaseException:
            os.close(fd)
            os.unlink(staging)
            raise
        self._building = (staging, fd, mapped)
        return _views(mapped, layout, num_experts)

    def abandon(self) -> None:
        """Drop the entry being built (``create``) after a failed build: its staging file and
        descriptor go now, the mapping with the last view of it."""
        if self._building is None:
            return
        staging, fd, _ = self._building
        self._building = None
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(staging)

    def publish(self) -> None:
        """Put the entry being built (``create``) in place: flushed, then renamed over the
        old one."""
        staging, fd, mapped = self._building
        self._building = None
        mapped.flush()
        os.fsync(fd)
        os.close(fd)
        os.replace(staging, self.path)
        self._written(os.path.getsize(self.path))

    def read(self, host: Dict[str, torch.Tensor]) -> bool:
        """Fill ``host`` (``[E, ...]`` per tensor) from the entry; False when there is none or
        it does not fit (a later ``write`` then replaces it)."""
        num_experts = next(iter(host.values())).shape[0]
        layout = ObjectLayout.of_host(host)
        if not os.path.exists(self.path):
            return False
        fd = None
        try:
            fd, direct, header = _open_direct(self.path, os.O_RDONLY, _read_header)
            if _decode_header(header) != self._expected(
                layout, num_experts
            ) or os.fstat(fd).st_size != layout.file_bytes(num_experts):
                self._does_not_fit()
                return False
            read_experts(fd, layout, host)
            if not direct:
                _drop_page_cache(fd)
            return True
        except Exception as e:  # noqa: BLE001
            self._cannot("read", e)
            return False
        finally:
            if fd is not None:
                os.close(fd)

    def write(self, host: Dict[str, torch.Tensor]) -> None:
        """Publish ``host`` as the entry, replacing any there and the layer's entries under
        other keys."""
        global _write_failed
        if _write_failed:  # an earlier layer could not write: this one would not
            _outcomes["failed"] += 1
            return
        num_experts = next(iter(host.values())).shape[0]
        layout = ObjectLayout.of_host(host)
        nbytes = layout.file_bytes(num_experts)
        directory = os.path.dirname(self.path)
        name = f"{_staging_prefix()}{uuid.uuid4().hex}-{os.path.basename(self.path)}"
        staging = os.path.join(directory, name)

        def begin(fd: int) -> None:
            # Reserve the space first: a full disk fails here, not halfway.
            os.posix_fallocate(fd, 0, nbytes)
            if os.pwrite(fd, header, 0) != _HEADER_BYTES:
                raise OSError("short write of the header")

        try:
            header = _aligned(_encode_header(self._expected(layout, num_experts)))
            os.makedirs(directory, exist_ok=True)
            # What this process never reads goes first, so that it can make the space.
            self._prune_layer_other_keys()
            _prune_other_keys(directory)
            free = shutil.disk_usage(directory).free
            if free < nbytes + _FREE_MARGIN_BYTES:
                raise OSError(
                    f"it needs {nbytes / (1 << 30):.1f} GiB + "
                    f"{_FREE_MARGIN_BYTES / (1 << 30):.0f} GiB margin, "
                    f"{free / (1 << 30):.1f} GiB are free; this and later layers are not "
                    "written"
                )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd, direct, _ = _open_direct(staging, flags, begin)
            try:
                write_experts(fd, layout, host)
                os.fsync(fd)
                if not direct:
                    _drop_page_cache(fd)
            finally:
                os.close(fd)
            # Replaces an entry that did not fit or reproduce; another start writing the same
            # key wrote the same bytes.
            os.replace(staging, self.path)
        except Exception as e:  # noqa: BLE001 - the cache only saves time
            _write_failed = True
            _outcomes["failed"] += 1
            self._cannot("write", e)
            with contextlib.suppress(OSError):
                os.unlink(staging)
            return
        self._written(nbytes)

    def _written(self, nbytes: int) -> None:
        directory = os.path.dirname(self.path)
        with contextlib.suppress(OSError):  # published; only its durability is unsure
            _fsync_dir(directory)
        _outcomes["written"] += 1
        _outcomes["bytes"] += nbytes

    def _prune_layer_other_keys(self) -> None:
        """Remove this layer's entries under other layer keys."""
        stem = self.path[: self.path.rindex("-")]
        for other in glob.glob(glob.escape(stem) + "-*.experts"):
            if other != self.path:
                with contextlib.suppress(OSError):
                    os.unlink(other)

    def save(self, host: Dict[str, torch.Tensor]) -> None:
        """Write ``host`` as the entry on the writer thread; ``host`` must not change any
        more. At most one write is pending."""
        global _writer, _pending
        if _writer is None:
            _writer = concurrent.futures.ThreadPoolExecutor(
                1, thread_name_prefix="expert-cache-writer"
            )
        wait_for_writes()
        _pending = _writer.submit(self.write, host)

    def _does_not_fit(self) -> None:
        _once(
            logging.WARNING,
            "invalid",
            "Paged experts: the expert cache %s does not fit its layer (another layout or a "
            "broken file); rebuilding it",
            self.path,
        )

    def warn_stale(self, experts: Sequence[int]) -> None:
        _once(
            logging.WARNING,
            "stale",
            "Paged experts: the expert cache %s differs from the checkpoint for experts %s; "
            "rebuilding it",
            self.path,
            list(experts),
        )

    def _cannot(self, what: str, error: Exception) -> None:
        _once(
            logging.WARNING,
            what,
            "Paged experts: cannot %s the expert cache %s: %s",
            what,
            self.path,
            error,
        )


_writer: Optional[concurrent.futures.ThreadPoolExecutor] = None
_pending: Optional[concurrent.futures.Future] = None
_write_failed = False  # a write failed: later layers would fail alike


def wait_for_writes() -> None:
    """Wait until the entry handed to the background writer is on disk (or failed)."""
    global _pending
    if _pending is not None:
        pending, _pending = _pending, None
        pending.result()


def _end_writes() -> None:
    """After the model's last layer: its entry written, the writer thread ends (a later
    model load starts another)."""
    global _writer
    wait_for_writes()
    if _writer is not None:
        writer, _writer = _writer, None
        writer.shutdown(wait=True)


def _views(mapped, layout: ObjectLayout, num_experts: int) -> Dict[str, torch.Tensor]:
    """``[E, ...]`` views of each tensor of ``layout`` on a mapping of an entry's file."""
    flat = torch.frombuffer(mapped, dtype=torch.uint8)
    views = {}
    for name, (shape, dtype, offset, nbytes) in layout.tensors.items():
        if not nbytes:
            views[name] = torch.empty((num_experts, *shape), dtype=dtype)
            continue
        strides = torch.empty(shape, device="meta").stride()
        views[name] = (
            flat[_HEADER_BYTES + offset :]
            .view(dtype)
            .as_strided(
                (num_experts, *shape),
                (layout.object_bytes // dtype.itemsize, *strides),
            )
        )
    return views


class ExpertReader:
    """Reads single experts of a layer's entry into host rows, for the disk tier. The file
    stays open while the layer is served: a start under another key may unlink it."""

    def __init__(self, cache: ExpertCache, host: Dict[str, torch.Tensor]):
        self.path = cache.path
        self.layout = ObjectLayout.of_host(host)
        self.fd, _, _ = _open_direct(self.path, os.O_RDONLY, _read_header)
        self._staging = None  # page-aligned, for rows a read cannot place directly
        self._staging_address = 0

    def __del__(self):
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)

    def drop_staging(self) -> None:
        """Give back the staging buffer, which is as large as the largest read so far; the
        next read that needs one maps one of its own size."""
        self._staging = None
        self._staging_address = 0

    def jobs(self, requests: List[Tuple[int, Dict[str, int]]]):
        """For ``(expert, {tensor: destination address})``: one read per expert, ``(reader,
        offset, [(address, bytes)])``, straight into page-aligned rows of whole pages, the
        others into staging; and the copies out of staging ``(dest, src, bytes)``."""
        need = len(requests) * self.layout.object_bytes
        if need and (self._staging is None or len(self._staging) < need):
            self._staging = mmap.mmap(-1, need)
            self._staging_address = ctypes.addressof(
                (ctypes.c_char * need).from_buffer(self._staging)
            )
        jobs, copies = [], []
        for k, (expert, dests) in enumerate(requests):
            stage = self._staging_address + k * self.layout.object_bytes
            segments = []  # contiguous in the file
            for name, (_, _, offset, nbytes) in self.layout.tensors.items():
                if not nbytes:
                    continue
                padded = -(-nbytes // _ROW_ALIGN) * _ROW_ALIGN
                dest = dests.get(name)
                if dest is not None and dest % _ROW_ALIGN == 0 and nbytes == padded:
                    segments.append((dest, nbytes))
                    continue
                segments.append((stage + offset, padded))
                if dest is not None:
                    copies.append((dest, stage + offset, nbytes))
            if segments:
                offset = _HEADER_BYTES + expert * self.layout.object_bytes
                jobs.append((self, offset, segments))
        return jobs, copies

    def preadv(self, offset: int, segments) -> None:
        buffers = [
            memoryview((ctypes.c_char * n).from_address(address)).cast("B")
            for address, n in segments
        ]
        expected = sum(n for _, n in segments)
        if os.preadv(self.fd, buffers, offset) != expected:
            raise OSError(f"Paged experts: short read at {offset} of {self.path}")


_reads: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _jobs(batches):
    jobs, copies = [], []
    for reader, requests in batches:
        j, c = reader.jobs(requests)
        jobs += j
        copies += c
    return jobs, copies


def read_many(batches) -> None:
    """The reads of several layers' ``(reader, requests)`` together, several in flight."""
    jobs, copies = _jobs(batches)
    if len(jobs) == 1:
        reader, offset, segments = jobs[0]
        reader.preadv(offset, segments)
        _copy(copies)
    elif jobs:
        wait_reads(_submit(jobs, copies))
    else:
        _copy(copies)


def wait_reads(started) -> None:
    """Wait for every read, then copy; a failed read raises once all of them are done (none
    may still write into a row after the error reaches the caller)."""
    futures, copies = started
    concurrent.futures.wait(futures)
    for future in futures:
        future.result()
    _copy(copies)


def _submit(jobs, copies):
    global _reads
    if _reads is None:
        _reads = concurrent.futures.ThreadPoolExecutor(
            _READ_WORKERS, thread_name_prefix="paged-experts-disk"
        )
    return [_reads.submit(r.preadv, o, g) for r, o, g in jobs], copies


def _copy(copies) -> None:
    for dest, src, n in copies:
        ctypes.memmove(dest, src, n)


# -- filling a layer's store ----------------------------------------------------------------


def fill_store(step, cache: Optional[ExpertCache], width: int) -> HostExpertStore:
    """The serving store of a layer whose loads ``step`` (a ``formats.PostLoad``) turns into
    paged rows: read from ``cache`` once the checked groups (``width`` wide: the router's
    top-k) give its bytes, else built from every expert, K at a time, and kept in ``cache``
    for later starts."""
    if cache is None:
        return build_store(step)
    if not _directory:
        _directory.append(os.path.dirname(cache.path))
    loads = step.loads
    groups = checked_groups(loads.num_experts, min(width, loads.num_slots))
    # The checked experts' checkpoint pages are read ahead at once and kept until the check
    # ends (a group shares experts with another). A cached start then gives them back; a
    # build leaves them to the chunk that replays them.
    checked = [e for e in sorted(set().union(*groups)) if e not in loads.keep]
    loads.read_ahead(checked)
    loads.keep.update(checked)
    restored = ran_check = False
    try:
        first = step.run(groups[0])  # also gives the paged layout
        store = HostExpertStore(step.rows_like, loads.num_experts)
        if cache.read(store.host):
            ran_check = True
            # Compared on the host: a repacking step's rows are views of the layer's whole
            # K-slot table on the GPU, which must not stay alive next to the later groups'.
            first = {name: rows.to("cpu", copy=True) for name, rows in first.items()}
            stale = check_groups(step, store, groups, first)
            if stale is None:
                restored = True
                _outcomes["read"] += 1
                return store
            cache.warn_stale(stale)
            _outcomes["rebuilt"] += 1
    finally:
        loads.keep.difference_update(checked)
        if restored:
            loads.page_out(checked)
    # The first group's rows are chunk 0's unless K is wider, or later groups reused them.
    reuse = not ran_check and groups[0] == list(
        range(min(loads.num_slots, loads.num_experts))
    )
    if reuse:  # the first group is chunk 0: copied, not run again
        for name, rows in first.items():
            store.host[name][: len(groups[0])].copy_(rows)
    first = rows = None  # its K-slot table must not stay alive next to the build's
    store = build_store(step, store, first_done=reuse)
    cache.save(store.host)
    return store


def check_groups(step, store: HostExpertStore, groups, first) -> Optional[List[int]]:
    """The first checked group the step does not give the restored bytes for (``first`` is
    the first group's rows, run already), or None."""
    for i, group in enumerate(groups):
        rows = first if i == 0 else step.run(group)
        stale = any(
            not same_bytes(r.cpu(), store.host[n][group]) for n, r in rows.items()
        )
        rows = None  # the next group's run builds another K-slot table
        if stale:
            return group
    return None


def checked_groups(num_experts: int, width: int) -> List[List[int]]:
    """The experts a start that reads the cache loads and compares, in groups of at most
    ``width``: 0..t-1, the last t, t sampled anew per start, and 1..t (the mixing check)."""
    t = min(width, num_experts)
    if num_experts <= t:
        return [list(range(num_experts))]
    groups = [list(range(t)), list(range(num_experts - t, num_experts))]
    seen = set(range(t + 1)) | set(groups[1])
    others = [e for e in range(num_experts) if e not in seen]
    if others:
        # Not the global generator: --random-seed would fix it across starts.
        groups.append(sorted(_sampler.sample(others, min(t, len(others)))))
    groups.append(list(range(1, t + 1)))
    return groups


_sampler = random.SystemRandom()

# What the cache did during this model load, for the line ``loading_done`` logs.
_outcomes = {"read": 0, "rebuilt": 0, "written": 0, "failed": 0, "bytes": 0}
_directory: List[str] = []  # the model's directory, once a layer has an entry


def loading_done() -> None:
    """After a model load (also one that failed; model_runner calls it): wait for the last
    entry to be written, say what the cache did (or, without the flag, what it would keep),
    and leave nothing of this load behind for the next one in the process."""
    global _write_failed
    try:
        _end_writes()
        if any(_outcomes.values()):
            _log_outcomes()
        elif _hint["bytes"]:
            _log_hint()
    finally:
        for k in _outcomes:
            _outcomes[k] = 0
        _hint.update(bytes=0, repacks=False)
        _directory.clear()
        _write_failed = False
        _said.clear()
        _models.clear()  # a later load may read other checkpoint files
        _prune_other_keys.cache_clear()
        _sweep_staging.cache_clear()


def _layers(n: int) -> str:
    return f"{n} layer" if n == 1 else f"{n} layers"


def _log_outcomes() -> None:
    done = []
    if _outcomes["read"]:
        done.append(f"read {_layers(_outcomes['read'])}")
    if _outcomes["rebuilt"]:
        done.append(f"rebuilt {_layers(_outcomes['rebuilt'])}")
    if _outcomes["written"]:
        done.append(
            f"wrote {_layers(_outcomes['written'])} "
            f"({_outcomes['bytes'] / (1 << 30):.1f} GiB)"
        )
    if _outcomes["failed"]:
        done.append(f"could not write {_layers(_outcomes['failed'])}")
    line = ", ".join(done)
    if _outcomes["written"]:
        line += "; later starts read them"
    directory = _directory[0] if _directory else cache_root()
    logger.info("Paged experts: expert cache %s: %s", directory, line)


def _log_hint() -> None:
    try:
        if _model() is None:
            return
    except Exception:  # noqa: BLE001 - only a hint
        return
    logger.info(
        "Paged experts: with --paged-experts-enable-disk, later starts would read these "
        "experts from %s (%.1f GiB) instead of %s them",
        cache_root(),
        _hint["bytes"] / (1 << 30),
        "loading and repacking" if _hint["repacks"] else "loading",
    )


# -- the key ----------------------------------------------------------------------------


class _Model(NamedTuple):
    label: str  # for humans: the model id or folder name
    identity: str  # digest of the checkpoint files' real paths, sizes and times


_models: Dict[Tuple[str, Optional[str]], _Model] = {}


def _model() -> Optional[_Model]:
    """The model, keyed by the checkpoint files the loader read; None if it did not report
    them (a loader other than the default one)."""
    from sglang.srt.model_loader.loader import loaded_checkpoint_files
    from sglang.srt.runtime_context import process_model_config

    config = process_model_config()
    key = (config.model_path, config.revision)
    if key not in _models:
        found = loaded_checkpoint_files(*key)
        if found is None:
            return None
        _models[key] = _Model(_model_label(config.model_path), _identity(*found))
    return _models[key]


def _identity(folder: str, files: Sequence[str]) -> str:
    """A digest of the checkpoint's weight files and its folder's ``*.json``: their real
    paths (an HF snapshot's links resolve to their blobs), sizes and modification times."""
    paths = sorted(files) + sorted(glob.glob(os.path.join(folder, "*.json")))
    parts = []
    for path in paths:
        st = os.stat(path)  # a broken link raises: no cache
        parts.append(f"{os.path.realpath(path)}:{st.st_size}:{st.st_mtime_ns}")
    return _digest(*parts)


def _load_format() -> str:
    from sglang.srt.runtime_context import get_model

    load_format = get_model().load_format
    return getattr(load_format, "value", load_format)


def _model_label(path: str) -> str:
    """``Qwen--Qwen3-30B`` for a hub id or a hub snapshot folder, the folder name and a hash
    of its real path otherwise."""
    if os.path.isdir(path):
        parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
        if (
            len(parts) >= 3
            and parts[-2] == "snapshots"
            and parts[-3].startswith("models--")
        ):
            label = parts[-3].removeprefix("models--")
        else:  # another folder of the same name is another model
            real = os.path.realpath(path)
            label = f"{parts[-1][:70]}-{hashlib.sha256(real.encode()).hexdigest()[:8]}"
    else:
        label = path.strip("/").replace("/", "--")
    label = _safe(label)
    if len(label) > 80:  # cut, with a hash: ids that share the start are other models
        label = f"{label[:71]}-{hashlib.sha256(path.encode()).hexdigest()[:8]}"
    return label


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


@functools.lru_cache(maxsize=None)
def _versions() -> str:
    # Not SGLang's own version: a git checkout's changes with every commit, and the sources
    # that decide the bytes are in the key by content.
    found = []
    for name in _PACKAGES:
        try:
            found.append(f"{name}={package_version(name)}")
        except PackageNotFoundError:
            pass
    return ";".join(found)


@functools.lru_cache(maxsize=None)
def _sources(owners, declared) -> str:
    """The content digest of the source files that load and repack the experts."""
    modules = set(declared)
    for owner in owners:
        for cls in getattr(owner, "__mro__", ()):
            if cls.__module__.startswith("sglang."):
                modules.add(cls.__module__)
    package = os.path.dirname(importlib.util.find_spec("sglang").origin)
    files = [_module_file(name, package) for name in sorted(modules)]
    files += [os.path.join(package, f) for f in _SOURCE_FILES]
    for directory in _SOURCE_DIRS:
        if not os.path.isdir(os.path.join(package, directory)):
            raise RuntimeError(f"its sources {directory} are not found")
        for root, dirs, names in os.walk(os.path.join(package, directory)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            files += [
                os.path.join(root, n) for n in names if n.endswith(_SOURCE_SUFFIXES)
            ]
    return _digest(
        *(
            f"{os.path.relpath(p, package)}:{_file_digest(p)}"
            for p in sorted(set(files))
        )
    )


def _module_file(name: str, package: str) -> str:
    """A module's source file; sglang's are found without importing them (or their parent
    packages, which may need a GPU)."""
    if name.startswith("sglang."):
        base = os.path.join(package, *name.split(".")[1:])
        for path in (base + ".py", os.path.join(base, "__init__.py")):
            if os.path.isfile(path):
                return path
    else:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            return spec.origin
    raise RuntimeError(f"its source {name} is not found")


@functools.lru_cache(maxsize=None)
def _file_digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


# -- helpers ----------------------------------------------------------------------------


def _read_header(fd: int) -> bytes:
    buf = mmap.mmap(-1, _HEADER_BYTES)  # page-aligned, as O_DIRECT needs
    n = os.preadv(fd, [buf], 0)
    return buf[:n]


def _aligned(data: bytes) -> mmap.mmap:
    """``data`` in a page-aligned buffer, as an O_DIRECT write needs."""
    buf = mmap.mmap(-1, len(data))
    buf.write(data)
    return buf


def _drop_page_cache(fd: int) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def _fsync_dir(directory: str) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


_KEYED = re.compile(r"(.+)-[0-9a-f]{24}")


@functools.lru_cache(maxsize=None)  # once per directory and process
def _prune_other_keys(directory: str) -> None:
    """Remove the model's entries under other keys, once one is written under this key: a
    model keeps one set. A start still reading one of them rebuilds what it has not read."""
    mine = _KEYED.fullmatch(os.path.basename(directory))
    if mine is None:
        return
    parent = os.path.dirname(directory)
    removed = 0
    try:
        for entry in os.scandir(parent):
            other = _KEYED.fullmatch(entry.name)
            if (
                other is None
                or other.group(1) != mine.group(1)
                or entry.path == directory
                or not entry.is_dir(follow_symlinks=False)
                or _building_there(entry.path)
            ):
                continue
            for file in os.scandir(entry.path):
                removed += file.stat(follow_symlinks=False).st_size
            shutil.rmtree(entry.path, ignore_errors=True)
    except OSError as e:
        _once(logging.WARNING, "prune", "Paged experts: cannot prune %s: %s", parent, e)
        return
    if removed:
        _once(
            logging.INFO,
            "pruned",
            "Paged experts: removed %.1f GiB of expert cache entries of %s under other keys",
            removed / (1 << 30),
            mine.group(1),
        )


def _staging_prefix(pid: Optional[int] = None) -> str:
    """Staging files are named for their writer: its PID namespace and PID, so that a later
    start sees whether the writer is gone."""
    return f".staging-{_pid_namespace()}-{os.getpid() if pid is None else pid}-"


def _pid_namespace() -> str:
    try:
        return str(os.stat("/proc/self/ns/pid").st_ino)
    except OSError:
        return "0"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # e.g. another user's process
        return True
    return True


_STAGING = re.compile(r"\.staging-(\d+)-(\d+)-")


@functools.lru_cache(maxsize=None)  # once per directory and process
def _sweep_staging(directory: str) -> None:
    """Remove staging files a killed start left behind: those of a process of this PID
    namespace that is gone, and, of another namespace (another container), those older than
    ``_STALE_STAGING_S``."""
    for path in glob.glob(os.path.join(directory, ".staging-*")):
        with contextlib.suppress(OSError):
            if _writer_gone(path):
                os.unlink(path)


def _writer_gone(path: str) -> bool:
    """Whether the start writing staging file ``path`` is gone: a process of this PID
    namespace that no longer runs, or, of another namespace (another container), a file older
    than ``_STALE_STAGING_S``."""
    found = _STAGING.match(os.path.basename(path))
    if found and found.group(1) == _pid_namespace():
        return not _alive(int(found.group(2)))
    return time.time() - os.stat(path).st_mtime > _STALE_STAGING_S


def _building_there(directory: str) -> bool:
    """Whether a running start is building a file in ``directory`` (another key's entries of
    the model): a disk tier's first build needs its directory until it publishes."""
    for path in glob.glob(os.path.join(directory, ".staging-*")):
        with contextlib.suppress(OSError):
            if not _writer_gone(path):
                return True
    return False
