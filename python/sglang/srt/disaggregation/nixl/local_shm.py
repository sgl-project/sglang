"""Opt-in same-node HiSparse host KV writes. NIXL still carries notifications.

An exported pool has a fresh endpoint and capability for its entire lifetime.
Imported mappings retain the backing file even if the exporting process exits.
"""

from __future__ import annotations

import logging
import mmap
import operator
import os
import secrets
import socket
import struct
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import msgspec

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

# A failed unregister must not turn a still-pinned mapping into freed memory.
_PINNED_MAPPING_QUARANTINE = []
logger = logging.getLogger(__name__)


# Internal state, never serialized; one lock protects one peer's lifecycle.
class LocalShmPeer(msgspec.Struct):
    lock: threading.RLock = msgspec.field(default_factory=threading.RLock)
    mapping: Optional[LocalShmMapping] = None
    error: Optional[Exception] = None
    retry_at: Optional[float] = None


def host_identity() -> list[str]:
    # Abstract AF_UNIX sockets are scoped to the network namespace.
    return [
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        os.readlink("/proc/self/ns/net"),
    ]


def same_host(descriptor: dict) -> bool:
    return descriptor.get("host") == host_identity()


class LocalShmExport:
    def __init__(self, pool: MLATokenToKVPoolHost) -> None:
        buffer = pool.kv_buffer
        if pool.fd is None or not buffer.is_contiguous():
            raise ValueError("Local SHM requires a contiguous shared host KV pool")
        ptrs, lengths, items = pool.get_contiguous_buf_infos()
        self.token = secrets.token_bytes(32)
        self.stopped = threading.Event()
        name = "sglang-pd-" + secrets.token_hex(16)
        with ExitStack() as rollback:
            self.fd = os.dup(pool.fd)
            rollback.callback(os.close, self.fd)
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            rollback.callback(self.socket.close)
            self.socket.bind("\0" + name)
            self.socket.listen(16)
            self.socket.settimeout(0.2)
            self.descriptor = dict(
                version=1,
                host=host_identity(),
                endpoint=name,
                token=self.token.hex(),
                size=buffer.numel() * buffer.element_size(),
                offsets=[ptr - buffer.data_ptr() for ptr in ptrs],
                lengths=lengths,
                items=items,
            )
            self.thread = threading.Thread(target=self._serve, daemon=True)
            self.thread.start()
            rollback.pop_all()

    def _serve(self) -> None:
        while not self.stopped.is_set():
            try:
                conn, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(1)
                    _, uid, _ = struct.unpack(
                        "3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                    )
                    if uid != os.getuid():
                        continue
                    token = bytearray()
                    while len(token) < len(self.token):
                        part = conn.recv(len(self.token) - len(token))
                        if not part:
                            break
                        token.extend(part)
                    if secrets.compare_digest(token, self.token):
                        socket.send_fds(conn, [b"F"], [self.fd])
                except (OSError, ValueError):
                    continue

    def close(self) -> None:
        self.stopped.set()
        self.socket.close()
        self.thread.join()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def build_copy_plan(
    src_lengths_bytes: Sequence[int],
    dst_lengths_bytes: Sequence[int],
    page_sizes_bytes: Sequence[int],
    src_indices: Sequence[int],
    dst_indices: Sequence[int],
) -> list[tuple[int, int, int, int]]:
    """Validate a chunk and return (region, src offset, dst offset, byte count).

    Offsets and sizes are bytes; source/destination indices select KV pages.
    No writes may be submitted until the entire plan has been validated.
    """
    if len(src_lengths_bytes) != len(dst_lengths_bytes) or len(page_sizes_bytes) != len(
        src_lengths_bytes
    ):
        raise ValueError("Local SHM KV region count mismatch")
    if len(src_indices) != len(dst_indices):
        raise ValueError("Local SHM page count mismatch")
    pairs = [
        (operator.index(s), operator.index(d)) for s, d in zip(src_indices, dst_indices)
    ]
    plan = []
    for layer, (src_size, dst_size, item) in enumerate(
        zip(src_lengths_bytes, dst_lengths_bytes, page_sizes_bytes)
    ):
        if item <= 0:
            raise ValueError("Invalid local SHM page size")
        for src, dst in pairs:
            if (
                src < 0
                or dst < 0
                or (src + 1) * item > src_size
                or (dst + 1) * item > dst_size
            ):
                raise ValueError("Local SHM page index outside KV region")
            plan.append((layer, src * item, dst * item, item))
    return plan


class LocalShmTransfer:
    def __init__(
        self,
        owner: LocalShmMapping,
        event: Optional[torch.cuda.Event],
        stream: torch.cuda.Stream,
        notify: Callable[[], None],
        error: Optional[Exception] = None,
    ) -> None:
        self.owner = owner  # Keep the mapping and stream alive until DMA ends.
        self.event = event
        self.stream = stream
        self.notify = notify
        self.error = error
        self.state = "PROC"

    def poll(self) -> str:
        if self.state != "PROC":
            return self.state
        if self.event is None:
            raise RuntimeError("Local SHM completion event could not be recorded")
        if not self.event.query():
            return "PROC"
        if self.error is None:
            try:
                self.notify()
            except Exception as exc:
                self.error = exc
        self.state = "ERR" if self.error is not None else "DONE"
        self.owner.pending.remove(self)
        return self.state


class LocalShmMapping:
    def __init__(self, descriptor: dict, gpu_id: int) -> None:
        import torch

        from sglang.srt.mem_cache.pool_host.common import _cuda_host_register

        if descriptor.get("version") != 1 or not same_host(descriptor):
            raise ValueError("Incompatible local SHM descriptor")
        self.gpu_id = gpu_id
        self.endpoint = descriptor["endpoint"]
        self.pending = set()
        self.buffer = None
        self.mapping = None
        self.fd = None
        self.lengths = descriptor["lengths"]
        self.items = descriptor["items"]
        offsets = descriptor["offsets"]
        size = descriptor["size"]
        if (
            size <= 0
            or len(offsets) != len(self.lengths)
            or len(offsets) != len(self.items)
        ):
            raise ValueError("Invalid local SHM regions")
        for offset, length, item in zip(offsets, self.lengths, self.items):
            if offset < 0 or length <= 0 or offset + length > size or item <= 0:
                raise ValueError("Invalid local SHM region bounds")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(5)
            conn.connect("\0" + descriptor["endpoint"])
            conn.sendall(bytes.fromhex(descriptor["token"]))
            message, fds, _, _ = socket.recv_fds(conn, 1, 1, socket.MSG_CMSG_CLOEXEC)
        if message != b"F" or len(fds) != 1:
            for fd in fds:
                os.close(fd)
            raise RuntimeError("Local SHM FD exchange failed")
        self.fd = fds[0]
        try:
            if os.fstat(self.fd).st_size < size:
                raise ValueError("Local SHM backing size mismatch")
            self.mapping = mmap.mmap(self.fd, size, flags=mmap.MAP_SHARED)
            self.buffer = torch.frombuffer(self.mapping, dtype=torch.uint8)
            with torch.cuda.device(gpu_id):
                _cuda_host_register(
                    self.buffer, registration_granularity_bytes=max(self.items)
                )
            self.ptrs = [self.buffer.data_ptr() + offset for offset in offsets]
        except Exception:
            if getattr(self.buffer, "_sglang_cuda_host_registered_ranges", []):
                _PINNED_MAPPING_QUARANTINE.append(self)
                raise
            self.buffer = None
            if self.mapping is not None:
                self.mapping.close()
            os.close(self.fd)
            raise

    def submit(
        self,
        src_ptrs: Sequence[int],
        src_lengths: Sequence[int],
        src_indices: Sequence[int],
        dst_indices: Sequence[int],
        notify: Callable[[], None],
    ) -> LocalShmTransfer:
        import torch
        from cuda.bindings import runtime

        if len(src_ptrs) != len(self.ptrs):
            raise ValueError("Local SHM source region count mismatch")
        plan = build_copy_plan(
            src_lengths, self.lengths, self.items, src_indices, dst_indices
        )
        with torch.cuda.device(self.gpu_id):
            stream = torch.cuda.Stream()
            event = torch.cuda.Event()
            handle = LocalShmTransfer(self, event, stream, notify)
            self.pending.add(handle)
            try:
                for layer, src, dst, size in plan:
                    (rc,) = runtime.cudaMemcpyAsync(
                        self.ptrs[layer] + dst,
                        src_ptrs[layer] + src,
                        size,
                        runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream.cuda_stream,
                    )
                    if int(rc):
                        raise RuntimeError(f"Local SHM cudaMemcpyAsync failed: {rc}")
            except Exception as exc:
                logger.exception(
                    "Local SHM submission failed; waiting for queued writes"
                )
                handle.error = exc
            finally:
                try:
                    event.record(stream)
                except Exception as exc:
                    # Unknown completion: retain mapping and never acknowledge drain.
                    logger.exception(
                        "Local SHM completion is unknown; retaining mapping"
                    )
                    handle.error = exc
                    handle.event = None
            return handle

    def exporter_closed(self) -> bool:
        """Only connection refusal proves this abstract endpoint is gone."""
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(1)
                conn.connect("\0" + self.endpoint)
        except ConnectionRefusedError:
            return True
        except OSError:
            # Timeout/resource pressure does not establish peer death.
            return False
        return False

    def close(self) -> None:
        from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

        if self.pending:
            raise RuntimeError("Cannot close local SHM while writes may be in flight")
        if self.buffer is not None:
            import torch

            with torch.cuda.device(self.gpu_id):
                _cuda_host_unregister(self.buffer)
            if getattr(self.buffer, "_sglang_cuda_host_registered_ranges", []):
                raise RuntimeError(
                    "Local SHM host unregister failed; retaining mapping"
                )
            self.buffer = None
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
