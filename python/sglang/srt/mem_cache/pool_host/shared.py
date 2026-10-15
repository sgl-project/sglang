"""Explicit shared CPU storage for one local MLA/DSA replica group."""

from __future__ import annotations

import math
import mmap
import os
import uuid

import torch
import torch.distributed as dist

from sglang.srt.utils.stale_shm_cleanup import make_shm_name

# A CUDA unregister error must not turn into an unmapped registered pointer if
# a failed constructor unwinds. Entries are removed when release is retried
# successfully; irrecoverable CUDA failures keep only these mappings to exit.
_PINNED_FAILURES: dict[tuple[int, int], torch.Tensor] = {}


def synchronize_shared_error(process_group, error, *, operation: str) -> None:
    """Raise on every participant if any rank failed the same operation."""
    errors = [None] * dist.get_world_size(process_group)
    dist.all_gather_object(
        errors, None if error is None else str(error), group=process_group
    )
    failures = [
        f"rank {rank}: {value}"
        for rank, value in enumerate(errors)
        if value is not None
    ]
    if failures:
        raise RuntimeError(f"Shared HiCache {operation} failed: " + "; ".join(failures))


class SharedHostTensorAllocator:
    """Map one tmpfs allocation in every rank without a resource-tracker patch.

    The supplied Gloo group is borrowed. Every rank must call allocate in the
    same order. Names are removed once all ranks attach; the mapping remains
    alive through torch's buffer ownership until its final tensor/view dies.
    """

    is_shared = True

    def __init__(self, process_group):
        if str(dist.get_backend(process_group)) != "gloo":
            raise ValueError("Shared HiCache host allocation requires a Gloo group")
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.owner_rank = dist.get_global_rank(process_group, 0)
        self.is_owner = self.rank == 0
        self._mappings: dict[int, mmap.mmap] = {}
        self._failed_registrations: dict[int, torch.Tensor] = {}
        self._closed = False

    def synchronize_error(self, error, *, operation: str) -> None:
        synchronize_shared_error(self.process_group, error, operation=operation)

    def _create_mapping(self, path: str, size: int, token: bytes):
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            # ftruncate alone leaves sparse tmpfs pages. Reserve backing now
            # so insufficient /dev/shm space raises instead of a later SIGBUS.
            os.posix_fallocate(fd, 0, size)
            mapping = mmap.mmap(fd, size)
            mapping[: len(token)] = token
            return mapping
        finally:
            os.close(fd)

    def _open_mapping(self, path: str, size: int, token: bytes):
        fd = os.open(path, os.O_RDWR)
        try:
            if os.fstat(fd).st_size != size:
                raise ValueError("shared mapping size differs from the owner")
            mapping = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        if mapping[: len(token)] != token:
            mapping.close()
            raise ValueError("ranks do not share the same /dev/shm namespace")
        return mapping

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        error = None
        spec = None
        try:
            if self._closed:
                raise RuntimeError("allocator has already been closed")
            if device != "cpu":
                raise ValueError("shared host tensors must be allocated on CPU")
            dims = tuple(dims)
            if not dims or any(not isinstance(dim, int) or dim <= 0 for dim in dims):
                raise ValueError(f"shared host dimensions must be positive: {dims}")
            spec = (dims, str(dtype), device, dtype.itemsize)
        except Exception as exc:
            error = exc
        self.synchronize_error(error, operation="allocation validation")
        specs = [None] * self.world_size
        dist.all_gather_object(specs, spec, group=self.process_group)
        if any(value != specs[0] for value in specs):
            raise ValueError(
                "Shared HiCache allocation shape/dtype differs across ranks"
            )

        data_bytes = math.prod(dims) * dtype.itemsize
        size = mmap.PAGESIZE + data_bytes
        mapping = None
        path = None
        payload = [None]
        error = None
        if self.is_owner:
            try:
                free = os.statvfs("/dev/shm")
                if size > free.f_bavail * free.f_frsize:
                    raise ValueError(
                        f"/dev/shm has insufficient space for {size} shared HiCache bytes"
                    )
                path = os.path.join("/dev/shm", make_shm_name("hicache"))
                token = uuid.uuid4().bytes
                mapping = self._create_mapping(path, size, token)
                payload[0] = (path, token)
            except Exception as exc:
                error = exc
        try:
            self.synchronize_error(error, operation="owner allocation")
            dist.broadcast_object_list(
                payload, src=self.owner_rank, group=self.process_group
            )
            path, token = payload[0]
            error = None
            if not self.is_owner:
                try:
                    mapping = self._open_mapping(path, size, token)
                except Exception as exc:
                    error = exc
            self.synchronize_error(error, operation="mapping attachment")

            error = None
            buffer = None
            try:
                buffer = torch.frombuffer(
                    mapping, dtype=dtype, count=math.prod(dims), offset=mmap.PAGESIZE
                ).reshape(dims)
            except Exception as exc:
                error = exc
            self.synchronize_error(error, operation="tensor construction")
            error = None
            if self.is_owner:
                try:
                    os.unlink(path)
                    path = None
                except Exception as exc:
                    error = exc
            self.synchronize_error(error, operation="mapping unlink")
            self._mappings[buffer.data_ptr()] = mapping
            return buffer
        finally:
            # All successful attachments keep the inode alive after unlink.
            # On failure each rank drops only its own mapping.
            if self.is_owner and path is not None:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            if mapping is not None and not any(
                value is mapping for value in self._mappings.values()
            ):
                # A successfully constructed tensor may still retain its
                # buffer while this exception unwinds. Let its storage close
                # the mmap when the final reference disappears.
                mapping = None

    def release(self, buffer: torch.Tensor) -> None:
        """Drop our reference after CUDA unregister; tensor storage keeps mmap safe."""
        if getattr(buffer, "_sglang_cuda_host_registered_ranges", None):
            self._failed_registrations[buffer.data_ptr()] = buffer
            _PINNED_FAILURES[(id(self), buffer.data_ptr())] = buffer
            raise RuntimeError(
                "Shared HiCache CUDA unregister failed; retaining the pinned mapping"
            )
        self._failed_registrations.pop(buffer.data_ptr(), None)
        _PINNED_FAILURES.pop((id(self), buffer.data_ptr()), None)
        self._mappings.pop(buffer.data_ptr(), None)

    def close(self) -> None:
        """Release local ownership after the caller drains copies and drops views.

        Do not call mmap.close explicitly: an external tensor view can still
        own its storage. torch.frombuffer retains the mmap until that storage
        dies, then normal mmap destruction closes it without dangling tensors.
        """
        if self._failed_registrations:
            raise RuntimeError(
                "Shared HiCache still has pinned mappings after rollback"
            )
        self._mappings.clear()
        self._closed = True
