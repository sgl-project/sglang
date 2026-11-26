import logging
import multiprocessing.shared_memory
import os
import threading
import time
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List

import torch

from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsClient
from sglang.srt.utils.cpp_extension_loader import load_extension_with_recovery

root = Path(__file__).parent.resolve()
hf3fs_utils = load_extension_with_recovery(
    name="hf3fs_utils", sources=[f"{root}/hf3fs_utils.cpp"]
)

logger = logging.getLogger(__name__)

HF3FS_AVAILABLE = True
try:
    from hf3fs_fuse.io import (
        deregister_fd,
        extract_mount_point,
        make_ioring,
        make_iovec,
        register_fd,
    )
except ImportError:
    HF3FS_AVAILABLE = False


@dataclass
class _RingState:
    generation: int = 0
    pending: set[tuple[int, int]] = field(default_factory=set)
    needs_submit: bool = False


class Hf3fsUsrBioClient(Hf3fsClient):
    """HF3FS client implementation using usrbio."""

    def __init__(
        self,
        path: str,
        size: int,
        bytes_per_page: int,
        entries: int,
        client_timeout: int,
    ):
        if not HF3FS_AVAILABLE:
            raise ImportError(
                "hf3fs_fuse.io is not available. Please install the hf3fs_fuse package."
            )

        self.path = path
        self.size = size
        self.bytes_per_page = bytes_per_page
        self.entries = entries
        self.client_timeout = client_timeout
        self.rlock = threading.RLock()
        self.wlock = threading.RLock()
        self._read_state = _RingState()
        self._write_state = _RingState()
        self.file = None
        self._fd_registered = False
        self.bs = self.bytes_per_page
        self.numa = -1
        try:
            self._open()
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception("Error cleaning up HF3FS client initialization")
            raise

    def _open(self) -> None:
        self.file = os.open(self.path, os.O_RDWR | os.O_CREAT)
        os.ftruncate(self.file, self.size)
        register_fd(self.file)
        self._fd_registered = True

        self.hf3fs_mount_point = extract_mount_point(self.path)
        self.shm_r = multiprocessing.shared_memory.SharedMemory(
            size=self.bs * self.entries, create=True
        )
        self.shm_w = multiprocessing.shared_memory.SharedMemory(
            size=self.bs * self.entries, create=True
        )
        self.shm_r_tensor = torch.frombuffer(self.shm_r.buf, dtype=torch.uint8)
        self.shm_w_tensor = torch.frombuffer(self.shm_w.buf, dtype=torch.uint8)
        self.ior_r = make_ioring(
            self.hf3fs_mount_point,
            self.entries,
            for_read=True,
            timeout=1,
            numa=self.numa,
        )
        self.ior_w = make_ioring(
            self.hf3fs_mount_point,
            self.entries,
            for_read=False,
            timeout=1,
            numa=self.numa,
        )
        self.iov_r = make_iovec(self.shm_r, self.hf3fs_mount_point)
        self.iov_w = make_iovec(self.shm_w, self.hf3fs_mount_point)
        self.shm_r.unlink()
        self.shm_w.unlink()

    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        with self.rlock:
            return self._batch_io(offsets, tensors, for_read=True)

    def batch_write(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        with self.wlock:
            return self._batch_io(offsets, tensors, for_read=False)

    def _batch_io(
        self, offsets: List[int], tensors: List[torch.Tensor], *, for_read: bool
    ) -> List[int]:
        sizes = [tensor.numel() * tensor.itemsize for tensor in tensors]
        self._check_sizes(offsets, sizes)
        if not offsets:
            return []
        results = [0] * len(offsets)
        if self.file is None:
            return results

        if for_read:
            ring, iov, shm = self.ior_r, self.iov_r, self.shm_r_tensor
            state = self._read_state
        else:
            ring, iov, shm = self.ior_w, self.iov_w, self.shm_w_tensor
            state = self._write_state

        timeout = max(0, self.client_timeout)
        deadline = None
        if state.pending:
            # A timed-out I/O still owns its slice until its CQE is reaped.
            deadline = time.monotonic() + timeout
            self._reap(ring, state, timeout)
            if state.pending:
                return results

        if not for_read:
            try:
                hf3fs_utils.write_shm(tensors, shm)
            except Exception:
                logger.exception("Error copying HF3FS write data to shared memory")
                return results

        state.generation += 1
        state.needs_submit = True
        current = 0
        for index, (offset, size) in enumerate(zip(offsets, sizes)):
            token = (state.generation, index)
            # Keep a Python reference until wait returns the corresponding CQE.
            state.pending.add(token)
            try:
                ring.prepare(
                    iov[current : current + size],
                    for_read,
                    self.file,
                    offset,
                    userdata=token,
                )
            except Exception:
                state.pending.remove(token)
                logger.exception("Error preparing HF3FS I/O at offset %s", offset)
                break
            current += size

        if not state.pending:
            return results
        if deadline is not None:
            timeout = max(0, deadline - time.monotonic())
        # prepare enqueues immediately, so reap even when only a prefix was prepared.
        self._reap(ring, state, timeout, results)
        if for_read:
            self._copy_reads(shm, tensors, sizes, results)
        return results

    @staticmethod
    def _reap(
        ring, state: _RingState, timeout: float, results: List[int] | None = None
    ) -> None:
        if state.needs_submit:
            try:
                ring.submit()
            except Exception:
                logger.exception("Error notifying HF3FS of prepared I/Os")
            else:
                state.needs_submit = False

        try:
            completions = ring.wait(
                min_results=len(state.pending), timeout=timedelta(seconds=timeout)
            )
        except Exception:
            logger.exception("Error waiting for HF3FS I/O completions")
            return

        for completion in completions:
            token = completion.userdata
            if token in state.pending:
                state.pending.remove(token)
                if results is not None and token[0] == state.generation:
                    results[token[1]] = completion.result

    @staticmethod
    def _copy_reads(
        shm: torch.Tensor,
        tensors: List[torch.Tensor],
        sizes: List[int],
        results: List[int],
    ) -> None:
        if results == sizes and 0 not in sizes:
            try:
                hf3fs_utils.read_shm(shm, tensors)
            except Exception:
                logger.exception("Error copying HF3FS read data from shared memory")
                results[:] = [0] * len(results)
            return

        # The bulk helper packs tensors consecutively, so start each copy at its
        # original byte offset when a request failed, was short, or is still in flight.
        current = 0
        for index, (tensor, size) in enumerate(zip(tensors, sizes)):
            if size and results[index] == size:
                try:
                    hf3fs_utils.read_shm(shm[current : current + size], [tensor])
                except Exception:
                    logger.exception("Error copying HF3FS read data from shared memory")
                    results[index] = 0
            current += size

    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:
        self._check_sizes(
            offsets, [tensor.numel() * tensor.itemsize for tensor in tensors]
        )

    def _check_sizes(self, offsets: List[int], sizes: List[int]) -> None:
        if (
            len(offsets) > self.entries
            or len(offsets) != len(sizes)
            or any(
                offset < 0 or offset + size > self.size
                for offset, size in zip(offsets, sizes)
            )
            or any(size > self.bytes_per_page for size in sizes)
        ):
            raise ValueError(f"Hf3fsClient.check: {offsets=}, {sizes=}")

    def get_size(self) -> int:
        return self.size

    def close(self) -> None:
        if getattr(self, "file", None) is None:
            return
        with self.rlock, self.wlock, ExitStack() as cleanup:
            if self.file is None:
                return
            fd, self.file = self.file, None
            cleanup.callback(os.close, fd)
            if self._fd_registered:
                cleanup.callback(deregister_fd, fd)

            for suffix, state in (("r", self._read_state), ("w", self._write_state)):
                ring = self.__dict__.pop(f"ior_{suffix}", None)
                if ring is not None and state.pending:
                    self._reap(ring, state, timeout=0)
                del ring
                state.pending.clear()
                self.__dict__.pop(f"iov_{suffix}", None)
                self.__dict__.pop(f"shm_{suffix}_tensor", None)
                shm = self.__dict__.pop(f"shm_{suffix}", None)
                if shm is not None:
                    cleanup.callback(self._close_shm, shm)

    @staticmethod
    def _close_shm(shm: multiprocessing.shared_memory.SharedMemory) -> None:
        try:
            # Initialization may have failed before the original unlink.
            with suppress(FileNotFoundError):
                shm.unlink()
        finally:
            shm.close()

    def flush(self) -> None:
        with self.wlock:
            if self.file is None:
                raise ValueError("HF3FS client is closed")
            os.fsync(self.file)
