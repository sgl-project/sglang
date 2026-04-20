"""
Shared-memory transfer engine for the shm_pinned backend.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import time
from typing import Any, Optional

import numpy as np
import torch

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.disaggregation.shm_pinned.utils import (
    DEFAULT_SLOT_COUNT,
    SlotMeta,
    SlotState,
    ShmHeader,
    ShmPinnedInfo,
    calculate_meta_shm_size,
    calculate_slot_bytes,
    generate_shm_names,
    get_slot_meta_offset,
)

try:
    import posix_ipc
except ImportError:
    posix_ipc = None

logger = logging.getLogger(__name__)


def _require_posix_ipc():
    if posix_ipc is None:
        raise ImportError(
            "shm_pinned requires the optional dependency `posix_ipc` on Linux."
        )
    return posix_ipc


class ShmPinnedTransferEngine:
    def __init__(
        self,
        session_id: str,
        gpu_id: int,
        slot_count: int = DEFAULT_SLOT_COUNT,
        chunk_pages: Optional[int] = None,
        kv_item_lens: Optional[list[int]] = None,
        extra_slot_bytes: int = 0,
        create: bool = True,
    ):
        self._posix_ipc = _require_posix_ipc()
        self.session_id = session_id
        self.gpu_id = gpu_id
        self.slot_count = int(slot_count)
        self.chunk_pages = 0 if chunk_pages is None else int(chunk_pages)
        self.kv_item_lens = list(kv_item_lens or [])
        self.extra_slot_bytes = int(extra_slot_bytes)
        self.create = create
        self._closed = False

        if create and chunk_pages is None:
            raise ValueError("chunk_pages is required when create=True")

        self.data_shm: Any = None
        self.meta_shm: Any = None
        self.data_mmap: Optional[mmap.mmap] = None
        self.meta_mmap: Optional[mmap.mmap] = None

        self.sem_free: Any = None
        self.sem_ready: Any = None
        self.sem_slot: Any = None

        self.data_shm_name = ""
        self.meta_shm_name = ""
        self.sem_free_name = ""
        self.sem_ready_name = ""
        self.sem_slot_name = ""

        self.slot_bytes = 0
        self.meta_size = 0

        self._cuda_registered = False
        self._data_ptr = 0
        self._cudart: Optional[CudaRTLibrary] = None

        if create:
            self._create_shm()

    def _create_shm(self) -> None:
        (
            self.data_shm_name,
            self.meta_shm_name,
            self.sem_free_name,
            self.sem_ready_name,
            self.sem_slot_name,
        ) = generate_shm_names(self.session_id)
        self._calculate_sizes()

        try:
            self.data_shm = self._posix_ipc.SharedMemory(
                self.data_shm_name,
                flags=self._posix_ipc.O_CREX,
                size=self.slot_count * self.slot_bytes,
            )
            self.meta_size = calculate_meta_shm_size(self.slot_count)
            self.meta_shm = self._posix_ipc.SharedMemory(
                self.meta_shm_name,
                flags=self._posix_ipc.O_CREX,
                size=self.meta_size,
            )

            self.data_mmap = mmap.mmap(
                self.data_shm.fd,
                self.slot_count * self.slot_bytes,
            )
            self.data_shm.close_fd()

            self.meta_mmap = mmap.mmap(self.meta_shm.fd, self.meta_size)
            self.meta_shm.close_fd()

            self.meta_mmap.seek(0)
            self.meta_mmap.write(
                ShmHeader(slot_count=self.slot_count, slot_bytes=self.slot_bytes).pack()
            )
            for slot_idx in range(self.slot_count):
                self._write_slot_meta(slot_idx, SlotMeta())

            self.sem_free = self._posix_ipc.Semaphore(
                self.sem_free_name,
                flags=self._posix_ipc.O_CREX,
                initial_value=self.slot_count,
            )
            self.sem_ready = self._posix_ipc.Semaphore(
                self.sem_ready_name,
                flags=self._posix_ipc.O_CREX,
                initial_value=0,
            )
            self.sem_slot = self._posix_ipc.Semaphore(
                self.sem_slot_name,
                flags=self._posix_ipc.O_CREX,
                initial_value=1,
            )

            self._register_cuda()
        except Exception:
            self._cleanup()
            raise

    def _calculate_sizes(self) -> None:
        self.slot_bytes = calculate_slot_bytes(
            self.chunk_pages,
            self.kv_item_lens,
            extra_slot_bytes=self.extra_slot_bytes,
        )

    def _register_cuda(self) -> None:
        if self.data_mmap is None or not torch.cuda.is_available():
            return

        try:
            data_array = np.frombuffer(self.data_mmap, dtype=np.uint8)
            self._data_ptr = data_array.ctypes.data_as(ctypes.c_void_p).value
            torch.cuda.cudart().cudaHostRegister(
                self._data_ptr,
                self.slot_count * self.slot_bytes,
                0,
            )
            self._cuda_registered = True
        except Exception as e:
            logger.warning("Failed to register CUDA pinned memory: %s", e)

    def _unregister_cuda(self) -> None:
        if not self._cuda_registered or self._data_ptr == 0:
            return
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.cudart().cudaHostUnregister(self._data_ptr)
        except Exception as e:
            logger.warning("Failed to unregister CUDA memory: %s", e)
        finally:
            self._cuda_registered = False

    def open_from_info(self, info: ShmPinnedInfo) -> None:
        self.data_shm_name = info.data_shm_name
        self.meta_shm_name = info.meta_shm_name
        self.sem_free_name = info.sem_free_name
        self.sem_ready_name = info.sem_ready_name
        self.sem_slot_name = info.sem_slot_name
        self.slot_count = int(info.slot_count)
        self.slot_bytes = int(info.slot_bytes)
        self.kv_item_lens = list(info.kv_item_lens)

        try:
            self.data_shm = self._posix_ipc.SharedMemory(self.data_shm_name)
            self.meta_shm = self._posix_ipc.SharedMemory(self.meta_shm_name)

            self.data_mmap = mmap.mmap(
                self.data_shm.fd,
                self.slot_count * self.slot_bytes,
            )
            self.data_shm.close_fd()

            self.meta_size = calculate_meta_shm_size(self.slot_count)
            self.meta_mmap = mmap.mmap(self.meta_shm.fd, self.meta_size)
            self.meta_shm.close_fd()

            if not self._read_header().validate():
                raise RuntimeError("Invalid shared memory header")

            self.sem_free = self._posix_ipc.Semaphore(self.sem_free_name)
            self.sem_ready = self._posix_ipc.Semaphore(self.sem_ready_name)
            self.sem_slot = (
                self._posix_ipc.Semaphore(self.sem_slot_name)
                if self.sem_slot_name
                else None
            )
            self._register_cuda()
        except Exception:
            self._cleanup()
            raise

    def export_info(self) -> ShmPinnedInfo:
        return ShmPinnedInfo(
            data_shm_name=self.data_shm_name,
            meta_shm_name=self.meta_shm_name,
            sem_free_name=self.sem_free_name,
            sem_ready_name=self.sem_ready_name,
            sem_slot_name=self.sem_slot_name,
            slot_count=self.slot_count,
            slot_bytes=self.slot_bytes,
            session_id=self.session_id,
            kv_item_lens=self.kv_item_lens,
        )

    def _read_header(self) -> ShmHeader:
        if self.meta_mmap is None:
            raise RuntimeError("Meta shared memory not mapped")
        self.meta_mmap.seek(0)
        return ShmHeader.unpack(self.meta_mmap.read(ShmHeader.size()))

    def _write_slot_meta(self, slot_idx: int, meta: SlotMeta) -> None:
        if self.meta_mmap is None:
            raise RuntimeError("Meta shared memory not mapped")
        self.meta_mmap.seek(get_slot_meta_offset(slot_idx))
        self.meta_mmap.write(meta.pack())

    def _read_slot_meta(self, slot_idx: int) -> SlotMeta:
        if self.meta_mmap is None:
            raise RuntimeError("Meta shared memory not mapped")
        self.meta_mmap.seek(get_slot_meta_offset(slot_idx))
        return SlotMeta.unpack(self.meta_mmap.read(SlotMeta.size()))

    def get_slot_data_ptr(self, slot_idx: int) -> int:
        if self._data_ptr == 0:
            raise RuntimeError("Data pointer not available")
        return self._data_ptr + int(slot_idx) * self.slot_bytes

    def _acquire_semaphore(self, sem: Any, timeout: Optional[float]) -> None:
        if timeout is None:
            sem.acquire()
        else:
            sem.acquire(timeout)

    def _remaining_timeout(self, deadline: Optional[float]) -> Optional[float]:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def wait_free(self, timeout: Optional[float] = None) -> int:
        if self.sem_free is None:
            raise RuntimeError("Semaphore not initialized")

        deadline = None if timeout is None else time.monotonic() + timeout
        self._acquire_semaphore(self.sem_free, timeout)
        if self.sem_slot is not None:
            try:
                self._acquire_semaphore(
                    self.sem_slot,
                    self._remaining_timeout(deadline),
                )
            except self._posix_ipc.BusyError:
                self.sem_free.release()
                raise

        try:
            for slot_idx in range(self.slot_count):
                meta = self._read_slot_meta(slot_idx)
                if meta.state == SlotState.FREE:
                    meta.state = SlotState.WRITING
                    self._write_slot_meta(slot_idx, meta)
                    return slot_idx
        finally:
            if self.sem_slot is not None:
                self.sem_slot.release()

        self.sem_free.release()
        raise RuntimeError("No free slot found despite semaphore signaling")

    def post_free(self, slot_idx: int) -> None:
        if self.sem_free is None:
            raise RuntimeError("Semaphore not initialized")
        meta = self._read_slot_meta(slot_idx)
        meta.state = SlotState.FREE
        self._write_slot_meta(slot_idx, meta)
        self.sem_free.release()

    def wait_ready(self, timeout: Optional[float] = None) -> int:
        if self.sem_ready is None:
            raise RuntimeError("Semaphore not initialized")

        deadline = None if timeout is None else time.monotonic() + timeout
        self._acquire_semaphore(self.sem_ready, timeout)
        if self.sem_slot is not None:
            try:
                self._acquire_semaphore(
                    self.sem_slot,
                    self._remaining_timeout(deadline),
                )
            except self._posix_ipc.BusyError:
                self.sem_ready.release()
                raise

        try:
            for slot_idx in range(self.slot_count):
                meta = self._read_slot_meta(slot_idx)
                if meta.state == SlotState.READY:
                    meta.state = SlotState.READING
                    self._write_slot_meta(slot_idx, meta)
                    return slot_idx
        finally:
            if self.sem_slot is not None:
                self.sem_slot.release()

        self.sem_ready.release()
        raise RuntimeError("No ready slot found despite semaphore signaling")

    def post_ready(self, slot_idx: int) -> None:
        if self.sem_ready is None:
            raise RuntimeError("Semaphore not initialized")
        meta = self._read_slot_meta(slot_idx)
        meta.state = SlotState.READY
        self._write_slot_meta(slot_idx, meta)
        self.sem_ready.release()

    def _get_cudart(self) -> CudaRTLibrary:
        if self._cudart is None:
            self._cudart = CudaRTLibrary()
        return self._cudart

    def cuda_memcpy(self, dst_ptr: int, src_ptr: int, num_bytes: int) -> None:
        if num_bytes <= 0:
            return
        cudart = self._get_cudart()
        cudart.cudaSetDevice(self.gpu_id)
        cudart.cudaMemcpy(
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(src_ptr),
            int(num_bytes),
        )

    def write_meta(
        self,
        slot_idx: int,
        room: int,
        index_start: int,
        index_len: int,
        is_last: bool,
        valid_bytes: int,
        seqno: int = 0,
    ) -> None:
        self._write_slot_meta(
            slot_idx,
            SlotMeta(
                state=SlotState.WRITING,
                room=int(room),
                index_start=int(index_start),
                index_len=int(index_len),
                is_last=int(is_last),
                valid_bytes=int(valid_bytes),
                seqno=int(seqno),
                owner_pid=os.getpid(),
            ),
        )

    def read_meta(self, slot_idx: int) -> SlotMeta:
        return self._read_slot_meta(slot_idx)

    def _cleanup(self) -> None:
        self._unregister_cuda()

        if self.data_mmap is not None:
            try:
                self.data_mmap.close()
            except Exception:
                pass
            self.data_mmap = None

        if self.meta_mmap is not None:
            try:
                self.meta_mmap.close()
            except Exception:
                pass
            self.meta_mmap = None

        if self.create:
            for name in (self.data_shm_name, self.meta_shm_name):
                if not name:
                    continue
                try:
                    self._posix_ipc.unlink_shared_memory(name)
                except Exception:
                    pass

            for name in (self.sem_free_name, self.sem_ready_name, self.sem_slot_name):
                if not name:
                    continue
                try:
                    self._posix_ipc.Semaphore(name).unlink()
                except Exception:
                    pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
