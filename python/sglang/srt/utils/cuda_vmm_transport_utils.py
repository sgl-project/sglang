from __future__ import annotations

import logging
import os
import secrets
import socket
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.runtime_context import (
    get_mm,
    get_parallel,
)
from sglang.srt.utils.cuda_ipc_transport_utils import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    MM_FEATURE_CACHE_SIZE,
    MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
    CudaIpcTensorTransportProxy,
    get_mm_feature_pool_size_per_worker,
)
from sglang.srt.utils.cuda_vmm_utils import (
    _FD_SEND_TIMEOUT_S,
    VmmReservation,
    _get_cuda_driver,
    _recv_fd,
    _send_fd,
    align_up,
    allocation_handle_type_name,
    check_drv,
    get_allocation_granularity,
    get_device_allocation_handle_type,
    import_and_map_alloc,
    make_device_allocation_prop,
    release_mappings,
    tensor_from_pointer,
)

logger = logging.getLogger(__name__)

_CONTROL_ALIGNMENT = 256
_CONTROL_WORD_BYTES = 4


class _PosixFdBroker:
    """Serve one exported CUDA allocation FD to local consumer processes."""

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self._stop = threading.Event()
        self._error: Exception | None = None
        self.socket_path = f"\0sgl_mm_vmm_{secrets.token_hex(16)}"
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._server.bind(self.socket_path)
        self._server.listen()
        self._server.settimeout(0.1)
        self._thread = threading.Thread(
            target=self._serve,
            name="CudaVmmPosixFdBroker",
            daemon=True,
        )
        try:
            self._thread.start()
        except BaseException:
            self._server.close()
            raise

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError as error:
                if self._stop.is_set():
                    return
                self._error = error
                logger.exception("CUDA VMM POSIX FD broker failed")
                return

            try:
                with conn:
                    _send_fd(conn, self.fd, src_rank=0, base_idx=0)
            except Exception as error:
                self._error = error
                logger.exception("CUDA VMM POSIX FD broker failed")
                return

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("CUDA VMM POSIX FD broker failed") from self._error

    def close(self) -> None:
        self._stop.set()
        self._server.close()
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            raise RuntimeError("CUDA VMM POSIX FD broker did not stop")


def _receive_posix_fd(socket_path: str) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
        sock.settimeout(_FD_SEND_TIMEOUT_S)
        sock.connect(socket_path)
        packet = _recv_fd(sock)
    if packet is None:
        raise RuntimeError("CUDA VMM POSIX FD broker returned no file descriptor")
    _src_rank, _base_idx, fd = packet
    return fd


@dataclass
class _CudaVmmMemoryChunk:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class _CudaVmmPackedTensorLayout:
    relative_offset: int
    data_nbytes: int
    shape: torch.Size
    dtype: torch.dtype


def _build_packed_tensor_layout(
    tensors: Sequence[torch.Tensor],
) -> tuple[list[_CudaVmmPackedTensorLayout], int]:
    layouts = []
    next_offset = 0
    for tensor in tensors:
        next_offset = align_up(next_offset, tensor.element_size())
        data_nbytes = tensor.numel() * tensor.element_size()
        layouts.append(
            _CudaVmmPackedTensorLayout(
                relative_offset=next_offset,
                data_nbytes=data_nbytes,
                shape=tensor.shape,
                dtype=tensor.dtype,
            )
        )
        next_offset += data_nbytes
    return layouts, next_offset


def _contains_tensor_container(value) -> bool:
    return isinstance(value, (list, tuple)) and any(
        isinstance(item, torch.Tensor) or _contains_tensor_container(item)
        for item in value
    )


def get_vmm_feature_consumer_count() -> int:
    if get_parallel().enable_dp_attention:
        return get_parallel().tp_size // get_parallel().dp_size
    return get_parallel().tp_size


class CudaVmmMemoryPool:
    """Bounded CUDA VMM pool shared through FABRIC or a local POSIX FD."""

    def __init__(
        self,
        memory_size: int,
        recycle_interval: float,
        base_gpu_id: int,
        consumer_count: int,
        allow_posix_fallback: bool = False,
    ) -> None:
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if consumer_count <= 0:
            raise ValueError("consumer_count must be positive")
        if recycle_interval <= 0:
            raise ValueError("recycle_interval must be positive")

        self.device_index = int(base_gpu_id)
        self.consumer_count = int(consumer_count)
        self._recycle_interval = float(recycle_interval)
        self._lock = threading.Lock()
        self._publisher_condition = threading.Condition(self._lock)
        self._shutdown_lock = threading.Lock()
        self._active_publishers = 0
        self._closing = False
        self._pool_full_warned = False
        self._stop_recycler = threading.Event()
        self._pool_error: BaseException | None = None
        self._closed = False

        self._allocation: VmmReservation | None = None
        self.allocation_size = 0
        self.shareable_handle = None
        self.memory_pool = None
        self._fd_broker: _PosixFdBroker | None = None
        self.posix_socket_path: str | None = None
        self._recycle_stream = None
        self._recycle_thread = None

        drv = _get_cuda_driver()
        fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        posix_fd = (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        self.handle_type = get_device_allocation_handle_type(self.device_index)
        if self.handle_type == posix_fd and not allow_posix_fallback:
            raise RuntimeError(
                "CUDA VMM multimodal transport selected POSIX_FD, but this "
                "pool requires FABRIC"
            )
        self.use_fabric = self.handle_type == fabric
        try:
            self._allocate(memory_size)
        except RuntimeError as error:
            if not allow_posix_fallback or self.handle_type != fabric:
                raise
            logger.warning(
                "CUDA FABRIC VMM allocation is unavailable; falling back to "
                "a POSIX FD handle: %s",
                error,
            )
            self.handle_type = posix_fd
            self.use_fabric = False
            self._allocate(memory_size)
        try:
            if not self.use_fabric:
                self._fd_broker = _PosixFdBroker(self.shareable_handle)
                self.posix_socket_path = self._fd_broker.socket_path

            self.available_chunks = [_CudaVmmMemoryChunk(0, self.allocation_size)]
            self.occupied_chunks = []
            self._recycle_stream = torch.cuda.Stream(device=self.device_index)
            self._recycle_thread = threading.Thread(
                target=self._recycle_loop,
                name="CudaVmmMemoryPoolRecycler",
                daemon=True,
            )
            self._recycle_thread.start()
        except BaseException as error:
            cleanup_errors = []
            self._stop_recycler.set()
            if self._recycle_thread is not None and self._recycle_thread.is_alive():
                self._recycle_thread.join(timeout=1.0)
                if self._recycle_thread.is_alive():
                    cleanup_errors.append(
                        RuntimeError("CUDA VMM recycler did not stop during rollback")
                    )
            if self._fd_broker is not None:
                try:
                    self._fd_broker.close()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            try:
                self._release_allocation()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                error.add_note(
                    f"{len(cleanup_errors)} CUDA VMM initialization rollback "
                    "operation(s) also failed"
                )
                raise error from cleanup_errors[0]
            raise

    @property
    def fabric_handle(self) -> bytes | None:
        return self.shareable_handle if self.use_fabric else None

    def _allocate(self, memory_size: int) -> None:
        drv = _get_cuda_driver()
        prop = make_device_allocation_prop(
            self.device_index,
            handle_types=self.handle_type,
            gpu_direct_rdma=self.use_fabric,
        )

        with torch.cuda.device(self.device_index):
            granularity = get_allocation_granularity(prop)
            allocation_size = memory_size // granularity * granularity
            if allocation_size == 0:
                raise ValueError(
                    f"memory_size={memory_size} is smaller than CUDA VMM "
                    f"granularity={granularity}"
                )

            allocation = VmmReservation(
                allocation_size,
                prop,
                self.device_index,
                alignment=granularity,
            )
            exported = None
            try:
                handle = allocation.map(
                    0,
                    allocation_size,
                    retain_handle=True,
                )
                exported = check_drv(
                    drv.cuMemExportToShareableHandle(handle, self.handle_type, 0),
                    "cuMemExportToShareableHandle(VMM transport)",
                )
                memory_pool = tensor_from_pointer(
                    allocation.base, allocation_size, device_id=self.device_index
                )
            except BaseException:
                allocation.close()
                if not self.use_fabric and exported is not None:
                    os.close(int(exported))
                raise

        self._allocation = allocation
        self.allocation_size = allocation_size
        self.shareable_handle = (
            bytes(exported.data) if self.use_fabric else int(exported)
        )
        logger.info(
            "CUDA VMM multimodal pool uses %s backing on device %d",
            allocation_handle_type_name(self.handle_type),
            self.device_index,
        )
        self.memory_pool = memory_pool

    @property
    def control_size(self) -> int:
        return align_up(self.consumer_count * _CONTROL_WORD_BYTES, _CONTROL_ALIGNMENT)

    def _raise_if_failed(self) -> None:
        if self._pool_error is not None:
            raise RuntimeError("CUDA VMM multimodal pool failed") from self._pool_error
        if self._fd_broker is not None:
            self._fd_broker.raise_if_failed()

    def _reserve_for_publish(self, required_size: int) -> _CudaVmmMemoryChunk | None:
        with self._publisher_condition:
            self._raise_if_failed()
            if self._closing or self._closed:
                raise RuntimeError("CUDA VMM multimodal pool is closing")
            chunk = self._reserve_chunk(required_size)
            if chunk is not None:
                self._active_publishers += 1
            return chunk

    def _finish_publish(self) -> None:
        with self._publisher_condition:
            self._active_publishers -= 1
            if self._active_publishers == 0:
                self._publisher_condition.notify_all()

    def wrap_tensor(self, tensor: torch.Tensor):
        self._raise_if_failed()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        data_nbytes = tensor.numel() * tensor.element_size()
        required_size = align_up(self.control_size + data_nbytes, _CONTROL_ALIGNMENT)
        source_bytes = tensor.reshape(-1).view(torch.uint8)

        chunk = self._reserve_for_publish(required_size)
        if chunk is None:
            self._warn_pool_full_once(data_nbytes)
            return tensor.cpu()

        producer_stream = None
        copy_synchronized = False
        try:
            with torch.cuda.device(self.device_index):
                producer_stream = torch.cuda.current_stream(self.device_index)
                control_offset = chunk.start
                data_offset = control_offset + self.control_size
                control_end = control_offset + self.control_size
                self.memory_pool[control_offset:control_end].zero_()
                self.memory_pool[data_offset : data_offset + data_nbytes].copy_(
                    source_bytes, non_blocking=True
                )
                # Imported VMM memory does not support cuStreamWaitValue32 on
                # current GB-class drivers, so publish only after this copy.
                producer_stream.synchronize()
                copy_synchronized = True

            proxy = CudaVmmTensorTransportProxy(
                fabric_handle=self.fabric_handle,
                posix_socket_path=self.posix_socket_path,
                allocation_size=self.allocation_size,
                data_offset=data_offset,
                data_nbytes=data_nbytes,
                control_offset=control_offset,
                consumer_count=self.consumer_count,
                shape=tensor.shape,
                dtype=tensor.dtype,
            )
            with self._lock:
                self.occupied_chunks.append(chunk)
            return proxy
        except BaseException:
            safe_to_release = copy_synchronized or producer_stream is None
            if not safe_to_release:
                try:
                    producer_stream.synchronize()
                    safe_to_release = True
                except BaseException as cleanup_error:
                    self._pool_error = cleanup_error
            if safe_to_release:
                with self._lock:
                    self._release_reserved_chunk(chunk)
            raise
        finally:
            self._finish_publish()

    def wrap_tensors(
        self,
        tensors: Sequence[torch.Tensor],
    ) -> list[CudaVmmPackedTensorTransportProxy] | None:
        """Publish tensors through one shared VMM chunk.

        Every tensor must have the same dispatch and consumer lifetime because
        reconstructing one child copies and acknowledges the full packed chunk.
        ``None`` means that no contiguous pool chunk was available. No tensor is
        published in that case, so the caller can use another transport path.
        """
        self._raise_if_failed()
        tensors = list(tensors)
        if not tensors:
            return []

        layouts, packed_data_nbytes = _build_packed_tensor_layout(tensors)
        required_size = align_up(
            self.control_size + packed_data_nbytes, _CONTROL_ALIGNMENT
        )
        chunk = self._reserve_for_publish(required_size)
        if chunk is None:
            return None

        producer_stream = None
        copy_synchronized = False
        try:
            contiguous_tensors = [
                tensor if tensor.is_contiguous() else tensor.contiguous()
                for tensor in tensors
            ]
            with torch.cuda.device(self.device_index):
                producer_stream = torch.cuda.current_stream(self.device_index)
                control_offset = chunk.start
                data_offset = control_offset + self.control_size
                control_end = control_offset + self.control_size
                self.memory_pool[control_offset:control_end].zero_()
                for tensor, layout in zip(contiguous_tensors, layouts):
                    data_start = data_offset + layout.relative_offset
                    self.memory_pool[
                        data_start : data_start + layout.data_nbytes
                    ].copy_(tensor.reshape(-1).view(torch.uint8), non_blocking=True)
                # A single synchronization publishes every child together.
                producer_stream.synchronize()
                copy_synchronized = True

            owner = _CudaVmmPackedTransportOwner(
                fabric_handle=self.fabric_handle,
                posix_socket_path=self.posix_socket_path,
                allocation_size=self.allocation_size,
                data_offset=data_offset,
                data_nbytes=packed_data_nbytes,
                control_offset=control_offset,
                consumer_count=self.consumer_count,
            )
            proxies = [
                CudaVmmPackedTensorTransportProxy(
                    owner=owner,
                    layout=layout,
                )
                for layout in layouts
            ]
            with self._lock:
                self.occupied_chunks.append(chunk)
            return proxies
        except BaseException:
            safe_to_release = copy_synchronized or producer_stream is None
            if not safe_to_release:
                try:
                    producer_stream.synchronize()
                    safe_to_release = True
                except BaseException as cleanup_error:
                    self._pool_error = cleanup_error
            if safe_to_release:
                with self._lock:
                    self._release_reserved_chunk(chunk)
            raise
        finally:
            self._finish_publish()

    def _reserve_chunk(self, required_size: int) -> _CudaVmmMemoryChunk | None:
        candidates = [
            chunk for chunk in self.available_chunks if chunk.size >= required_size
        ]
        if not candidates:
            return None

        available = min(candidates, key=lambda chunk: chunk.size)
        self.available_chunks.remove(available)
        occupied = _CudaVmmMemoryChunk(
            start=available.start,
            end=available.start + required_size,
        )
        if occupied.end < available.end:
            self.available_chunks.append(
                _CudaVmmMemoryChunk(occupied.end, available.end)
            )
        return occupied

    def _release_reserved_chunk(self, chunk: _CudaVmmMemoryChunk) -> None:
        if chunk in self.occupied_chunks:
            self.occupied_chunks.remove(chunk)
        self.available_chunks.append(_CudaVmmMemoryChunk(chunk.start, chunk.end))
        self._merge_chunks()

    def _cancel_control_offset(self, control_offset: int) -> None:
        with self._lock:
            chunk = next(
                (
                    chunk
                    for chunk in self.occupied_chunks
                    if chunk.start == control_offset
                ),
                None,
            )
            if chunk is None:
                raise RuntimeError(
                    "CUDA VMM pool has no occupied slice at control offset "
                    f"{control_offset}"
                )
            self._release_reserved_chunk(chunk)

    def cancel_proxy(self, proxy: CudaVmmTensorTransportProxy) -> None:
        """Return a published slice when its request was never dispatched."""
        if isinstance(proxy, CudaVmmPackedTensorTransportProxy):
            proxy._packed_owner.cancel_from_pool(self)
            return
        self._cancel_control_offset(proxy.control_offset)

    def _warn_pool_full_once(self, data_nbytes: int) -> None:
        if self._pool_full_warned:
            return
        self._pool_full_warned = True
        logger.warning(
            "CUDA VMM multimodal pool has no free chunk for a %.2f MiB tensor "
            "(pool size: %.2f MiB); falling back to CPU transport. Increase "
            "SGLANG_MM_FEATURE_CACHE_MB to avoid inline request broadcasts.",
            data_nbytes / (1024 * 1024),
            self.allocation_size / (1024 * 1024),
        )

    def _recycle_loop(self) -> None:
        while not self._stop_recycler.wait(self._recycle_interval):
            try:
                with self._lock:
                    self._recycle_chunks()
                    self._merge_chunks()
            except Exception as error:
                logger.exception("CUDA VMM multimodal pool recycle failed")
                self._pool_error = error
                self._stop_recycler.set()

    def _recycle_chunks(self) -> None:
        remaining = []
        recycled = []
        with (
            torch.cuda.device(self.device_index),
            torch.cuda.stream(self._recycle_stream),
        ):
            for chunk in self.occupied_chunks:
                ack_start = chunk.start
                ack_end = ack_start + self.consumer_count * _CONTROL_WORD_BYTES
                ack_count = int(
                    torch.count_nonzero(
                        self.memory_pool[ack_start:ack_end].view(torch.int32)
                    ).item()
                )
                if ack_count == self.consumer_count:
                    recycled.append(_CudaVmmMemoryChunk(chunk.start, chunk.end))
                else:
                    remaining.append(chunk)

        self.available_chunks.extend(recycled)
        self.occupied_chunks = remaining

    def _merge_chunks(self) -> None:
        merged = []
        for chunk in sorted(self.available_chunks, key=lambda item: item.start):
            if merged and merged[-1].end == chunk.start:
                merged[-1].end = chunk.end
            else:
                merged.append(chunk)
        self.available_chunks = merged

    def _release_allocation(self) -> None:
        self.memory_pool = None
        if not self.use_fabric and self.shareable_handle is not None:
            os.close(self.shareable_handle)
            self.shareable_handle = None
        if self._allocation is None:
            return
        with torch.cuda.device(self.device_index):
            self._allocation.close()
        self._allocation = None

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._closed:
                return
            with self._publisher_condition:
                self._closing = True
                while self._active_publishers:
                    self._publisher_condition.wait()

            self._stop_recycler.set()
            self._recycle_thread.join(timeout=1.0)
            if self._recycle_thread.is_alive():
                raise RuntimeError("CUDA VMM recycler did not stop")
            if self._fd_broker is not None:
                self._fd_broker.close()
                self._fd_broker = None
            self._release_allocation()
            self._closed = True


@dataclass
class _ImportedCudaVmmPool:
    pointer: int
    allocation_size: int
    memory: torch.Tensor | None

    def close(self) -> None:
        self.memory = None
        release_mappings(
            [
                (
                    self.pointer,
                    self.allocation_size,
                    [(0, self.allocation_size)],
                )
            ]
        )


_imported_pool_cache: dict[tuple, _ImportedCudaVmmPool] = {}
_imported_pool_cache_lock = threading.Lock()


def _get_imported_pool(
    *,
    fabric_handle: bytes | None,
    posix_socket_path: str | None,
    allocation_size: int,
    device_index: int,
) -> _ImportedCudaVmmPool:
    use_fabric = fabric_handle is not None
    transport_handle = fabric_handle if use_fabric else posix_socket_path
    if transport_handle is None:
        raise RuntimeError("CUDA VMM proxy has no shareable handle")
    key = (device_index, allocation_size, transport_handle)
    pool = _imported_pool_cache.get(key)
    if pool is not None:
        return pool

    with _imported_pool_cache_lock:
        pool = _imported_pool_cache.get(key)
        if pool is not None:
            return pool

        fd = None
        try:
            if not use_fabric:
                fd = _receive_posix_fd(posix_socket_path)
            with torch.cuda.device(device_index):
                pointer = import_and_map_alloc(
                    fabric_handle,
                    fd,
                    allocation_size,
                    device_index,
                    use_fabric=use_fabric,
                    peer_rank=-1,
                )
                try:
                    memory = tensor_from_pointer(
                        pointer, allocation_size, device_id=device_index
                    )
                except Exception:
                    release_mappings(
                        [(pointer, allocation_size, [(0, allocation_size)])]
                    )
                    raise
        finally:
            if fd is not None:
                os.close(fd)

        pool = _ImportedCudaVmmPool(
            pointer=pointer,
            allocation_size=allocation_size,
            memory=memory,
        )
        _imported_pool_cache[key] = pool
        return pool


def _imported_pool_cache_clear() -> None:
    with _imported_pool_cache_lock:
        pools = list(_imported_pool_cache.values())
        _imported_pool_cache.clear()
    for pool in pools:
        pool.close()


class CudaVmmTensorTransportProxy(CudaIpcTensorTransportProxy):
    """Multimodal tensor proxy backed by a shared CUDA VMM pool."""

    def __init__(
        self,
        *,
        fabric_handle: bytes | None,
        posix_socket_path: str | None,
        allocation_size: int,
        data_offset: int,
        data_nbytes: int,
        control_offset: int,
        consumer_count: int,
        shape,
        dtype,
    ) -> None:
        self.fabric_handle = fabric_handle
        self.posix_socket_path = posix_socket_path
        self.allocation_size = allocation_size
        self.data_offset = data_offset
        self.data_nbytes = data_nbytes
        self.control_offset = control_offset
        self.consumer_count = consumer_count
        self.shape = shape
        self.dtype = dtype
        self.reconstruct_tensor = None
        self._consumer_acknowledged = False

    def _pool(self, device_index: int) -> _ImportedCudaVmmPool:
        return _get_imported_pool(
            fabric_handle=self.fabric_handle,
            posix_socket_path=self.posix_socket_path,
            allocation_size=self.allocation_size,
            device_index=device_index,
        )

    def _acknowledgement_range(self, consumer_count: int) -> tuple[int, int]:
        if consumer_count <= 0:
            raise ValueError("consumer_count must be positive")
        if consumer_count == self.consumer_count:
            return 0, self.consumer_count

        parallel = get_parallel()
        group_start = parallel.attn_cp_rank * parallel.attn_tp_size
        group_end = group_start + parallel.attn_tp_size
        if not 0 <= group_start < group_end <= self.consumer_count:
            raise ValueError(
                "attention group range "
                f"[{group_start}, {group_end}) is outside "
                f"consumer_count={self.consumer_count}"
            )
        if consumer_count == 1:
            slot = group_start + parallel.attn_tp_rank
            return slot, slot + 1
        if consumer_count == parallel.attn_tp_size:
            return group_start, group_end
        raise ValueError(
            "consumer_count must be 1, the attention TP size, or the full "
            f"consumer count ({self.consumer_count}); got {consumer_count}"
        )

    def _resolve_consumer_count(self, consumer_count: int | None) -> int:
        return 1 if consumer_count is None else consumer_count

    def _acknowledge_consumption(self, device_index: int, consumer_count: int) -> None:
        if self._consumer_acknowledged:
            return
        pool = self._pool(device_index)
        ack_start = self.control_offset
        ack_end = ack_start + self.consumer_count * _CONTROL_WORD_BYTES
        ack_words = pool.memory[ack_start:ack_end].view(torch.int32)
        slot_start, slot_end = self._acknowledgement_range(consumer_count)
        # This kernel is ordered after the remote read on the consumer stream;
        # observing the flag therefore means the pool slice is safe to reuse.
        ack_words[slot_start:slot_end].fill_(1)
        self._consumer_acknowledged = True

    def acknowledge_consumption(self, consumer_count: int | None = None) -> None:
        consumer_count = self._resolve_consumer_count(consumer_count)
        device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            self._acknowledge_consumption(device_index, consumer_count)

    def reconstruct_on_target_device(
        self, rebuild_device_idx, consumer_count: int | None = None
    ):
        consumer_count = self._resolve_consumer_count(consumer_count)
        rebuild_device = torch.device(f"cuda:{rebuild_device_idx}")
        if (
            isinstance(self.reconstruct_tensor, torch.Tensor)
            and self.reconstruct_tensor.device == rebuild_device
        ):
            return self.reconstruct_tensor
        if self._consumer_acknowledged:
            raise RuntimeError("CUDA VMM tensor has already released its pool slice")

        pool = self._pool(rebuild_device_idx)
        try:
            with torch.cuda.device(rebuild_device):
                source = pool.memory[
                    self.data_offset : self.data_offset + self.data_nbytes
                ]
                reconstructed = torch.empty(
                    self.shape, dtype=self.dtype, device=rebuild_device
                ).contiguous()
                reconstructed.reshape(-1).view(torch.uint8).copy_(
                    source, non_blocking=True
                )
                self._acknowledge_consumption(rebuild_device_idx, consumer_count)
        except BaseException as error:
            try:
                with torch.cuda.device(rebuild_device):
                    self._acknowledge_consumption(rebuild_device_idx, consumer_count)
            except BaseException as cleanup_error:
                error.add_note(
                    "CUDA VMM reconstruction cleanup also failed; the pool "
                    "slice was not acknowledged"
                )
                raise error from cleanup_error
            raise

        self.reconstruct_tensor = reconstructed
        return reconstructed


class _CudaVmmPackedTransportOwner(CudaVmmTensorTransportProxy):
    def __init__(
        self,
        *,
        fabric_handle: bytes | None,
        posix_socket_path: str | None,
        allocation_size: int,
        data_offset: int,
        data_nbytes: int,
        control_offset: int,
        consumer_count: int,
    ) -> None:
        super().__init__(
            fabric_handle=fabric_handle,
            posix_socket_path=posix_socket_path,
            allocation_size=allocation_size,
            data_offset=data_offset,
            data_nbytes=data_nbytes,
            control_offset=control_offset,
            consumer_count=consumer_count,
            shape=(data_nbytes,),
            dtype=torch.uint8,
        )
        self._producer_cancelled = False

    def cancel_from_pool(self, pool: CudaVmmMemoryPool) -> None:
        if self._producer_cancelled:
            return
        pool._cancel_control_offset(self.control_offset)
        self._producer_cancelled = True


class CudaVmmPackedTensorTransportProxy(CudaVmmTensorTransportProxy):
    """One typed view within a packed CUDA VMM transfer."""

    def __init__(
        self,
        *,
        owner: _CudaVmmPackedTransportOwner,
        layout: _CudaVmmPackedTensorLayout,
    ) -> None:
        super().__init__(
            fabric_handle=owner.fabric_handle,
            posix_socket_path=owner.posix_socket_path,
            allocation_size=owner.allocation_size,
            data_offset=owner.data_offset + layout.relative_offset,
            data_nbytes=layout.data_nbytes,
            control_offset=owner.control_offset,
            consumer_count=owner.consumer_count,
            shape=layout.shape,
            dtype=layout.dtype,
        )
        self._packed_owner = owner
        self._packed_relative_offset = layout.relative_offset

    def acknowledge_consumption(self, consumer_count: int | None = None) -> None:
        raise RuntimeError(
            "Packed CUDA VMM features must be reconstructed before release"
        )

    def reconstruct_on_target_device(
        self, rebuild_device_idx, consumer_count: int | None = None
    ):
        rebuild_device = torch.device(f"cuda:{rebuild_device_idx}")
        if (
            isinstance(self.reconstruct_tensor, torch.Tensor)
            and self.reconstruct_tensor.device == rebuild_device
        ):
            return self.reconstruct_tensor
        if self._consumer_acknowledged:
            raise RuntimeError("CUDA VMM tensor has already released its pool slice")

        packed_buffer = self._packed_owner.reconstruct_on_target_device(
            rebuild_device_idx, consumer_count=consumer_count
        )
        tensor_bytes = packed_buffer[
            self._packed_relative_offset : self._packed_relative_offset
            + self.data_nbytes
        ]
        reconstructed = tensor_bytes.view(self.dtype).reshape(self.shape)
        self.reconstruct_tensor = reconstructed
        self._consumer_acknowledged = True
        return reconstructed


class CudaVmmFeatureTransport:
    """Tokenizer-owned VMM transport for one tokenizer worker."""

    def __init__(self, server_args, mm_processor) -> None:
        self.pool: CudaVmmMemoryPool | None = None
        if get_mm().mm_feature_transport != "cuda_vmm":
            return
        if mm_processor is None:
            raise RuntimeError(
                "A CUDA VMM-enabled model must provide a multimodal processor"
            )

        per_worker_pool_size = get_mm_feature_pool_size_per_worker(
            MM_FEATURE_CACHE_SIZE, server_args.tokenizer_worker_num
        )
        self.pool = CudaVmmMemoryPool(
            memory_size=per_worker_pool_size,
            recycle_interval=MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
            base_gpu_id=server_args.base_gpu_id,
            consumer_count=get_vmm_feature_consumer_count(),
            allow_posix_fallback=server_args.nnodes == 1,
        )

    def prepare_for_dispatch(
        self,
        mm_inputs_batch: Iterable[MultimodalProcessorOutput | None],
    ) -> list[MultimodalDataItem]:
        if self.pool is None:
            return []

        prepared_mm_items = []
        preparation_complete = False
        try:
            for mm_inputs in mm_inputs_batch:
                if mm_inputs is None or not mm_inputs.mm_items:
                    continue
                mm_items = mm_inputs.mm_items
                self.wrap_items(mm_items)
                prepared_mm_items.extend(mm_items)
            preparation_complete = True
            return prepared_mm_items
        finally:
            if not preparation_complete:
                self.cancel_for_dispatch(prepared_mm_items)

    def wrap_items(self, mm_items: list[MultimodalDataItem]) -> None:
        if self.pool is None:
            return

        updates = []
        try:
            pack_candidates = [
                (item, item.feature)
                for item in mm_items
                if item.modality == Modality.IMAGE
                and isinstance(item.feature, torch.Tensor)
                and item.feature.numel() > 0
                and not item.model_specific_data.get(
                    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY, False
                )
            ]
            if len(pack_candidates) >= 2:
                packed = self.pool.wrap_tensors(
                    [tensor for _, tensor in pack_candidates]
                )
                if packed is not None:
                    for (item, tensor), proxy in zip(
                        pack_candidates, packed, strict=True
                    ):
                        item.feature = proxy
                        updates.append((item, "feature", tensor, proxy))

            for item in mm_items:
                fields = (
                    ("feature", item.feature),
                    ("precomputed_embeddings", item.precomputed_embeddings),
                )
                for field, tensor in fields:
                    if _contains_tensor_container(tensor):
                        raise TypeError(
                            "CUDA VMM feature transport requires each feature "
                            "field to contain a single tensor"
                        )
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    wrapped = self.pool.wrap_tensor(tensor)
                    setattr(item, field, wrapped)
                    updates.append((item, field, tensor, wrapped))
        except BaseException as error:
            rollback_errors = []
            for item, field, tensor, wrapped in reversed(updates):
                try:
                    if isinstance(wrapped, CudaVmmTensorTransportProxy):
                        self.pool.cancel_proxy(wrapped)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
                finally:
                    setattr(item, field, tensor)
            if rollback_errors:
                error.add_note(
                    f"{len(rollback_errors)} VMM rollback operation(s) also failed"
                )
                raise error from rollback_errors[0]
            raise

    def cancel_for_dispatch(self, mm_items: list[MultimodalDataItem]) -> None:
        if self.pool is None or not mm_items:
            return

        errors = []
        for item in mm_items:
            fields = (
                ("feature", item.feature),
                ("precomputed_embeddings", item.precomputed_embeddings),
            )
            for field, proxy in fields:
                if not isinstance(proxy, CudaVmmTensorTransportProxy):
                    continue
                try:
                    self.pool.cancel_proxy(proxy)
                except BaseException as error:
                    errors.append(error)
                finally:
                    setattr(item, field, None)
        if errors:
            raise RuntimeError(
                f"Failed to cancel {len(errors)} VMM transport slice(s)"
            ) from errors[0]

    def shutdown(self) -> None:
        if self.pool is None:
            return
        self.pool.shutdown()
