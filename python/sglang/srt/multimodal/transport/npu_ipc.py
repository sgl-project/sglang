"""NPU IPC transport for multimodal feature tensors on Ascend devices.

Ascend exposes no stream-ordered control-word writes (the CUDA pool's
``cuStreamWaitValue32`` ready/ack words), so this transport keeps its
liveness state in POSIX shared memory instead:

* one acknowledgement slot per consumer rank, so concurrent acknowledgements
  from different scheduler processes stay idempotent and lossless; each slot
  holds the lease id it acknowledges, so an acknowledgement for an already
  recycled lease can never complete a board handed to a newer lease;
* the pool owner drains the streams carrying its pool copy before publishing a
  proxy, because a consumer process cannot order its own stream against the
  producer's copy.

Mapping the pool handle on the consumer side is cached process-wide, which is
what ``SGLANG_USE_IPC_POOL_HANDLE_CACHE`` controls here -- the pool handle and
its lease are always used, the flag only decides whether the consumer reopens
the mapping per reconstruction.
"""

import logging
import os
import threading
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.multiprocessing.reductions import reduce_tensor

from sglang.srt.multimodal.transport.cuda_ipc import _normalize_pool_cache_key
from sglang.srt.multimodal.transport.memory_pool import resolve_consumer_rank
from sglang.srt.utils.stale_shm_cleanup import make_shm_name

logger = logging.getLogger(__name__)

# Consumer-side pool mappings, keyed by (device_index, pool handle tuple).
_pool_tensor_cache: dict = {}
_pool_cache_lock = threading.Lock()

# Acknowledgement slots are int64 and hold the lease id they acknowledge, so
# the monotonic lease counter stays inside that range.
_MAX_LEASE_ID = 0x7FFFFFFFFFFFFFFF


def _drain_current_streams(*devices: Union[int, torch.device]) -> None:
    """Synchronize only the current streams of the given NPU devices."""
    drained = set()
    for device in devices:
        if isinstance(device, torch.device):
            if device.type != "npu":
                raise ValueError(f"expected an NPU device, got {device}")
            index = (
                device.index if device.index is not None else torch.npu.current_device()
            )
        else:
            index = int(device)
        if index in drained:
            continue
        drained.add(index)
        torch.npu.current_stream(index).synchronize()


def _open_pooled_tensor_uncached(pool_handle) -> torch.Tensor:
    """Materialize the whole pool the handle points at, on its home device."""
    func, args = pool_handle
    return func(*args)


def _pooled_tensor_cache_get_or_open(pool_handle, rebuild_device_idx: int):
    cache_key = _normalize_pool_cache_key(pool_handle, rebuild_device_idx)
    pool_tensor = _pool_tensor_cache.get(cache_key)
    if pool_tensor is not None:
        return pool_tensor, cache_key
    with _pool_cache_lock:
        pool_tensor = _pool_tensor_cache.get(cache_key)
        if pool_tensor is None:
            pool_tensor = _open_pooled_tensor_uncached(pool_handle)
            _pool_tensor_cache[cache_key] = pool_tensor
    return pool_tensor, cache_key


def _pooled_tensor_cache_set(cache_key, pool_tensor) -> None:
    with _pool_cache_lock:
        _pool_tensor_cache[cache_key] = pool_tensor


def _pooled_tensor_cache_invalidate(cache_key) -> None:
    with _pool_cache_lock:
        _pool_tensor_cache.pop(cache_key, None)


class ShmSyncBuffer:
    """Shared-memory acknowledgement slots, one per consumer rank.

    A slot holds the lease id it acknowledges (0 while pending), so a write
    from a stale proxy can never complete the board of a newer lease.
    """

    def __init__(self, consumer_count: int):
        ack_word_dtype = np.dtype(np.int64)
        self.consumer_count = max(int(consumer_count), 1)
        self.buffer = shared_memory.SharedMemory(
            create=True,
            size=self.consumer_count * ack_word_dtype.itemsize,
            name=make_shm_name("sync"),
        )
        self.buffer_wrapper = np.ndarray(
            self.consumer_count, dtype=ack_word_dtype, buffer=self.buffer.buf
        )
        self.buffer_wrapper *= 0
        self.meta_data = {
            "handle": self.buffer.name,
            "shape": self.buffer_wrapper.shape,
            "dtype": self.buffer_wrapper.dtype,
        }

    def reset(self):
        self.buffer_wrapper *= 0

    def all_slots_equal(self, value: int) -> bool:
        """Whether every consumer rank acknowledged the lease ``value``."""
        return bool((self.buffer_wrapper == value).all())

    def close(self):
        if self.buffer is not None:
            self.buffer.close()
            self.buffer.unlink()
            self.buffer = None
            # Drop the view with the mapping: a read after close would fault.
            self.buffer_wrapper = None


class MmItemMemoryChunk:
    def __init__(
        self, area: Tuple, sync_buffer: ShmSyncBuffer, lease_id: Optional[int] = None
    ):
        self.area = area
        self.sync_buffer = sync_buffer
        self.lease_id = lease_id

    @property
    def mem_size(self):
        return self.area[1] - self.area[0]

    @property
    def start(self):
        return self.area[0]

    @property
    def end(self):
        return self.area[1]

    def try_to_recycle(self) -> bool:
        """Whether every consumer acknowledged this chunk's current lease.

        A board outlives its lease, so the slots are matched against the
        lease id, not against "any slot is set". Free chunks never recycle.
        """
        if self.lease_id is None:
            return False
        if self.sync_buffer.all_slots_equal(self.lease_id):
            self.sync_buffer.reset()
            return True
        return False


class MmItemMemoryPool:
    def __init__(self, memory_size, recycle_interval, base_npu_id, tp_size):
        self.tp_size = max(int(tp_size), 1)
        memory_size_mb = memory_size / (1024 * 1024)
        self.memory_pool = torch.empty(
            memory_size, dtype=torch.int8, device=f"npu:{base_npu_id}"
        ).contiguous()
        self._pool_ipc_handle = reduce_tensor(self.memory_pool)
        self._pool_device_index = self.memory_pool.device.index

        self.sync_buffer_list = []

        init_chunk = MmItemMemoryChunk((0, memory_size), self.pop_sync_buffer())
        self.available_chunks = [init_chunk]
        # Occupied ranges are disjoint, so chunk.start is a unique key.
        self.occupied_chunks = {}
        # Monotonic per-occupancy ids: a late cancel for an offset that was
        # freed and re-leased must not release the new occupant.
        self._next_lease_id = 0

        self._lock = threading.Lock()
        self._pool_full_warned = False
        self._lease_failed_warned = False

        self._recycle_interval = recycle_interval
        self._stop_recycler = False
        self._recycle_thread = threading.Thread(
            target=self._recycle_loop, name="MmItemMemoryPoolRecycler", daemon=True
        )
        self._recycle_thread.start()

        logger.info(
            "[NPU IPC Pool] Initialized: size=%.1f MB, device=npu:%d, tp_size=%d",
            memory_size_mb,
            base_npu_id,
            self.tp_size,
        )

    def shutdown(self):
        self._stop_recycler = True
        if self._recycle_thread.is_alive():
            self._recycle_thread.join(timeout=1.0)

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return len(self.occupied_chunks)

    def _recycle_loop(self):
        while not self._stop_recycler:
            try:
                with self._lock:
                    self.recycle_chunks()
                    self.merge_chunks()
            except Exception as e:
                logger.warning(
                    "[MmItemMemoryPool(NPU)] recycle loop error: %s", e, exc_info=True
                )

            time.sleep(self._recycle_interval)

    def pop_sync_buffer(self):
        """Take an ack board, recycling from the free list when possible.

        Boards are created here and never freed while the pool runs; they
        only return to the free list. The shm footprint therefore rises to
        the historical peak of simultaneously live chunks and then stays
        flat -- bounded, not a leak (tp_size int64 words per board,
        page-granular).
        """
        if len(self.sync_buffer_list) == 0:
            try:
                new_sync_buffer = ShmSyncBuffer(self.tp_size)
                return new_sync_buffer
            except Exception as e:
                logger.warning("allocate shm buffer failed: %s", e)
                raise RuntimeError("failed to allocate NPU IPC sync shm buffer") from e
        else:
            shm_buffer = self.sync_buffer_list.pop()
            # A recycled buffer must start from zero or the chunk it guards
            # could be recycled immediately after reuse.
            shm_buffer.reset()
            return shm_buffer

    def push_sync_buffer(self, sync_buffer):
        self.sync_buffer_list.append(sync_buffer)

    def get_available_chunk(self, nbytes: int) -> Optional[MmItemMemoryChunk]:
        """Best-fit allocation. The sole caller holds ``self._lock``."""
        min_size = float("inf")
        selected_chunk = None
        for chunk in self.available_chunks:
            if chunk.mem_size >= nbytes:
                if chunk.mem_size < min_size:
                    min_size = chunk.mem_size
                    selected_chunk = chunk

        if not selected_chunk:
            logger.debug(
                "[NPU IPC Pool] No available chunk for tensor size=%.2f MB, "
                "pool is FULL. available_chunks=%d, occupied_chunks=%d",
                nbytes / (1024 * 1024),
                len(self.available_chunks),
                len(self.occupied_chunks),
            )
            return None

        if self._next_lease_id >= _MAX_LEASE_ID:
            # Checked before the board is popped so a failure strands none.
            raise RuntimeError("NPU IPC pool lease id exhausted")

        occupied_chunk_area = (selected_chunk.start, selected_chunk.start + nbytes)
        previous_sync_buffer = selected_chunk.sync_buffer
        new_sync_buffer = self.pop_sync_buffer()
        self._next_lease_id += 1
        new_occupied_chunk = MmItemMemoryChunk(
            occupied_chunk_area, new_sync_buffer, lease_id=self._next_lease_id
        )

        self.available_chunks.remove(selected_chunk)
        self.occupied_chunks[new_occupied_chunk.start] = new_occupied_chunk

        if new_occupied_chunk.end == selected_chunk.end:
            # The allocation consumed the chunk exactly; its buffer is free.
            self.push_sync_buffer(previous_sync_buffer)
        else:
            split_available_chunk = MmItemMemoryChunk(
                (new_occupied_chunk.end, selected_chunk.end),
                previous_sync_buffer,
            )
            self.available_chunks.append(split_available_chunk)

        logger.debug(
            "[NPU IPC Pool] Chunk allocated: size=%.2f MB, offset=%d, "
            "available=%d, occupied=%d",
            nbytes / (1024 * 1024),
            new_occupied_chunk.start,
            len(self.available_chunks),
            len(self.occupied_chunks),
        )
        return new_occupied_chunk

    def _lease_chunk_for_bytes(self, nbytes: int) -> Optional[MmItemMemoryChunk]:
        """Lease a chunk; never raises -- an unservable pool degrades to
        CPU transport like a full one instead of failing the request."""
        with self._lock:
            try:
                chunk = self.get_available_chunk(nbytes)
            except Exception as e:
                if not self._lease_failed_warned:
                    logger.warning(
                        "[NPU IPC Pool] could not lease a %.2f MiB chunk (%s); "
                        "falling back to CPU transport.",
                        nbytes / (1024 * 1024),
                        e,
                    )
                    self._lease_failed_warned = True
                return None
            if chunk is not None:
                return chunk
            if not self._pool_full_warned:
                logger.warning(
                    "[NPU IPC Pool] has no free chunk large enough for a "
                    "%.2f MiB tensor; falling back to CPU transport. "
                    "Consider increasing SGLANG_MM_FEATURE_CACHE_MB.",
                    nbytes / (1024 * 1024),
                )
                self._pool_full_warned = True
            return None

    def wrap_tensor(self, tensor: torch.Tensor, *, use_pool_handle_cache: bool):
        source = tensor.contiguous()
        nbytes = source.numel() * source.element_size()
        if nbytes == 0:
            # Nothing to transport over IPC; degrade to the CPU path instead
            # of failing the request.
            logger.warning(
                "[NPU IPC Pool] refusing to transport an empty tensor; "
                "falling back to CPU transport."
            )
            return None

        chunk = self._lease_chunk_for_bytes(nbytes)
        if chunk is None:
            return None
        try:
            available_slice = self.memory_pool[chunk.start : chunk.end]
            available_slice.copy_(
                source.view(torch.int8).reshape(-1), non_blocking=True
            )
            # Consumers read this slice from another process, and Ascend has no
            # stream-ordered control word to order that read against this copy,
            # so the producer drains the copy's streams before publishing.
            _drain_current_streams(self._pool_device_index, torch.npu.current_device())
            return NpuIpcTensorTransportProxy(
                data=available_slice,
                info_data=tensor,
                sync_buffer=chunk.sync_buffer,
                pool_ipc_handle=self._pool_ipc_handle,
                pool_byte_offset=chunk.start,
                pool_lease_id=chunk.lease_id,
                pool_device_index=self._pool_device_index,
                tp_size=self.tp_size,
                use_pool_handle_cache=use_pool_handle_cache,
            )
        except Exception as e:
            # A failed wrap must degrade to CPU transport, not fail the
            # request; give the leased chunk back first.
            logger.warning(
                "[NPU IPC Pool] wrapping a %.2f MiB tensor failed (%s); "
                "falling back to CPU transport.",
                nbytes / (1024 * 1024),
                e,
            )
            with self._lock:
                self._reclaim_lease_locked(chunk)
            return None

    def _return_lease_locked(self, byte_offset: int, lease_id: Optional[int]) -> bool:
        """Free a lease known only by its serialized coordinates.

        The caller holds ``self._lock``. The lease id check rejects stale
        proxies whose offset was freed and re-leased since they were built.
        """
        chunk = self.occupied_chunks.get(byte_offset)
        if chunk is None or chunk.lease_id != lease_id:
            return False
        self._reclaim_lease_locked(chunk)
        return True

    def _reclaim_lease_locked(self, chunk: MmItemMemoryChunk) -> None:
        """Return one leased chunk to the free list. The caller holds ``self._lock``."""
        chunk.sync_buffer.reset()
        self.reclaim_chunk(chunk)
        self.merge_chunks()

    def cancel_proxy(self, proxy: "NpuIpcTensorTransportProxy") -> None:
        """Return a published slice when its request was never dispatched."""
        ipc_extra = proxy.proxy_state["ipc_extra"]
        pool_handle = ipc_extra["pool_handle"]
        if tuple(pool_handle) != tuple(self._pool_ipc_handle):
            raise RuntimeError("NPU IPC proxy does not belong to this pool")

        byte_offset = ipc_extra["pool_byte_offset"]
        lease_id = ipc_extra.get("lease_id")
        with self._lock:
            if self._return_lease_locked(byte_offset, lease_id):
                return
        logger.warning(
            "[NPU IPC Pool] cancel_proxy found no live lease for offset=%d lease_id=%s",
            byte_offset,
            lease_id,
        )

    def reclaim_chunk(self, chunk: MmItemMemoryChunk):
        self.occupied_chunks.pop(chunk.start)
        self.available_chunks.append(chunk)

    def recycle_chunks(self):
        to_recycle_chunks = []
        for chunk in self.occupied_chunks.values():
            if chunk.try_to_recycle():
                to_recycle_chunks.append(chunk)

        for chunk in to_recycle_chunks:
            self.reclaim_chunk(chunk)

    def merge_chunks(self):
        if len(self.available_chunks) <= 1:
            return

        self.available_chunks.sort(key=lambda chunk: chunk.start)
        merged_chunks = []
        to_merge_chunk = None
        for chunk in self.available_chunks:
            if to_merge_chunk is None:
                to_merge_chunk = chunk
            elif to_merge_chunk.end == chunk.start:
                # The absorbed chunk's ack board returns to the free list.
                self.push_sync_buffer(chunk.sync_buffer)
                to_merge_chunk = MmItemMemoryChunk(
                    (to_merge_chunk.start, chunk.end),
                    to_merge_chunk.sync_buffer,
                )
            else:
                merged_chunks.append(to_merge_chunk)
                to_merge_chunk = chunk
        if to_merge_chunk is not None:
            merged_chunks.append(to_merge_chunk)

        self.available_chunks = merged_chunks


class NpuIpcTensorTransportProxy:
    """Serializable handle of a tensor stored in a bounded NPU IPC pool."""

    def __init__(
        self,
        data: torch.Tensor,
        info_data: torch.Tensor,
        sync_buffer,
        pool_ipc_handle,
        pool_lease_id: int,
        pool_byte_offset: int = 0,
        pool_device_index: int = 0,
        tp_size: int = 1,
        use_pool_handle_cache: bool = True,
    ):
        if (not isinstance(data, torch.Tensor)) or (
            not isinstance(info_data, torch.Tensor)
        ):
            raise TypeError(
                f"'data' and 'info_data' must be torch.Tensor, "
                f"but got {type(data)} and {type(info_data)}"
            )

        sync_buffer_meta = sync_buffer.meta_data if sync_buffer else None
        self.proxy_state = {
            "ipc_extra": {
                "pool_handle": pool_ipc_handle,
                "pool_byte_offset": pool_byte_offset,
                "lease_id": pool_lease_id,
                "pool_device_index": pool_device_index,
                "shape": data.shape,
                "dtype": data.dtype,
                "stride": data.stride(),
                "recons_shape": info_data.shape,
                "recons_dtype": info_data.dtype,
                "sync_buffer_meta": sync_buffer_meta,
                "use_pool_handle_cache": use_pool_handle_cache,
            }
        }
        self.reconstruct_tensor = None
        self._consumer_acknowledged = False
        # Consumers acknowledge in scheduler processes; the pool-owner process
        # may still reconstruct for hashing, but that read must not consume
        # acknowledgement quota or the chunk could be recycled too early.
        self._origin_pid = os.getpid()
        self.total_consumer_count = max(int(tp_size), 1)
        # Keeps an uncached pool mapping alive until this proxy is dropped.
        self._pool_tensor = None

    def __getstate__(self):
        # Process-local pool mappings must never travel: the receiver reopens
        # them from proxy_state, and pickling the pool tensor would copy it.
        state = self.__dict__.copy()
        state["_sync_buffer"] = None
        state["_sync_buffer_data"] = None
        state["_pool_tensor"] = None
        # The owner's pre-dispatch hashing read is not a consumer copy:
        # shipped, it would make a same-device receiver take the reconstruct
        # early return and never ack, stranding the lease.
        state["reconstruct_tensor"] = None
        state["_consumer_acknowledged"] = False
        return state

    @property
    def sync_data_meta(self):
        ipc_extra = self.proxy_state.get("ipc_extra")
        if isinstance(ipc_extra, dict):
            ipc_extra_sync = ipc_extra.get("sync_buffer_meta", {})
            if ipc_extra_sync:
                return ipc_extra_sync
        return {}

    @property
    def sync_buffer(self):
        if getattr(self, "_sync_buffer", None) is None:
            shm_name = self.sync_data_meta.get("handle")
            if shm_name:
                self._sync_buffer = shared_memory.SharedMemory(name=shm_name)
            else:
                # No sync metadata attached: nothing to mmap.
                self._sync_buffer = None
        return self._sync_buffer

    @property
    def sync_buffer_data(self):
        if getattr(self, "_sync_buffer_data", None) is None:
            shm = self.sync_buffer
            if shm is None:
                self._sync_buffer_data = None
            else:
                shape = self.sync_data_meta["shape"]
                dtype = self.sync_data_meta["dtype"]
                self._sync_buffer_data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return self._sync_buffer_data

    def close_shm(self):
        if getattr(self, "_sync_buffer", None) is not None:
            self._sync_buffer.close()
            self._sync_buffer = None
            # Drop the cached view too: writing through it after close would
            # hit unmapped memory.
            self._sync_buffer_data = None

    def _slice_of_pool_tensor(self, pool_tensor, ipc_extra):
        # Hold the mapping so it outlives this reconstruction.
        self._pool_tensor = pool_tensor
        return pool_tensor.as_strided(
            size=ipc_extra["shape"],
            stride=ipc_extra["stride"],
            storage_offset=ipc_extra["pool_byte_offset"],
        )

    def _pool_slice_tensor(self, ipc_extra, rebuild_device_idx: int):
        """Open the pool on this device, reusing a cached mapping when enabled."""
        pool_handle = ipc_extra["pool_handle"]
        cache_key = None
        try:
            if ipc_extra.get("use_pool_handle_cache", False):
                pool_tensor, cache_key = _pooled_tensor_cache_get_or_open(
                    pool_handle, rebuild_device_idx
                )
            else:
                pool_tensor = _open_pooled_tensor_uncached(pool_handle)
            # The slice lives at pool_byte_offset inside the whole-pool storage.
            return self._slice_of_pool_tensor(pool_tensor, ipc_extra)
        except Exception as exc:
            if cache_key is None:
                raise
            logger.info(
                "[NPU IPC] Cached pool mapping on npu:%d is unusable (%s); "
                "reopening the pool handle.",
                rebuild_device_idx,
                exc,
            )
            _pooled_tensor_cache_invalidate(cache_key)

        pool_tensor = _open_pooled_tensor_uncached(pool_handle)
        _pooled_tensor_cache_set(cache_key, pool_tensor)
        return self._slice_of_pool_tensor(pool_tensor, ipc_extra)

    def acknowledge_consumption(self, consumer_count: int = 1):
        self._acknowledge_consumption(consumer_count)

    def release_without_reconstruction(self, consumer_count: int = 1) -> None:
        """Acknowledge abandoned consumers so the pool slice can be recycled.

        Forced: an explicit abandon of a not-yet-dispatched proxy means no other
        consumer will ever ack, so it counts even from the pool-owner process.
        """
        self._acknowledge_consumption(consumer_count, force=True)
        # Keep the shm mapping open for other consumers unless fully acked.
        if self._consumer_acknowledged:
            self.close_shm()

    def _acknowledge_consumption(self, consumer_count: int = 1, force: bool = False):
        if self._consumer_acknowledged:
            return
        in_origin_process = self._origin_pid == os.getpid()
        if in_origin_process and not force:
            # Origin-process reads (e.g. hash_feature reconstruction) are not
            # consumer acknowledgements.
            return

        sync_data = self.sync_buffer_data
        if sync_data is None:
            # A proxy without sync metadata has no acknowledgement board.
            return

        ack_value = (self.proxy_state.get("ipc_extra") or {}).get("lease_id")
        if ack_value is None:
            raise ValueError(
                "NPU IPC acknowledgement needs the pool lease id, and this "
                "proxy carries none"
            )

        slot_count = len(sync_data)
        if in_origin_process:
            # Nothing was dispatched, so no consumer will ever ack this slice.
            consumer_ranks = range(slot_count)
        elif consumer_count >= slot_count:
            # One consumer covers the whole group (encoder-DP assignment).
            consumer_ranks = range(slot_count)
        elif consumer_count == 1:
            consumer_ranks = (resolve_consumer_rank(slot_count, None, "NPU IPC"),)
        else:
            raise ValueError(
                "NPU IPC acknowledgements support one consumer or the complete "
                f"consumer group, got {consumer_count}/{slot_count}"
            )

        try:
            sync_data[list(consumer_ranks)] = ack_value
            self._consumer_acknowledged = True
        except Exception as e:
            logger.warning("[NPU IPC] acknowledge_consumption error: %s", e)

    def _copy_slice_tensor_to_target(
        self, slice_tensor, target_device, recons_shape, recons_dtype, pool_device_index
    ):
        reconstructed_tensor = torch.empty(
            recons_shape, dtype=recons_dtype, device=target_device
        ).contiguous()
        slice_int8 = slice_tensor.view(torch.int8).reshape(-1)
        reconstructed_tensor.view(torch.int8).reshape(-1).copy_(
            slice_int8, non_blocking=True
        )
        # The pool is reused by another process with no shared stream, so the
        # async read must complete before the chunk is acknowledged. The copy
        # can be enqueued on either device's queue, so drain both.
        _drain_current_streams(target_device, pool_device_index)
        return reconstructed_tensor

    def reconstruct_on_target_device(self, rebuild_device_idx, consumer_count: int = 1):
        rebuild_device = torch.device(f"npu:{rebuild_device_idx}")
        if (
            isinstance(self.reconstruct_tensor, torch.Tensor)
            and self.reconstruct_tensor.device == rebuild_device
        ):
            return self.reconstruct_tensor

        ipc_extra = self.proxy_state["ipc_extra"]
        slice_tensor = self._pool_slice_tensor(
            ipc_extra, rebuild_device_idx=rebuild_device_idx
        )
        reconstructed_tensor = self._copy_slice_tensor_to_target(
            slice_tensor,
            rebuild_device,
            ipc_extra["recons_shape"],
            ipc_extra["recons_dtype"],
            ipc_extra["pool_device_index"],
        )

        self.reconstruct_tensor = reconstructed_tensor
        self._acknowledge_consumption(consumer_count)
        return self.reconstruct_tensor
