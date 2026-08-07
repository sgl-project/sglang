import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MM_FEATURE_CACHE_SIZE = envs.SGLANG_MM_FEATURE_CACHE_MB.get() * 1024 * 1024

MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL = (
    envs.SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC.get()
)

_CONTROL_WORD_BYTES = 4
_DATA_ALIGNMENT = 256
_DEFAULT_MAX_INFLIGHT_SLICES = 4096

# Processors set this marker only when their encoder consumes each IPC feature
# on a single TP rank.  The scheduler then keeps the feature lazy until the
# model has computed the data-parallel assignment.
DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY = (
    "_sglang_defer_cuda_ipc_feature_reconstruction"
)


def get_mm_feature_pool_size_per_worker(
    total_pool_size: int, tokenizer_worker_num: int
) -> int:
    """Split the CUDA IPC feature-pool budget without exceeding it.

    Each tokenizer worker owns a distinct CUDA allocation, even though all pools
    are created on ``base_gpu_id``.  Therefore a minimum per-worker allocation
    would make the aggregate HBM reservation larger than the configured budget.
    Keep the configured value as a hard per-node cap and leave at most
    ``tokenizer_worker_num - 1`` bytes unused when it is not evenly divisible.
    """
    if total_pool_size <= 0:
        raise ValueError("total_pool_size must be positive")
    if tokenizer_worker_num <= 0:
        raise ValueError("tokenizer_worker_num must be positive")

    return total_pool_size // tokenizer_worker_num


# Cache for pool-level IPC handles on the consumer side.
# Key: the pool CUDA IPC handle tuple. Value: opened UntypedStorage.
_pool_storage_cache: dict = {}
_pool_cache_lock = threading.Lock()


def _normalize_pool_cache_key(pool_handle, device_index: int) -> tuple[Any, ...]:
    normalized_handle = (
        pool_handle if isinstance(pool_handle, tuple) else tuple(pool_handle)
    )
    return (device_index, normalized_handle)


def _open_pooled_storage_uncached(pool_handle):
    return torch.UntypedStorage._new_shared_cuda(*pool_handle)


def _pool_handle_cache_get_or_open(cache_key, pool_handle):
    storage = _pool_storage_cache.get(cache_key)
    if storage is None:
        with _pool_cache_lock:
            storage = _pool_storage_cache.get(cache_key)
            if storage is None:
                storage = _open_pooled_storage_uncached(pool_handle)
                _pool_storage_cache[cache_key] = storage
    return storage


def _pool_handle_cache_set(cache_key, storage):
    with _pool_cache_lock:
        _pool_storage_cache[cache_key] = storage


def _pool_handle_cache_invalidate(cache_key):
    with _pool_cache_lock:
        _pool_storage_cache.pop(cache_key, None)


def _pool_handle_cache_clear():
    with _pool_cache_lock:
        _pool_storage_cache.clear()


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _driver_modules():
    from cuda.bindings import driver as cuda

    from sglang.srt.distributed.device_communicators.vmm_utils import check_drv

    return cuda, check_drv


def _stream_wait_value32(device_id: int, address: int, value: int) -> None:
    cuda, check_drv = _driver_modules()
    stream = torch.cuda.current_stream(device_id)
    check_drv(
        cuda.cuStreamWaitValue32(stream.cuda_stream, address, value, 0),
        "cuStreamWaitValue32(mm CUDA IPC)",
    )


def _stream_write_value32(device_id: int, address: int, value: int) -> None:
    cuda, check_drv = _driver_modules()
    stream = torch.cuda.current_stream(device_id)
    check_drv(
        cuda.cuStreamWriteValue32(stream.cuda_stream, address, value, 0),
        "cuStreamWriteValue32(mm CUDA IPC)",
    )


def _resolve_consumer_rank(
    total_consumer_count: int, consumer_rank: Optional[int] = None
) -> int:
    if total_consumer_count == 1:
        return 0
    if consumer_rank is None:
        try:
            from sglang.srt.runtime_context import get_parallel

            rank = int(get_parallel().tp_rank)
        except Exception as exc:
            raise RuntimeError(
                "Cannot resolve the CUDA IPC consumer rank before parallel state "
                "initialization"
            ) from exc
    else:
        rank = int(consumer_rank)
    if not 0 <= rank < total_consumer_count:
        raise RuntimeError(
            f"CUDA IPC consumer rank {rank} is outside " f"[0, {total_consumer_count})"
        )
    return rank


@dataclass(frozen=True)
class MmItemMemoryChunk:
    start: int
    end: int
    slot: int
    generation: int

    @property
    def mem_size(self) -> int:
        return self.end - self.start


class MmItemMemoryPool:
    def __init__(
        self,
        memory_size: int,
        recycle_interval: float,
        base_gpu_id: int,
        consumer_count: int,
        max_inflight_slices: int = _DEFAULT_MAX_INFLIGHT_SLICES,
    ):
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if consumer_count <= 0:
            raise ValueError("consumer_count must be positive")

        self.device_id = base_gpu_id
        self.consumer_count = consumer_count
        self.control_words_per_slot = 1 + consumer_count
        self.max_inflight_slices = max_inflight_slices
        control_bytes = (
            max_inflight_slices * self.control_words_per_slot * _CONTROL_WORD_BYTES
        )
        self.data_start = _align_up(control_bytes, _DATA_ALIGNMENT)
        if memory_size <= self.data_start:
            raise ValueError(
                "CUDA IPC pool is too small after control metadata: "
                f"pool={memory_size}, control={self.data_start}"
            )

        self.memory_pool = torch.empty(
            memory_size, dtype=torch.uint8, device=f"cuda:{base_gpu_id}"
        ).contiguous()
        control_word_count = max_inflight_slices * self.control_words_per_slot
        self._control_words = (
            self.memory_pool[: control_word_count * _CONTROL_WORD_BYTES]
            .view(torch.int32)
            .view(max_inflight_slices, self.control_words_per_slot)
        )
        self._control_words.zero_()
        torch.cuda.synchronize(base_gpu_id)
        storage = self.memory_pool.untyped_storage()
        self._pool_ipc_handle = storage._share_cuda_()

        self._available_ranges = [(self.data_start, memory_size)]
        self._available_slots = list(reversed(range(max_inflight_slices)))
        self._slot_generations = [0] * max_inflight_slices
        self._occupied: dict[int, MmItemMemoryChunk] = {}

        self._lock = threading.Lock()
        self._pool_full_warned = False

        self._recycle_interval = recycle_interval
        self._stop_recycler = False
        self._recycle_thread = threading.Thread(
            target=self._recycle_loop, name="MmItemMemoryPoolRecycler", daemon=True
        )
        self._recycle_thread.start()

        logger.debug(
            f"[MmItemMemoryPool] init: memory_size={memory_size}, "
            f"recycle_interval={recycle_interval}s"
        )

    def shutdown(self):
        self._stop_recycler = True
        if self._recycle_thread.is_alive():
            self._recycle_thread.join(timeout=1.0)

    def _recycle_loop(self):
        torch.cuda.set_device(self.device_id)
        while not self._stop_recycler:
            try:
                with self._lock, torch.cuda.device(self.device_id):
                    self._recycle_ready_chunks_locked()
            except Exception as e:
                logger.warning(
                    f"[MmItemMemoryPool] recycle loop error: {e}", exc_info=True
                )

            time.sleep(self._recycle_interval)

    def _allocate_locked(self, nbytes: int) -> Optional[MmItemMemoryChunk]:
        candidates = [
            (end - start, index, start, end)
            for index, (start, end) in enumerate(self._available_ranges)
            if end - start >= nbytes
        ]
        if not candidates or not self._available_slots:
            return None
        _, index, start, end = min(candidates)
        self._available_ranges.pop(index)
        if start + nbytes < end:
            self._available_ranges.append((start + nbytes, end))
        slot = self._available_slots.pop()
        generation = self._slot_generations[slot] + 1
        if generation > 0x7FFFFFFF:
            raise RuntimeError("CUDA IPC pool slot generation exhausted")
        self._slot_generations[slot] = generation
        chunk = MmItemMemoryChunk(start, start + nbytes, slot, generation)
        self._occupied[slot] = chunk
        return chunk

    def _release_chunk_locked(self, chunk: MmItemMemoryChunk) -> None:
        self._occupied.pop(chunk.slot, None)
        self._available_slots.append(chunk.slot)
        self._available_ranges.append((chunk.start, chunk.end))

    def _merge_ranges_locked(self) -> None:
        merged = []
        for start, end in sorted(self._available_ranges):
            if merged and merged[-1][1] == start:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        self._available_ranges = merged

    def _recycle_ready_chunks_locked(self) -> None:
        if not self._occupied:
            return
        chunks = list(self._occupied.values())
        slot_indices = torch.tensor(
            [chunk.slot for chunk in chunks],
            dtype=torch.long,
            device=f"cuda:{self.device_id}",
        )
        expected = torch.tensor(
            [chunk.generation for chunk in chunks],
            dtype=torch.int32,
            device=f"cuda:{self.device_id}",
        ).unsqueeze(1)
        completed = (
            (self._control_words.index_select(0, slot_indices) == expected)
            .all(dim=1)
            .cpu()
            .tolist()
        )
        for chunk, is_complete in zip(chunks, completed):
            if is_complete:
                self._release_chunk_locked(chunk)
        self._merge_ranges_locked()

    def wrap_tensor(
        self, tensor: torch.Tensor, *, use_pool_handle_cache: bool
    ) -> Optional["CudaIpcTensorTransportProxy"]:
        if not tensor.is_cuda:
            raise ValueError("CUDA IPC transport requires a CUDA tensor")
        source = tensor.contiguous()
        nbytes = source.numel() * source.element_size()
        allocation_bytes = _align_up(nbytes, _DATA_ALIGNMENT)
        with self._lock:
            chunk = self._allocate_locked(allocation_bytes)
        if chunk is None:
            self._warn_pool_full_once(nbytes)
            return None

        try:
            with torch.cuda.device(self.device_id):
                destination = self.memory_pool[chunk.start : chunk.start + nbytes]
                destination.copy_(
                    source.view(torch.uint8).reshape(-1), non_blocking=True
                )
                ready_byte_offset = (
                    chunk.slot * self.control_words_per_slot * _CONTROL_WORD_BYTES
                )
                _stream_write_value32(
                    self.device_id,
                    self.memory_pool.data_ptr() + ready_byte_offset,
                    chunk.generation,
                )
        except Exception:
            with self._lock:
                self._release_chunk_locked(chunk)
                self._merge_ranges_locked()
            raise

        return CudaIpcTensorTransportProxy(
            data=destination,
            info_data=tensor,
            pool_ipc_handle=self._pool_ipc_handle,
            pool_byte_offset=chunk.start,
            ready_byte_offset=ready_byte_offset,
            ack_byte_offset=ready_byte_offset + _CONTROL_WORD_BYTES,
            generation=chunk.generation,
            total_consumer_count=self.consumer_count,
            use_pool_handle_cache=use_pool_handle_cache,
        )

    def _warn_pool_full_once(self, nbytes: int):
        if self._pool_full_warned:
            return
        self._pool_full_warned = True
        pool_mb = (
            self.memory_pool.numel() * self.memory_pool.element_size() / (1024 * 1024)
        )
        need_mb = nbytes / (1024 * 1024)
        logger.warning(
            "MmItemMemoryPool has no free chunk large enough for a %.2f MiB tensor "
            "(pool size: %.2f MiB); falling back to non-IPC transport. "
            "Consider increasing SGLANG_MM_FEATURE_CACHE_MB.",
            need_mb,
            pool_mb,
        )


class CudaIpcTensorTransportProxy:
    """Serializable view of one tensor stored in a CUDA IPC memory pool.

    The producer-ready word and one acknowledgement word per consumer live in
    the same CUDA allocation as the tensor. CUDA stream memory operations order
    the producer copy, consumer copy, and pool reuse without CPU shared memory
    or device-wide synchronization.
    """

    def __init__(
        self,
        data: torch.Tensor,
        info_data: torch.Tensor,
        pool_ipc_handle,
        pool_byte_offset: int,
        ready_byte_offset: int,
        ack_byte_offset: int,
        generation: int,
        total_consumer_count: int,
        use_pool_handle_cache: bool,
    ):
        if (not isinstance(data, torch.Tensor)) or (
            not isinstance(info_data, torch.Tensor)
        ):
            raise TypeError(
                f"Input 'data' must be a torch.Tensor, but got {type(data)}"
            )

        if total_consumer_count <= 0:
            raise ValueError("total_consumer_count must be positive")

        self.proxy_state = {
            "ipc_extra": {
                "pool_handle": pool_ipc_handle,
                "pool_byte_offset": pool_byte_offset,
                "shape": data.shape,
                "dtype": data.dtype,
                "stride": data.stride(),
                "storage_offset": 0,
                "nbytes": data.numel() * data.element_size(),
                "recons_shape": info_data.shape,
                "recons_dtype": info_data.dtype,
                "ready_byte_offset": ready_byte_offset,
                "ack_byte_offset": ack_byte_offset,
                "generation": generation,
                "total_consumer_count": total_consumer_count,
                "use_pool_handle_cache": use_pool_handle_cache,
            },
            "tensor_data": None,
        }
        self.reconstruct_tensor = None
        self._consumer_acknowledged = False
        # Keep uncached mappings alive until the work enqueued on the consumer
        # stream has completed.
        self._pool_storage = None

    def _reconstruct_from_ipc_extra(
        self, ipc_extra, *, use_cache: bool, rebuild_device_idx: int
    ):
        shape = ipc_extra["shape"]
        dtype = ipc_extra["dtype"]
        stride = ipc_extra["stride"]
        # Redirect handle[0] to the consumer's device so _new_shared_cuda's
        # CUDAGuard stays there; peer access handles the cross-GPU open.
        pool_handle = ipc_extra["pool_handle"]
        redirected_handle = (rebuild_device_idx,) + tuple(pool_handle)[1:]
        target_device = torch.device(f"cuda:{rebuild_device_idx}")
        cache_key = _normalize_pool_cache_key(pool_handle, rebuild_device_idx)

        with torch.cuda.device(target_device):
            if use_cache:
                storage = _pool_handle_cache_get_or_open(cache_key, redirected_handle)
            else:
                storage = _open_pooled_storage_uncached(redirected_handle)
            slice_storage = storage[
                ipc_extra["pool_byte_offset"] : ipc_extra["pool_byte_offset"]
                + ipc_extra["nbytes"]
            ]
            slice_tensor = torch.empty(0, dtype=dtype, device=target_device).set_(
                slice_storage,
                storage_offset=ipc_extra["storage_offset"],
                size=shape,
                stride=stride,
            )

        return slice_tensor, storage

    def _open_pool_slice(self, rebuild_device_idx: int):
        ipc_extra = self.proxy_state["ipc_extra"]
        use_cache = ipc_extra["use_pool_handle_cache"]
        try:
            return self._reconstruct_from_ipc_extra(
                ipc_extra,
                use_cache=use_cache,
                rebuild_device_idx=rebuild_device_idx,
            )
        except Exception as exc:
            if not use_cache:
                raise
            cache_key = _normalize_pool_cache_key(
                ipc_extra["pool_handle"], rebuild_device_idx
            )
            logger.info(
                "Failed to deserialize from cached pooled CUDA IPC handle (%s). "
                "Invalidating cache entry and retrying uncached.",
                exc,
            )
            _pool_handle_cache_invalidate(cache_key)
            result = self._reconstruct_from_ipc_extra(
                ipc_extra,
                use_cache=False,
                rebuild_device_idx=rebuild_device_idx,
            )
            _pool_handle_cache_set(cache_key, result[1])
            return result

    @staticmethod
    def _control_address(storage, byte_offset: int) -> int:
        return storage.data_ptr() + byte_offset

    def _wait_until_ready(self, storage, device_id: int) -> None:
        ipc_extra = self.proxy_state["ipc_extra"]
        _stream_wait_value32(
            device_id,
            self._control_address(storage, ipc_extra["ready_byte_offset"]),
            ipc_extra["generation"],
        )

    def _acknowledge_on_stream(
        self,
        storage,
        device_id: int,
        consumer_count: int,
        consumer_rank: Optional[int],
    ) -> None:
        if self._consumer_acknowledged:
            return
        ipc_extra = self.proxy_state["ipc_extra"]
        total_consumer_count = ipc_extra["total_consumer_count"]
        if consumer_count == total_consumer_count:
            consumer_ranks = range(total_consumer_count)
        elif consumer_count == 1:
            consumer_ranks = (
                _resolve_consumer_rank(total_consumer_count, consumer_rank),
            )
        else:
            raise ValueError(
                "consumer_count must be either 1 or total_consumer_count "
                f"({total_consumer_count}), got {consumer_count}"
            )

        ack_base = ipc_extra["ack_byte_offset"]
        for rank in consumer_ranks:
            _stream_write_value32(
                device_id,
                self._control_address(storage, ack_base + rank * _CONTROL_WORD_BYTES),
                ipc_extra["generation"],
            )
        self._consumer_acknowledged = True

    def acknowledge_consumption(
        self, consumer_count: int = 1, consumer_rank: Optional[int] = None
    ) -> None:
        """Stream-order pool release when a cache hit needs no tensor copy."""
        if self._consumer_acknowledged:
            return
        device_id = torch.cuda.current_device()
        with torch.cuda.device(device_id):
            _, storage = self._open_pool_slice(device_id)
            self._wait_until_ready(storage, device_id)
            self._acknowledge_on_stream(
                storage, device_id, consumer_count, consumer_rank
            )
        self._pool_storage = storage

    def reconstruct_on_target_device(
        self,
        rebuild_device_idx,
        consumer_count: int = 1,
        consumer_rank: Optional[int] = None,
    ):
        rebuild_device = torch.device(f"cuda:{rebuild_device_idx}")
        if (
            isinstance(self.reconstruct_tensor, torch.Tensor)
            and self.reconstruct_tensor.device == rebuild_device
        ):
            return self.reconstruct_tensor

        ipc_extra = self.proxy_state["ipc_extra"]
        with torch.cuda.device(rebuild_device):
            slice_tensor, storage = self._open_pool_slice(rebuild_device_idx)
            self._wait_until_ready(storage, rebuild_device_idx)
            reconstructed_tensor = torch.empty(
                ipc_extra["recons_shape"],
                dtype=ipc_extra["recons_dtype"],
                device=rebuild_device,
            ).contiguous()
            reconstructed_tensor.view(torch.uint8).reshape(-1).copy_(slice_tensor)
            self._acknowledge_on_stream(
                storage,
                rebuild_device_idx,
                consumer_count,
                consumer_rank,
            )

        self._pool_storage = storage
        self.reconstruct_tensor = reconstructed_tensor
        return self.reconstruct_tensor
