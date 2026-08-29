import logging
import threading
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.multimodal.transport.memory_pool import (
    DEFAULT_MAX_INFLIGHT_SLICES,
    StreamOrderedMmFeaturePool,
    StreamOrderedPoolConsumerMixin,
)

logger = logging.getLogger(__name__)

MM_FEATURE_CACHE_SIZE = envs.SGLANG_MM_FEATURE_CACHE_MB.get() * 1024 * 1024

MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL = (
    envs.SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC.get()
)

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


class MmItemMemoryPool:
    def __init__(
        self,
        memory_size: int,
        recycle_interval: float,
        base_gpu_id: int,
        consumer_count: int,
        max_inflight_slices: int = DEFAULT_MAX_INFLIGHT_SLICES,
    ):
        self.device_id = base_gpu_id
        self.consumer_count = consumer_count
        self.memory_pool = torch.empty(
            memory_size, dtype=torch.uint8, device=f"cuda:{base_gpu_id}"
        ).contiguous()
        self._pool = StreamOrderedMmFeaturePool(
            memory_size=memory_size,
            byte_tensor=self.memory_pool,
            base_address=self.memory_pool.data_ptr(),
            device_id=base_gpu_id,
            consumer_count=consumer_count,
            recycle_interval=recycle_interval,
            transport_name="CUDA IPC",
            max_inflight_slices=max_inflight_slices,
        )
        storage = self.memory_pool.untyped_storage()
        self._pool_ipc_handle = storage._share_cuda_()
        self._pool_full_warned = False

        logger.debug(
            f"[MmItemMemoryPool] init: memory_size={memory_size}, "
            f"recycle_interval={recycle_interval}s"
        )

    def shutdown(self):
        self._pool.shutdown()

    @property
    def active_lease_count(self) -> int:
        return self._pool.active_lease_count

    def wrap_tensor(
        self, tensor: torch.Tensor, *, use_pool_handle_cache: bool
    ) -> Optional["CudaIpcTensorTransportProxy"]:
        lease, destination = self._pool.copy_tensor(tensor)
        if lease is None:
            nbytes = tensor.numel() * tensor.element_size()
            self._warn_pool_full_once(nbytes)
            return None

        return CudaIpcTensorTransportProxy(
            data=destination,
            info_data=tensor,
            pool_ipc_handle=self._pool_ipc_handle,
            pool_byte_offset=lease.start,
            ready_byte_offset=lease.ready_byte_offset,
            ack_byte_offset=lease.ack_byte_offset,
            generation=lease.generation,
            total_consumer_count=self.consumer_count,
            use_pool_handle_cache=use_pool_handle_cache,
        )

    def cancel_proxy(self, proxy: "CudaIpcTensorTransportProxy") -> None:
        """Return a published slice when its request was never dispatched."""
        ipc_extra = proxy.proxy_state["ipc_extra"]
        if tuple(ipc_extra["pool_handle"]) != tuple(self._pool_ipc_handle):
            raise RuntimeError("CUDA IPC proxy does not belong to this pool")
        self._pool.cancel_lease(
            ready_byte_offset=proxy.ready_byte_offset,
            ack_byte_offset=proxy.ack_byte_offset,
            generation=proxy.generation,
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


class CudaIpcTensorTransportProxy(StreamOrderedPoolConsumerMixin):
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

        self._init_stream_ordered_consumer(
            ready_byte_offset=ready_byte_offset,
            ack_byte_offset=ack_byte_offset,
            generation=generation,
            total_consumer_count=total_consumer_count,
            transport_name="CUDA IPC",
        )

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
                "use_pool_handle_cache": use_pool_handle_cache,
            },
            "tensor_data": None,
        }
        self.reconstruct_tensor = None
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

    def _retain_storage_until_stream_completes(self, storage, device_id: int) -> None:
        if self.proxy_state["ipc_extra"]["use_pool_handle_cache"]:
            # The process-wide cache owns the mapping after this proxy is
            # replaced by its reconstructed tensor.
            self._pool_storage = storage
        else:
            # An uncached mapping is owned only by this proxy. The caller
            # replaces the proxy immediately, so finish the current stream
            # before allowing the mapping to close.
            torch.cuda.current_stream(device_id).synchronize()

    def acknowledge_consumption(
        self, consumer_count: int = 1, consumer_rank: Optional[int] = None
    ) -> None:
        """Stream-order pool release when a cache hit needs no tensor copy."""
        if self._consumer_acknowledged:
            return
        device_id = torch.cuda.current_device()
        with torch.cuda.device(device_id):
            _, storage = self._open_pool_slice(device_id)
            base_address = storage.data_ptr()
            self._wait_until_ready(base_address, device_id)
            self._acknowledge_on_stream(
                base_address, device_id, consumer_count, consumer_rank
            )
        self._retain_storage_until_stream_completes(storage, device_id)

    def release_without_reconstruction(self, consumer_count: int = 1) -> None:
        """Release a pool slice when its request abandons this proxy."""
        self.acknowledge_consumption(consumer_count)

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
            base_address = storage.data_ptr()
            self._wait_until_ready(base_address, rebuild_device_idx)
            reconstructed_tensor = torch.empty(
                ipc_extra["recons_shape"],
                dtype=ipc_extra["recons_dtype"],
                device=rebuild_device,
            ).contiguous()
            reconstructed_tensor.view(torch.uint8).reshape(-1).copy_(slice_tensor)
            self._acknowledge_on_stream(
                base_address,
                rebuild_device_idx,
                consumer_count,
                consumer_rank,
            )

        self._retain_storage_until_stream_completes(storage, rebuild_device_idx)
        self.reconstruct_tensor = reconstructed_tensor
        return self.reconstruct_tensor
