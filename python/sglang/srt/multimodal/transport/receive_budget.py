"""Bound the GPU memory held by CUDA VMM features copied at receive time.

A receive-time copy lives until the request's first prefill forward offloads it,
so a deep queue holds one copy per request on every rank. Copies stay on the GPU
while they fit the per-rank reservation; the rest go to pinned host memory, which
every encoder already accepts from the CPU transport.
"""

from __future__ import annotations

import logging
import time
import weakref
from collections import deque
from collections.abc import Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.req_types import (
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.mem_cache.host_memory import host_memory_headroom_bytes
from sglang.srt.mem_cache.pool_host.base import (
    HICACHE_HOST_MEMORY_RESERVE_BYTES,
    ranks_per_host,
)
from sglang.srt.multimodal.cache.identity import resolve_multimodal_item_identity
from sglang.srt.multimodal.transport.cuda_vmm import CudaVmmTensorTransportProxy

logger = logging.getLogger(__name__)

# A headroom read parses system files; between reads, pinned growth is checked locally.
_HOST_READ_INTERVAL_S = 1.0


class ReceiveFeatureBudget:
    """Places receive-time feature copies on this rank's GPU or pinned host."""

    def __init__(self, *, device_index: int, gpu_budget_bytes: int) -> None:
        self._device_index = device_index
        self._gpu_budget_bytes = gpu_budget_bytes
        # Caps this process's pinned blocks, cached ones included.
        self._host_cap_bytes = 0
        self._host_read_at: float | None = None
        # Includes freed copies the allocator cannot reuse yet.
        self._live_gpu_bytes = 0
        # Finalizers append from any thread; only the scheduler thread pops.
        self._freed: deque[tuple[torch.cuda.Event | None, int]] = deque()
        self._staging_stream: torch.cuda.Stream | None = None

    @property
    def live_gpu_bytes(self) -> int:
        while self._freed and (self._freed[0][0] is None or self._freed[0][0].query()):
            self._live_gpu_bytes -= self._freed.popleft()[1]
        return self._live_gpu_bytes

    def materialize(
        self,
        processor_output: MultimodalProcessorOutput,
        *,
        requires_mm_token_modalities: bool,
    ) -> MultimodalInputs:
        # The items from_processor_output reconstructs at receive time.
        items = [
            item
            for item in processor_output.mm_items
            if item.is_valid()
            and item.has_cuda_ipc_proxy()
            and not item.can_defer_cuda_ipc_feature_reconstruction()
        ]
        proxies = [
            value
            for item in items
            for value in (item.feature, item.precomputed_embeddings)
            if isinstance(value, CudaVmmTensorTransportProxy)
        ]
        nbytes = sum(proxy.data_nbytes for proxy in proxies)
        if proxies and self.live_gpu_bytes + nbytes > self._gpu_budget_bytes:
            self._check_host_room(nbytes)
            self._stage_on_host(items)
        mm_inputs = MultimodalInputs.from_processor_output(
            processor_output,
            requires_mm_token_modalities=requires_mm_token_modalities,
        )
        self._track_gpu_copies(proxies)
        return mm_inputs

    def _check_host_room(self, nbytes: int) -> None:
        # Power-of-two rounding keeps each pinned block under twice its size.
        needed = 2 * nbytes
        now = time.monotonic()
        if (
            self._host_read_at is None
            or now - self._host_read_at > _HOST_READ_INTERVAL_S
        ):
            if self._host_read_at is not None and (
                _pinned() + needed > self._host_cap_bytes
            ):
                # The OS counts freed pinned blocks torch keeps cached as used.
                torch.accelerator.empty_host_cache()
            # What this rank holds plus its share of the host left above the reserve.
            self._host_cap_bytes = _host_allowance() + _pinned()
            self._host_read_at = now
        if _pinned() + needed > self._host_cap_bytes:
            raise RuntimeError(
                f"Cannot stage {nbytes >> 20} MiB of multimodal features: pinned "
                f"host memory is capped at {self._host_cap_bytes >> 20} MiB"
            )

    def _stage_on_host(self, items: Sequence[MultimodalDataItem]) -> None:
        for item in items:
            # Hash the pool's GPU bytes so the pad value matches ranks that keep
            # this request on GPU.
            if item.pad_value is None and item.hash is None:
                if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
                    # GPU ranks would hash the float32 buffer copy instead.
                    raise RuntimeError(
                        "Cannot stage unhashed multimodal items with "
                        "SGLANG_MM_BUFFER_SIZE_MB set"
                    )
                item.set_identity(
                    resolve_multimodal_item_identity(
                        feature=self._gpu_view(item.feature),
                        precomputed_embeddings=self._gpu_view(
                            item.precomputed_embeddings
                        ),
                        modality=item.modality.name,
                        input_format=item.format.name,
                        metadata=item.model_specific_data,
                        namespace=item.processor_fingerprint,
                    )
                )
        if self._staging_stream is None:
            self._staging_stream = torch.cuda.Stream(device=self._device_index)
            logger.warning(
                "Receive-time multimodal copies on GPU %d exceed %d MiB; staging "
                "the overflow in pinned host memory.",
                self._device_index,
                self._gpu_budget_bytes >> 20,
            )
        with torch.cuda.stream(self._staging_stream):
            for item in items:
                if isinstance(item.feature, CudaVmmTensorTransportProxy):
                    item.feature = item.feature.reconstruct_on_target_device(
                        self._device_index, to_host=True
                    )
                if isinstance(item.precomputed_embeddings, CudaVmmTensorTransportProxy):
                    item.precomputed_embeddings = (
                        item.precomputed_embeddings.reconstruct_on_target_device(
                            self._device_index, to_host=True
                        )
                    )
        # Consumers read staged features on the CPU or from other streams.
        self._staging_stream.synchronize()

    def _gpu_view(self, value):
        if isinstance(value, CudaVmmTensorTransportProxy):
            return value.mapped_view(self._device_index)
        return value

    def _track_gpu_copies(self, proxies: Sequence[CudaVmmTensorTransportProxy]) -> None:
        # Packed children share one storage: charge it once.
        storages = {}
        for proxy in proxies:
            copy = proxy.reconstruct_tensor
            if isinstance(copy, torch.Tensor) and copy.is_cuda:
                storage = copy.untyped_storage()
                storages[storage.data_ptr()] = storage
        for storage in storages.values():
            self._live_gpu_bytes += storage.nbytes()
            finalizer = weakref.finalize(storage, self._on_copy_freed, storage.nbytes())
            finalizer.atexit = False

    def _on_copy_freed(self, nbytes: int) -> None:
        event = None
        # An event recorded under capture cannot be queried later.
        if not torch.cuda.is_current_stream_capturing():
            # prepare_mm_inputs drops the copy with its record_stream'd stream
            # current, so the block is reusable once this event completes.
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self._device_index))
        self._freed.append((event, nbytes))


def _pinned() -> int:
    return torch.cuda.host_memory_stats()["allocated_bytes.current"]


def _host_allowance() -> int:
    headroom = host_memory_headroom_bytes() - HICACHE_HOST_MEMORY_RESERVE_BYTES
    return max(0, headroom // ranks_per_host())
