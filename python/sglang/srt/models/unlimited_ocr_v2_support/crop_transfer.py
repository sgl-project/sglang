"""Pinned host staging and asynchronous crop transfer for Unlimited OCR."""

import contextlib
import logging
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.utils.nvtx_utils import operations_nvtx_range

logger = logging.getLogger(__name__)

# Background threads staging crops into pinned host memory.
_PREFETCH_WORKERS = 4
# Prepared crops held at once, alongside the SGLANG_UOCR_CROP_PREFETCH_MAX_MB cap.
_PREFETCH_MAX_ITEMS = 128
# Event-protected pinned buffers per crop batch shape for synchronous staging.
_PIN_SLOTS = 1


class CropTransferConfig(msgspec.Struct, frozen=True):
    """Store resolved crop-transfer switches and resource limits."""

    enabled: bool
    max_bytes: int


class CropSignature(msgspec.Struct, frozen=True):
    """Describe the identity and mutation state of a crop tensor."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: object
    data_ptr: int
    version: Optional[int]


class CropEntry(msgspec.Struct):
    """Track one prepared crop and its active transfers."""

    source_ref: object
    signature: CropSignature
    size: int
    future: Future
    leases: int = 0
    invalidated: bool = False
    released: bool = False


class RetiredTransfer(msgspec.Struct):
    """Hold crop buffers until their CUDA transfer completes."""

    event: object
    buffers: list[torch.Tensor]
    entries: list[CropEntry]


class PinnedSlot(msgspec.Struct):
    """Store a reusable pinned buffer and its completion event."""

    buffer: torch.Tensor
    event: object = None


def _read_config() -> CropTransferConfig:
    megabytes = envs.SGLANG_UOCR_CROP_PREFETCH_MAX_MB.get()
    if not 1 <= megabytes <= 16384:
        raise ValueError("SGLANG_UOCR_CROP_PREFETCH_MAX_MB must be between 1 and 16384")
    return CropTransferConfig(
        enabled=envs.SGLANG_ENABLE_UOCR_CROP_PREFETCH.get(),
        max_bytes=megabytes * 1024 * 1024,
    )


def _signature(source: torch.Tensor) -> CropSignature:
    try:
        version = source._version
    except RuntimeError:
        version = None
    return CropSignature(
        shape=tuple(source.shape),
        stride=tuple(source.stride()),
        dtype=source.dtype,
        data_ptr=source.data_ptr(),
        version=version,
    )


def _stage(
    source: torch.Tensor,
    dtype: torch.dtype,
    inference_mode: bool,
    grad_enabled: bool,
) -> torch.Tensor:
    with torch.inference_mode(inference_mode), torch.set_grad_enabled(grad_enabled):
        with operations_nvtx_range("ocr.crop_prefetch_allocate"):
            try:
                result = torch.empty(
                    source.shape, dtype=dtype, device="cpu", pin_memory=True
                )
            except RuntimeError:
                with operations_nvtx_range("ocr.crop_prefetch_pageable"):
                    result = torch.empty(source.shape, dtype=dtype, device="cpu")
        with operations_nvtx_range("ocr.crop_prefetch_cpu_copy"):
            result.copy_(source)
    return result


def iter_crop_sources(requests) -> list[torch.Tensor]:
    """Collect eligible crop tensors from tokenized requests in order."""
    from sglang.srt.managers.io_struct import (
        BaseBatchReq,
        TokenizedEmbeddingReqInput,
        TokenizedGenerateReqInput,
    )
    from sglang.srt.managers.schedule_batch import (
        MultimodalInputs,
        MultimodalProcessorOutput,
    )

    sources = []
    for request in requests:
        values = request.batch if isinstance(request, BaseBatchReq) else (request,)
        for value in values:
            if not isinstance(
                value, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
            ):
                continue
            mm_inputs = value.mm_inputs
            if not isinstance(mm_inputs, (MultimodalProcessorOutput, MultimodalInputs)):
                continue
            for item in mm_inputs.mm_items:
                source = item.model_specific_data.get("images_crop")
                if UnlimitedOcrCropTransfer._eligible(source):
                    sources.append(source)
    return sources


class UnlimitedOcrCropTransfer:
    """Prepare Unlimited-OCR crops and transfer them to the model device."""

    def __init__(self, *, model_dtype: Optional[torch.dtype] = None):
        self.closed = True
        self.model_dtype = model_dtype
        self.config = _read_config()
        self.executor = (
            ThreadPoolExecutor(
                max_workers=_PREFETCH_WORKERS,
                thread_name_prefix="sglang-ocr-crop-prefetch",
            )
            if self.config.enabled
            else None
        )
        self.entries: dict[tuple[int, torch.dtype], CropEntry] = {}
        self.retired: list[RetiredTransfer] = []
        self.quarantined: list[tuple[object, list[torch.Tensor], list[CropEntry]]] = []
        self.copy_streams: dict[str, object] = {}
        self.pinned_slots: dict[tuple, list[PinnedSlot]] = {}
        self.pin_cursors: dict[tuple, int] = {}
        self.reserved_bytes = 0
        self.reserved_items = 0
        self.closed = False

    @staticmethod
    def _eligible(source: object) -> bool:
        return (
            isinstance(source, torch.Tensor)
            and source.device.type == "cpu"
            and source.layout == torch.strided
            and source.numel() > 0
            and all(stride != 0 for stride in source.stride())
        )

    def submit(self, source: object, dtype: torch.dtype) -> bool:
        """Submit one eligible crop for background preparation."""
        if self.closed or self.quarantined or not self._eligible(source):
            return False
        key = (id(source), dtype)
        if key in self.entries:
            return False
        size = source.numel() * torch.empty((), dtype=dtype).element_size()
        if (
            self.reserved_bytes + size > self.config.max_bytes
            or self.reserved_items >= _PREFETCH_MAX_ITEMS
        ):
            with operations_nvtx_range("ocr.crop_prefetch_budget_skip"):
                pass
            return False
        future = self.executor.submit(
            _stage,
            source,
            dtype,
            torch.is_inference_mode_enabled(),
            torch.is_grad_enabled(),
        )
        self.entries[key] = CropEntry(
            source_ref=weakref.ref(source),
            signature=_signature(source),
            size=size,
            future=future,
        )
        self.reserved_bytes += size
        self.reserved_items += 1
        return True

    def prefetch_items(self, items: list[object], dtype: torch.dtype) -> None:
        """Prepare crops from multimodal items before model execution."""
        self.prefetch_sources(
            sources=[item.model_specific_data.get("images_crop") for item in items],
            dtype=dtype,
        )

    def prefetch_requests(self, requests) -> None:
        """Prepare crops from scheduler-published requests."""
        if self.model_dtype is None:
            raise RuntimeError("crop prefetch model dtype is not initialized")
        self.prefetch_sources(
            sources=iter_crop_sources(requests),
            dtype=self.model_dtype,
        )

    def prefetch_sources(self, sources, dtype: torch.dtype) -> None:
        """Submit all eligible crop tensors from an input sequence."""
        if not self.config.enabled or self.closed:
            return
        self.reap()
        eligible = [source for source in sources if self._eligible(source)]
        if not eligible:
            return
        with operations_nvtx_range("ocr.crop_prefetch_submit"):
            for source in eligible:
                self.submit(source, dtype)

    def _release(self, entries: list[CropEntry]) -> None:
        fresh = [entry for entry in entries if not entry.released]
        self.reserved_bytes -= sum(entry.size for entry in fresh)
        self.reserved_items -= len(fresh)
        for entry in fresh:
            entry.released = True

    def _invalid(self, entry: CropEntry) -> bool:
        if entry.invalidated:
            return True
        if not entry.future.done():
            return entry.source_ref() is None
        return (
            entry.source_ref() is None
            or entry.future.cancelled()
            or entry.future.exception() is not None
        )

    def _retire_if_unused(self, key: tuple[int, torch.dtype], entry: CropEntry) -> None:
        if entry.leases or not entry.future.done() or not self._invalid(entry):
            return
        if self.entries.get(key) is entry:
            del self.entries[key]
        self._release([entry])

    def reap(self) -> None:
        """Release invalid entries and completed transfer leases."""
        for key, entry in list(self.entries.items()):
            if entry.future.done() and self._invalid(entry):
                if (
                    not entry.future.cancelled()
                    and entry.future.exception() is not None
                ):
                    logger.warning(
                        "Dropping failed crop prefetch: %s", entry.future.exception()
                    )
                entry.invalidated = True
                self._retire_if_unused(key, entry)
        remaining = []
        completed = []
        for retired in self.retired:
            if retired.event is None or retired.event.query():
                completed.extend(retired.entries)
            else:
                remaining.append(retired)
        self.retired = remaining
        for entry in completed:
            if not entry.leases:
                raise RuntimeError("crop prefetch lease accounting underflow")
            entry.leases -= 1
        for key, entry in list(self.entries.items()):
            self._retire_if_unused(key, entry)
        for entry in completed:
            if entry.invalidated and not entry.leases:
                self._release([entry])

    def _copy_stream(self, device: torch.device):
        key = str(device)
        stream = self.copy_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self.copy_streams[key] = stream
        return stream

    def _take_entries(
        self, keys: list[tuple[int, torch.dtype]]
    ) -> dict[tuple[int, torch.dtype], CropEntry]:
        unique_keys = dict.fromkeys(keys)
        return {key: self.entries[key] for key in unique_keys if key in self.entries}

    def _resolve_prepared(
        self,
        key: tuple[int, torch.dtype],
        source: torch.Tensor,
        entry: Optional[CropEntry],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        prepared = None
        if entry is not None:
            span = (
                "ocr.crop_prefetch_ready"
                if entry.future.done()
                else "ocr.crop_prefetch_wait"
            )
            with operations_nvtx_range(span):
                try:
                    candidate = entry.future.result()
                except Exception:
                    logger.warning(
                        "Crop prefetch failed; using native conversion", exc_info=True
                    )
                else:
                    if entry.source_ref() is source and entry.signature == _signature(
                        source
                    ):
                        prepared = candidate
                    else:
                        entry.invalidated = True
        if prepared is None:
            with operations_nvtx_range("ocr.crop_prefetch_fallback_stage"):
                prepared = _stage(
                    source,
                    dtype,
                    torch.is_inference_mode_enabled(),
                    torch.is_grad_enabled(),
                )
        return prepared

    def _finalize_taken(
        self,
        taken: dict[tuple[int, torch.dtype], CropEntry],
        event: object,
        buffers: list[torch.Tensor],
    ) -> None:
        finished = []
        for key, entry in taken.items():
            if not entry.future.done():
                self.entries[key] = entry
                continue
            if not entry.invalidated:
                entry.leases += 1
                finished.append(entry)
                continue
            entry.invalidated = True
            if self.entries.get(key) is entry:
                del self.entries[key]
            if event is None:
                self._release([entry])
            else:
                entry.leases += 1
                finished.append(entry)
        if finished:
            self.retired.append(
                RetiredTransfer(event=event, buffers=buffers, entries=finished)
            )

    def try_stack_prefetched(
        self,
        tensors: list[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Stack prepared crops on the target device when cache entries exist."""
        if self.quarantined:
            raise RuntimeError("Crop prefetch disabled after CUDA stream failure")
        if self.closed or not self.config.enabled or not tensors:
            return None
        first_shape = tuple(tensors[0].shape)
        if any(
            not self._eligible(tensor) or tuple(tensor.shape) != first_shape
            for tensor in tensors
        ):
            return None
        keys = [(id(tensor), dtype) for tensor in tensors]
        if not any(key in self.entries for key in keys):
            with operations_nvtx_range("ocr.crop_prefetch_batch_miss"):
                pass
            return None
        self.reap()
        taken = self._take_entries(keys)
        buffers = []
        prepared_by_key = {}
        event = None
        stream = None
        consumer_stream = None
        try:
            target = torch.device(device)
            # CUDA copies run on a side stream the consumer stream waits on.
            use_copy_stream = target.type == "cuda"
            if use_copy_stream:
                consumer_stream = torch.cuda.current_stream(target)
                stream = self._copy_stream(target)
                event = torch.cuda.Event()
            result = torch.empty(
                (len(tensors), *first_shape), dtype=dtype, device=target
            )
            if use_copy_stream:
                stream.wait_stream(consumer_stream)
            context = (
                torch.cuda.stream(stream)
                if use_copy_stream
                else contextlib.nullcontext()
            )
            with context:
                with operations_nvtx_range("ocr.crop_prefetch_h2d_enqueue"):
                    for index, (key, source) in enumerate(zip(keys, tensors)):
                        prepared = prepared_by_key.get(key)
                        if prepared is None:
                            prepared = self._resolve_prepared(
                                key, source, taken.get(key), dtype
                            )
                        buffers.append(prepared)
                        prepared_by_key[key] = prepared
                        result[index].copy_(prepared, non_blocking=True)
                if event is not None:
                    event.record(stream)
            if use_copy_stream:
                with operations_nvtx_range("ocr.crop_prefetch_copy_stream_wait"):
                    consumer_stream.wait_event(event)
                result.record_stream(stream)
            return result
        finally:
            # An event exists only for a CUDA target, which uses the copy stream.
            if event is not None:
                try:
                    event.query()
                except Exception:
                    try:
                        stream.synchronize()
                    except Exception:
                        quarantined = list(taken.values())
                        for entry in quarantined:
                            if not entry.released:
                                entry.leases += 1
                        self.quarantined.append((stream, buffers, quarantined))
                        raise
                    event = None
            self._finalize_taken(taken, event, buffers)

    def _native_stack(
        self,
        tensors: list[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        first = tensors[0]
        target = torch.device(device)
        if first.device == target:
            return torch.stack(tensors, dim=0).type(dtype)
        count = len(tensors)
        shape = (count, *first.shape)
        key = (shape, dtype, str(target))
        slots = self.pinned_slots.get(key)
        if slots is None:
            try:
                slots = [
                    PinnedSlot(torch.empty(shape, dtype=dtype, pin_memory=True))
                    for _ in range(_PIN_SLOTS)
                ]
            except RuntimeError:
                return torch.stack(tensors, dim=0).type(dtype).to(device=target)
            self.pinned_slots[key] = slots
        cursor = self.pin_cursors.get(key, 0) % len(slots)
        self.pin_cursors[key] = cursor + 1
        slot = slots[cursor]
        if slot.event is not None:
            try:
                ready = slot.event.query()
            except Exception:
                self.quarantined.append((None, [slot.buffer], []))
                raise
            if not ready:
                with operations_nvtx_range("ocr.crop_pin_wait"):
                    slot.event.synchronize()
        buffer = slot.buffer[:count]
        with operations_nvtx_range("ocr.crop_cpu_stage"):
            for index, tensor in enumerate(tensors):
                buffer[index].copy_(tensor)
        if target.type != "cuda":
            with operations_nvtx_range("ocr.crop_h2d_enqueue"):
                return buffer.to(device=target, non_blocking=True)
        stream = torch.cuda.current_stream(target)
        event = torch.cuda.Event()
        slot.event = event
        result = None
        try:
            with operations_nvtx_range("ocr.crop_h2d_enqueue"):
                result = buffer.to(device=target, non_blocking=True)
            event.record(stream)
            return result
        except Exception:
            try:
                stream.synchronize()
            except Exception:
                retained = [slot.buffer]
                if result is not None:
                    retained.append(result)
                self.quarantined.append((stream, retained, []))
                raise
            slot.event = None
            raise

    def stack(
        self,
        tensors: list[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Stack crops on the target device using prefetch when available."""
        if not tensors:
            raise ValueError("crop transfer requires at least one tensor")
        with operations_nvtx_range("ocr.crop_prefetch_consume"):
            prefetched = self.try_stack_prefetched(tensors, dtype, device)
        if prefetched is not None:
            return prefetched
        return self._native_stack(tensors, dtype, device)

    def close(self) -> None:
        """Stop workers and wait for all crop transfers to release storage."""
        if self.closed:
            return
        self.closed = True
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        for retired in self.retired:
            if retired.event is not None:
                retired.event.synchronize()
            for entry in retired.entries:
                entry.leases = max(0, entry.leases - 1)
        for stream, _, entries in self.quarantined:
            if stream is not None:
                stream.synchronize()
            for entry in entries:
                entry.leases = max(0, entry.leases - 1)
        for slots in self.pinned_slots.values():
            for slot in slots:
                if slot.event is not None:
                    slot.event.synchronize()
        self.entries.clear()
        self.retired.clear()
        self.quarantined.clear()
        self.copy_streams.clear()
        self.pinned_slots.clear()
        self.reserved_bytes = 0
        self.reserved_items = 0

    def __del__(self):
        self.close()
