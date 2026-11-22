from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import cache
from typing import Any, Callable, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.base import shared_host_layout_domains
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


@cache
def _timing_events_supported() -> bool:
    try:
        device_module.Event(enable_timing=True)
        return True
    except (TypeError, NotImplementedError):
        logger.warning(
            "%s.Event does not support timing; L2 transfer timing is disabled",
            device_module.__name__,
        )
        return False


@cache
def _warn_direct_without_metadata_kernels() -> None:
    logger.warning(
        "HiCache io backend 'direct' moves KV with copy-engine DMA, which can queue "
        "ahead of the forward's small metadata host-to-device copies and stall the "
        "forward; set SGLANG_ENABLE_MLP_SYNC_METADATA_FILL=1 to write that metadata "
        "with kernels instead."
    )


def make_timing_event_pair():
    timing_enabled = _timing_events_supported()
    kwargs = {"enable_timing": True} if timing_enabled else {}
    return device_module.Event(**kwargs), device_module.Event(**kwargs), timing_enabled


class L2Transfer(NamedTuple):
    host_pool: Any
    device_pool: Any
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    layer_mapper: Optional[Callable[[int], Optional[int]]] = None
    is_draft: bool = False
    # Controller ids -> the rows the device pool indexes (a unified sub-pool's
    # id space); None when they index it as they are.
    device_translate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class TransferCompletion(NamedTuple):
    start_event: Any
    finish_event: Any
    timing_enabled: bool


def host_dcp_size(transfer: L2Transfer) -> int:
    """The decode-context-parallel width of the transfer's host pool: a host
    pool that shards under DCP hands widened controller ids."""
    return getattr(transfer.host_pool, "dcp_size", 1)


def controller_virtual_ids(
    transfer: L2Transfer, indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """The device pool's local virtual ids that a transfer's controller ids
    name: what ``device_translate`` is applied to. Widened ids collapse by the
    host pool's DCP width. The copy and any tracking of the rows it touches
    read the ids through this one rule."""
    ids = transfer.device_indices if indices is None else indices
    dcp_size = host_dcp_size(transfer)
    return ids // dcp_size if dcp_size > 1 else ids


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        if (
            io_backend == "direct"
            and device_module is torch.cuda
            and not envs.SGLANG_ENABLE_MLP_SYNC_METADATA_FILL.get()
        ):
            _warn_direct_without_metadata_kernels()
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()
        # Index preparation runs here, never queued behind earlier copies: the
        # direct backend reads the device indices back on the host.
        self.index_stream = device_module.Stream()
        # Set inside `on_current_stream`: host-to-device submissions then run on
        # the caller's current stream instead of the engine's streams.
        self._h2d_on_current_stream = False

    @contextmanager
    def on_current_stream(self):
        """Run host-to-device submissions made inside the block on the caller's
        current stream, index preparation included, in order with the caller's
        other work there; the engine's own streams stay idle."""
        previous = self._h2d_on_current_stream
        self._h2d_on_current_stream = True
        try:
            yield
        finally:
            self._h2d_on_current_stream = previous

    @staticmethod
    def _resolve_device_indices(transfer: L2Transfer) -> torch.Tensor:
        """Resolve controller IDs into this pool's device-buffer indices.

        Call on the index stream after the producer event: a page move made
        before the submission is ordered before this translation, and one made
        after it waits for the copy's finish event.
        """
        translate = transfer.device_translate
        if translate is None:
            return transfer.device_indices
        original_device = transfer.device_indices.device
        # The direct backend supplies CPU indices even for a CUDA pool.
        # Translate alongside the v2p table, then restore the backend's device.
        indices = transfer.device_indices.to(
            getattr(transfer.device_pool, "device", original_device)
        ).contiguous()
        dcp_size = host_dcp_size(transfer)
        if dcp_size > 1:
            # The MLA host pool selects this rank and collapses logical IDs.
            # Translate in local virtual space, then preserve that widened
            # interface (including token order) for the host pool.
            resolved = (
                translate(controller_virtual_ids(transfer, indices)) * dcp_size
                + indices % dcp_size
            )
        else:
            resolved = translate(indices)
        return resolved.to(original_device)

    def _prepare_transfers(self, transfers: list[L2Transfer]) -> list[L2Transfer]:
        prepared = []
        for transfer in transfers:
            host_indices, device_indices = transfer.host_pool.prepare_transfer_indices(
                transfer.host_indices,
                self._resolve_device_indices(transfer),
                self.io_backend,
            )
            prepared.append(
                transfer._replace(
                    host_indices=host_indices,
                    device_indices=device_indices,
                )
            )
        return prepared

    @contextmanager
    def _submission(self, transfers, stream, transfer_key, start_event=None):
        submission = _TransferSubmission(
            self, transfers, stream, transfer_key, start_event
        )
        try:
            with device_module.stream(stream):
                yield submission.prepared, submission.completion
        except Exception:
            submission.abort()
            raise
        submission.finish()

    def submit_device_to_host(self, transfers: list[L2Transfer]) -> TransferCompletion:
        with self._submission(
            transfers, self.device_to_host_stream, "device_to_host"
        ) as (transfers, completion):
            for transfer in transfers:
                transfer.host_pool.backup_from_device_all_layer_physical(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    self.io_backend,
                )
        return completion

    def submit_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        transfer_layer_id_max: int,
        start_event=None,
        on_layer_done=None,
    ) -> TransferCompletion:
        submission = self.begin_host_to_device(
            transfers,
            transfer_layer_id_max=transfer_layer_id_max,
            start_event=start_event,
            on_layer_done=on_layer_done,
        )
        submission.finish()
        return submission.completion

    def begin_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        transfer_layer_id_max: int,
        start_event=None,
        on_layer_done=None,
    ) -> LayerwiseHostToDevice:
        """Start a host-to-device submission that enqueues no layer yet; the
        caller must ``finish`` it before reading its completion events."""
        stream = (
            device_module.current_stream()
            if self._h2d_on_current_stream
            else self.host_to_device_stream
        )
        return LayerwiseHostToDevice(
            self,
            _TransferSubmission(self, transfers, stream, "host_to_device", start_event),
            num_layers=transfer_layer_id_max,
            on_layer_done=on_layer_done,
        )

    def _load_layer(self, transfers: list[L2Transfer], layer_id: int) -> None:
        primary = transfers[0] if transfers else None
        delay_cycles = envs.SGLANG_TEST_HICACHE_H2D_DELAY_CYCLES.get()
        if delay_cycles:
            torch.cuda._sleep(delay_cycles)
        for transfer in transfers:
            local_layer_id = (
                transfer.layer_mapper(layer_id)
                if transfer.layer_mapper is not None
                else layer_id
            )
            if local_layer_id is None or (
                transfer is not primary
                and transfer.layer_mapper is None
                and layer_id >= transfer.host_pool.layer_num
            ):
                continue
            transfer.host_pool.load_to_device_per_layer_physical(
                transfer.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                local_layer_id,
                self.io_backend,
                is_draft=transfer.is_draft,
            )

    @staticmethod
    def _start_event(start_event):
        if start_event is None:
            start_event = device_module.Event()
            start_event.record()
        return start_event

    @staticmethod
    def _record_stream(transfers: list[L2Transfer], stream, resolved=()) -> None:
        tensors = []
        for transfer in transfers:
            tensors.extend((transfer.host_indices, transfer.device_indices))
        # Keep temporary translated indices alive until the transfer completes.
        tensors.extend(resolved)
        for indices in tensors:
            if indices is not None and (
                indices.is_cuda or indices.device.type == "xpu"
            ):
                indices.record_stream(stream)


class _TransferSubmission:
    """One transfer between index preparation and its finish event."""

    def __init__(self, engine, transfers, stream, transfer_key, start_event):
        self._engine = engine
        self._transfers = transfers
        self.stream = stream
        start_event = engine._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        self.completion = TransferCompletion(ack_start, ack_finish, timing_enabled)
        self._domains = shared_host_layout_domains(t.host_pool for t in transfers)
        for domain in self._domains:
            domain.acquire_layout()
        own_stream = stream in (
            engine.host_to_device_stream,
            engine.device_to_host_stream,
        )
        # A caller's stream prepares indices in order with its own copies.
        self._index_stream = engine.index_stream if own_stream else stream
        # The host layout keeps one finish event per key, and a newer event
        # covers an older one only on the same stream: a caller's stream keeps
        # its own entry beside the engine stream's.
        self._layout_key = (
            (id(engine), transfer_key)
            if own_stream
            else (id(engine), transfer_key, stream)
        )
        try:
            with device_module.stream(self._index_stream):
                # Index preparation can read producer-owned device indices. On its
                # own stream, a host read of them waits only for their producer,
                # never for copies queued earlier on the transfer stream.
                start_event.wait(self._index_stream)
                self.prepared = engine._prepare_transfers(transfers)
                prepared_event = device_module.Event()
                prepared_event.record()
            with device_module.stream(stream):
                prepared_event.wait(stream)
                ack_start.record()
        except Exception:
            self.abort()
            raise

    def finish(self) -> None:
        with device_module.stream(self.stream):
            self.completion.finish_event.record()
        self._engine._record_stream(self._transfers + self.prepared, self.stream)
        self._release(self.completion.finish_event)

    def abort(self) -> None:
        self._index_stream.synchronize()
        self.stream.synchronize()
        self._release(None)

    def _release(self, finish_event) -> None:
        for domain in reversed(self._domains):
            domain.release_layout(finish_event, self._layout_key)


class LayerwiseHostToDevice:
    """Host-to-device submission enqueued layer by layer on demand: a whole batch
    of queued loads would stall every stream sharing its hardware queue."""

    def __init__(
        self,
        engine: L2TransferEngine,
        submission: _TransferSubmission,
        *,
        num_layers: int,
        on_layer_done,
    ):
        self._engine = engine
        self._submission = submission
        self._num_layers = num_layers
        self._on_layer_done = on_layer_done
        self._next_layer = 0
        self.completion = submission.completion

    @property
    def finished(self) -> bool:
        return self._submission is None

    def advance(self, num_layers: int) -> None:
        """Enqueue every layer below ``num_layers``; finishes after the last."""
        submission = self._submission
        if submission is None:
            return
        stop = min(num_layers, self._num_layers)
        try:
            with device_module.stream(submission.stream):
                for layer_id in range(self._next_layer, stop):
                    self._engine._load_layer(submission.prepared, layer_id)
                    if self._on_layer_done is not None:
                        self._on_layer_done(layer_id)
        except Exception:
            self._submission = None
            submission.abort()
            raise
        self._next_layer = max(self._next_layer, stop)
        if self._next_layer == self._num_layers:
            self._submission = None
            submission.finish()

    def finish(self) -> None:
        self.advance(self._num_layers)
