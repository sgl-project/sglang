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


class TransferCompletion(NamedTuple):
    start_event: Any
    finish_event: Any
    timing_enabled: bool


class PrefetchHandle:
    """Handle for on-demand per-layer H2D loading.

    Only layer 0 is loaded in submit; subsequent layers are triggered
    during forward via trigger_layer(), so H2D overlaps with attention
    compute instead of being issued as a single bulk transfer.
    """

    def __init__(
        self,
        engine: "L2TransferEngine",
        transfers: list[L2Transfer],
        layer_num: int,
        ack_finish_event,
        on_layer_done,
    ):
        self._engine = engine
        self._transfers = transfers
        self._layer_num = layer_num
        self._ack_finish_event = ack_finish_event
        self._on_layer_done = on_layer_done
        self._next_layer = 1  # layer 0 already loaded

    def trigger_layer(self, layer_id: int) -> None:
        if layer_id < self._next_layer:
            return
        if layer_id >= self._layer_num:
            return
        self._next_layer = layer_id + 1
        primary = self._transfers[0] if self._transfers else None
        # Hold the shared-layout lease while enqueueing this layer's loads so
        # compaction cannot relocate host pages mid-transfer; the recorded
        # done_event keeps that protection until the loads complete.
        domains = shared_host_layout_domains(t.host_pool for t in self._transfers)
        for domain in domains:
            domain.acquire_layout()
        done_event = None
        try:
            with device_module.stream(self._engine.host_to_device_stream):
                for transfer in self._transfers:
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
                        self._engine.io_backend,
                        is_draft=transfer.is_draft,
                    )
                if self._on_layer_done is not None:
                    self._on_layer_done(layer_id)
                done_event = device_module.Event()
                done_event.record()
                if layer_id == self._layer_num - 1:
                    self._ack_finish_event.record()
        except Exception:
            self._engine.host_to_device_stream.synchronize()
            raise
        finally:
            for domain in reversed(domains):
                domain.release_layout(done_event)

    def drain(self) -> None:
        """Record completion events for any layers not yet triggered."""
        if self._next_layer >= self._layer_num:
            return
        with device_module.stream(self._engine.host_to_device_stream):
            for layer_id in range(self._next_layer, self._layer_num):
                if self._on_layer_done is not None:
                    self._on_layer_done(layer_id)
            self._ack_finish_event.record()
        self._next_layer = self._layer_num


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()
        self._prefetch_handle: Optional[PrefetchHandle] = None

    @property
    def prefetch_handle(self) -> Optional[PrefetchHandle]:
        return self._prefetch_handle

    @staticmethod
    def _resolve_device_indices(transfer: L2Transfer) -> torch.Tensor:
        """Resolve controller IDs into this pool's device-buffer indices.

        Call on the transfer stream after the producer event. The host-transfer
        move gate prevents relocation until the transfer completes.
        """
        # Mamba state pools do not inherit KVCache's default attributes.
        translate = getattr(transfer.device_pool, "host_transfer_translate", None)
        if translate is None:
            return transfer.device_indices
        original_device = transfer.device_indices.device
        # The direct backend supplies CPU indices even for a CUDA pool.
        # Translate alongside the v2p table, then restore the backend's device.
        indices = transfer.device_indices.to(
            getattr(transfer.device_pool, "device", original_device)
        ).contiguous()
        dcp_size = getattr(transfer.host_pool, "dcp_size", 1)
        if dcp_size > 1:
            # The MLA host pool selects this rank and collapses logical IDs.
            # Translate in local virtual space, then preserve that widened
            # interface (including token order) for the host pool.
            resolved = translate(indices // dcp_size) * dcp_size + indices % dcp_size
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
        start_event = self._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        completion = TransferCompletion(ack_start, ack_finish, timing_enabled)
        domains = shared_host_layout_domains(t.host_pool for t in transfers)
        for domain in domains:
            domain.acquire_layout()
        finish_recorded = False
        try:
            with device_module.stream(stream):
                # Index preparation can read producer-owned device indices.
                start_event.wait(stream)
                prepared = self._prepare_transfers(transfers)
                ack_start.record()
                yield prepared, completion
                ack_finish.record()
                finish_recorded = True
                self._record_stream(transfers + prepared, stream)
        except Exception:
            stream.synchronize()
            raise
        finally:
            for domain in reversed(domains):
                domain.release_layout(
                    ack_finish if finish_recorded else None,
                    (id(self), transfer_key),
                )

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
        on_demand: bool = False,
    ) -> TransferCompletion:
        if on_demand:
            return self._submit_host_to_device_on_demand(
                transfers,
                transfer_layer_id_max=transfer_layer_id_max,
                start_event=start_event,
                on_layer_done=on_layer_done,
            )
        with self._submission(
            transfers, self.host_to_device_stream, "host_to_device", start_event
        ) as (transfers, completion):
            primary = transfers[0] if transfers else None
            delay_cycles = envs.SGLANG_TEST_HICACHE_H2D_DELAY_CYCLES.get()
            for layer_id in range(transfer_layer_id_max):
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
                if on_layer_done is not None:
                    on_layer_done(layer_id)
        self._prefetch_handle = None
        return completion

    def _submit_host_to_device_on_demand(
        self,
        transfers: list[L2Transfer],
        *,
        transfer_layer_id_max: int,
        start_event=None,
        on_layer_done=None,
    ) -> TransferCompletion:
        """Load only layer 0; the PrefetchHandle triggers the rest in forward.

        Mirrors _submission's layout-lease and index-preparation semantics, but
        defers the ack_finish recording to the last triggered layer (or drain),
        so the completion event fires only after the whole on-demand transfer.
        """
        start_event = self._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        domains = shared_host_layout_domains(t.host_pool for t in transfers)
        for domain in domains:
            domain.acquire_layout()
        layer0_done = None
        try:
            with device_module.stream(self.host_to_device_stream):
                # Index preparation can read producer-owned device indices.
                start_event.wait(self.host_to_device_stream)
                prepared = self._prepare_transfers(transfers)
                ack_start.record()
                primary = prepared[0] if prepared else None
                for transfer in prepared:
                    local_layer_id = (
                        transfer.layer_mapper(0)
                        if transfer.layer_mapper is not None
                        else 0
                    )
                    if local_layer_id is None or (
                        transfer is not primary
                        and transfer.layer_mapper is None
                        and 0 >= transfer.host_pool.layer_num
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
                if on_layer_done is not None:
                    on_layer_done(0)
                # A recorded event covering layer 0's loads; compaction
                # synchronizes it before relocating host pages. Later layers
                # register their own events in PrefetchHandle.trigger_layer.
                layer0_done = device_module.Event()
                layer0_done.record()
                if transfer_layer_id_max <= 1:
                    ack_finish.record()
                self._record_stream(transfers + prepared, self.host_to_device_stream)
        except Exception:
            self.host_to_device_stream.synchronize()
            raise
        finally:
            for domain in reversed(domains):
                domain.release_layout(layer0_done, (id(self), "host_to_device"))
        self._prefetch_handle = PrefetchHandle(
            self, prepared, transfer_layer_id_max, ack_finish, on_layer_done
        )
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

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
