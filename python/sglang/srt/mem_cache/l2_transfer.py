from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import cache
from typing import Any, Callable, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.pool_host.base import uses_shared_host_layout
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


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()

    @staticmethod
    def _layout_domains(transfers: list[L2Transfer]) -> list[Any]:
        domains = []
        seen = set()
        for transfer in transfers:
            if not uses_shared_host_layout(transfer.host_pool):
                continue
            domain = transfer.host_pool.shared_allocation_domain
            if domain is None or id(domain) in seen:
                continue
            seen.add(id(domain))
            domains.append(domain)
        return domains

    def _prepare_transfers(self, transfers: list[L2Transfer]) -> list[L2Transfer]:
        prepared = []
        for transfer in transfers:
            if uses_shared_host_layout(transfer.host_pool):
                host_indices, device_indices = (
                    transfer.host_pool.prepare_transfer_indices(
                        transfer.host_indices,
                        transfer.device_indices,
                        self.io_backend,
                    )
                )
            else:
                host_indices, device_indices = (
                    transfer.host_indices,
                    transfer.device_indices,
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
        domains = self._layout_domains(transfers)
        for domain in domains:
            domain.acquire_layout()
        finish_recorded = False
        try:
            with device_module.stream(stream):
                # Index preparation can read producer-owned device indices.
                start_event.wait(stream)
                transfers = self._prepare_transfers(transfers)
                ack_start.record()
                yield transfers, completion
                ack_finish.record()
                finish_recorded = True
                self._record_stream(transfers, stream)
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
                if uses_shared_host_layout(transfer.host_pool):
                    transfer.host_pool.backup_from_device_all_layer_physical(
                        transfer.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        self.io_backend,
                    )
                else:
                    transfer.host_pool.backup_from_device_all_layer(
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
        layer_num: int,
        start_event=None,
        on_layer_done=None,
    ) -> TransferCompletion:
        with self._submission(
            transfers, self.host_to_device_stream, "host_to_device", start_event
        ) as (transfers, completion):
            primary = transfers[0] if transfers else None
            for layer_id in range(layer_num):
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
                    if uses_shared_host_layout(transfer.host_pool):
                        transfer.host_pool.load_to_device_per_layer_physical(
                            transfer.device_pool,
                            transfer.host_indices,
                            transfer.device_indices,
                            local_layer_id,
                            self.io_backend,
                            is_draft=transfer.is_draft,
                        )
                    else:
                        transfer.host_pool.load_to_device_per_layer(
                            transfer.device_pool,
                            transfer.host_indices,
                            transfer.device_indices,
                            local_layer_id,
                            self.io_backend,
                            is_draft=transfer.is_draft,
                        )
                if on_layer_done is not None:
                    on_layer_done(layer_id)
        return completion

    @staticmethod
    def _start_event(start_event):
        if start_event is None:
            start_event = device_module.Event()
            start_event.record()
        return start_event

    @staticmethod
    def _record_stream(transfers: list[L2Transfer], stream) -> None:
        for transfer in transfers:
            for indices in (transfer.host_indices, transfer.device_indices):
                if indices.is_cuda:
                    indices.record_stream(stream)
