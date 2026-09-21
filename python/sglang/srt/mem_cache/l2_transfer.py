from __future__ import annotations

import logging
from functools import cache
from typing import Any, Callable, NamedTuple, Optional

import torch

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

    def submit_device_to_host(self, transfers: list[L2Transfer]) -> TransferCompletion:
        start_event = self._start_event(None)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        with device_module.stream(self.device_to_host_stream):
            start_event.wait(self.device_to_host_stream)
            device_indices = [self._resolve_device_indices(t) for t in transfers]
            ack_start.record()
            for transfer, dev_idx in zip(transfers, device_indices):
                transfer.host_pool.backup_from_device_all_layer(
                    transfer.device_pool,
                    transfer.host_indices,
                    dev_idx,
                    self.io_backend,
                )
            ack_finish.record()
            self._record_stream(transfers, self.device_to_host_stream, device_indices)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def submit_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        transfer_layer_id_max: int,
        start_event=None,
        on_layer_done=None,
    ) -> TransferCompletion:
        start_event = self._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        primary = transfers[0] if transfers else None
        with device_module.stream(self.host_to_device_stream):
            start_event.wait(self.host_to_device_stream)
            device_indices = [self._resolve_device_indices(t) for t in transfers]
            ack_start.record()
            for layer_id in range(transfer_layer_id_max):
                for transfer, dev_idx in zip(transfers, device_indices):
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
                    transfer.host_pool.load_to_device_per_layer(
                        transfer.device_pool,
                        transfer.host_indices,
                        dev_idx,
                        local_layer_id,
                        self.io_backend,
                        is_draft=transfer.is_draft,
                    )
                if on_layer_done is not None:
                    on_layer_done(layer_id)
            ack_finish.record()
            self._record_stream(transfers, self.host_to_device_stream, device_indices)
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
            if indices is not None and indices.is_cuda:
                indices.record_stream(stream)
