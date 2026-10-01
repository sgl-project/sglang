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
        # Set by enable_rank_shard (SGLANG_ENABLE_HICACHE_RANK_SHARD).
        self.rank_shard = None

    def submit_device_to_host(self, transfers: list[L2Transfer]) -> TransferCompletion:
        start_event = self._start_event(None)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        with device_module.stream(self.device_to_host_stream):
            start_event.wait(self.device_to_host_stream)
            ack_start.record()
            for transfer in transfers:
                transfer.host_pool.backup_from_device_all_layer(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    self.io_backend,
                )
            ack_finish.record()
            self._record_stream(transfers, self.device_to_host_stream)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def submit_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        layer_num: int,
        start_event=None,
        on_layer_done=None,
        fenced: bool = False,
    ) -> TransferCompletion:
        self._note_rank_shard_submit(transfers, fenced)
        start_event = self._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        with device_module.stream(self.host_to_device_stream):
            start_event.wait(self.host_to_device_stream)
            ack_start.record()
            self._load_layers(transfers, layer_num, on_layer_done)
            ack_finish.record()
            self._record_stream(transfers, self.host_to_device_stream)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def _load_layers(self, transfers, layer_num, on_layer_done) -> None:
        """Per layer: its H2D copies, then its completion (on the current stream)."""
        shard = self._begin_rank_shard_load(transfers)
        for layer_id in range(layer_num):
            self._load_layer(transfers, layer_id)
            if shard is not None:
                # Owners' layers go to the other ranks; the layer completes after.
                shard.exchange_layer(layer_id, on_layer_done)
            elif on_layer_done is not None:
                on_layer_done(layer_id)
        if shard is not None:
            shard.finish()

    def enable_rank_shard(self, exchange) -> None:
        self.rank_shard = exchange

    def _begin_rank_shard_load(self, transfers: list[L2Transfer]):
        from sglang.srt.mem_cache.hicache_rank_shard import is_rank_shard_load

        if not is_rank_shard_load(transfers):
            return None
        if self.rank_shard is None:
            raise RuntimeError(
                "HiCache rank-sharded host pool loaded without the rank-shard exchange"
            )
        return self.rank_shard.begin_load(transfers, device_module.current_stream())

    def _note_rank_shard_submit(self, transfers: list[L2Transfer], fenced: bool):
        """Scheduler thread, at submit: the next forward gates on this load."""
        if self.rank_shard is None:
            return
        from sglang.srt.mem_cache.hicache_rank_shard import is_rank_shard_load

        if not is_rank_shard_load(transfers):
            return
        if not fenced:
            # I2 needs the exchange behind every forward already enqueued.
            raise RuntimeError(
                "HiCache rank shard: load-back without the forward-stream load "
                "fence (only cache_controller.start_loading fences it); unset "
                "SGLANG_ENABLE_HICACHE_RANK_SHARD for this path"
            )
        self.rank_shard.submitted += 1

    def gate_rank_shard_forward(self, stream=None) -> None:
        """Before a forward's first kernel (scheduler thread): make `stream` (default:
        current) wait for every submitted rank-shard exchange (I2). Load-backs are
        issued synchronously on this thread in submit_host_to_device, so the whole
        exchange is already enqueued when this runs (I1)."""
        ex = self.rank_shard
        if ex is None or ex.gated == ex.submitted:
            return
        ex.gate_forward(
            device_module.current_stream() if stream is None else stream, 0.0
        )

    def _load_layer(self, transfers: list[L2Transfer], layer_id: int) -> None:
        primary = transfers[0] if transfers else None
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
            transfer.host_pool.load_to_device_per_layer(
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
    def _record_stream(transfers: list[L2Transfer], stream) -> None:
        for transfer in transfers:
            for indices in (transfer.host_indices, transfer.device_indices):
                if indices.is_cuda:
                    indices.record_stream(stream)
