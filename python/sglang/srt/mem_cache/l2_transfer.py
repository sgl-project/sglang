from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.pool_host.base import shared_host_layout_domains
from sglang.srt.utils import get_device_module, is_cuda

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host.direct import DirectTransfer

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


def local_layers_for_block(
    transfer: L2Transfer, first: int, end: int, last: int | None = None
) -> list[int]:
    if transfer.layer_mapper is not None:
        return [
            local
            for layer in range(first, end)
            if (local := transfer.layer_mapper(layer)) is not None
        ]
    # Batched draft restores can extend beyond the target's final block.
    local_end = (
        transfer.host_pool.layer_num
        if end == last
        else min(end, transfer.host_pool.layer_num)
    )
    return list(range(first, local_end))


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str, host_pool=None):
        self.io_backend = io_backend
        self.batch_direct = False
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()
        self._direct_layouts: dict[bool, tuple[tuple, DirectTransfer]] = {}
        if host_pool is not None:
            self.configure_batch_direct(host_pool)

    def _prepare_direct(
        self, transfers: list[L2Transfer], block_ends: tuple[int, ...] | None = None
    ) -> DirectTransfer | None:
        if not self.batch_direct:
            return None
        from sglang.srt.mem_cache.pool_host.direct import (
            _direct_layout_key,
            prepare_direct_transfer,
        )

        # Cache one layout per direction. Buffer changes invalidate it and release
        # the old pools and pinned views.
        key = _direct_layout_key(transfers, block_ends)
        direction = block_ends is not None
        cached = self._direct_layouts.get(direction)
        if cached is None or cached[0] != key:
            cached = (key, prepare_direct_transfer(transfers, block_ends))
            self._direct_layouts[direction] = cached
        return cached[1]

    def configure_batch_direct(self, host_pool) -> None:
        if self.io_backend != "direct" or not is_cuda():
            return
        from sglang.srt.mem_cache.pool_host.direct import can_batch_direct

        self._direct_layouts.clear()
        self.batch_direct = can_batch_direct(host_pool)
        if self.batch_direct:
            self.enable_batch_direct()

    def enable_batch_direct(self) -> None:
        from sglang.srt.mem_cache.pool_host.direct import load_block_dma_module

        assert self.io_backend == "direct" and is_cuda()
        # Compile during cache initialization, never on the first restore.
        load_block_dma_module()
        self.batch_direct = True

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
        return self._submit(transfers, None)

    def submit_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        transfer_layer_id_max: int,
        start_event=None,
        on_layer_done=None,
        layer_block_ends: Optional[tuple[int, ...]] = None,
    ) -> TransferCompletion:
        block_ends = (
            layer_block_ends
            if layer_block_ends is not None
            else tuple(range(1, transfer_layer_id_max + 1))
        )
        return self._submit(transfers, block_ends, start_event, on_layer_done)

    def _submit(
        self,
        transfers: list[L2Transfer],
        block_ends: tuple[int, ...] | None,
        start_event: Any = None,
        on_layer_done: Callable[[int], None] | None = None,
    ) -> TransferCompletion:
        stream = (
            self.host_to_device_stream
            if block_ends is not None
            else self.device_to_host_stream
        )
        transfer_key = "host_to_device" if block_ends is not None else "device_to_host"
        with self._submission(transfers, stream, transfer_key, start_event) as (
            transfers,
            completion,
        ):
            direct = self._prepare_direct(transfers, block_ends)
            device_indices = [transfer.device_indices for transfer in transfers]
            if direct is not None:
                direct.submit(transfers, device_indices, stream, on_layer_done)
            else:
                self._submit_pools(transfers, device_indices, block_ends, on_layer_done)
        return completion

    def _submit_pools(
        self,
        transfers: list[L2Transfer],
        device_indices: list[torch.Tensor],
        block_ends: tuple[int, ...] | None,
        on_layer_done: Callable[[int], None] | None,
    ) -> None:
        if block_ends is None:
            for transfer, dev_idx in zip(transfers, device_indices):
                transfer.host_pool.backup_from_device_all_layer_physical(
                    transfer.device_pool,
                    transfer.host_indices,
                    dev_idx,
                    self.io_backend,
                )
            return
        for first, end in zip((0, *block_ends), block_ends):
            for transfer, dev_idx in zip(transfers, device_indices):
                for layer in local_layers_for_block(transfer, first, end):
                    transfer.host_pool.load_to_device_per_layer_physical(
                        transfer.device_pool,
                        transfer.host_indices,
                        dev_idx,
                        layer,
                        self.io_backend,
                        is_draft=transfer.is_draft,
                    )
            if on_layer_done is not None:
                # Includes all auxiliary pools, including a partial last block.
                on_layer_done(end - 1)

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
