"""Carry captured per-token routing rows with HiCache KV.

The routed-experts and indexer-topk capturers index their host buffer by device KV
slot. HiCache moves KV between device and host slots without recomputing it, so
without this module a host-tier hit reads whatever another request left in the
new device slot.

``backup`` copies a capturer's rows from device slots into a store indexed by host
slot when HiCache writes those slots to host; ``load`` copies them back into the
new device slots when HiCache loads the KV again. A host row only loads after a
successful backup: rows without one raise instead of returning stale routing.

Under overlap or pipeline scheduling a chunk can be stashed (and written through)
before its own forward's capture has been finalized, so its device rows still hold
old data. Inside ``deferring_inflight`` such backups are deferred: the host rows
stay invalid until every capture output still in flight has been finalized, then
copy from the finalized buffer. The stashed node stays on the device meanwhile.

All calls run on the scheduler thread, which also finalizes capture outputs.
"""

from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING, Iterator, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.state_capturer.base import BaseTopkCapturer

_output_ids = itertools.count(1)
_deferral_ids = itertools.count(1)
# Capture outputs created but not yet finalized.
_outstanding: set[int] = set()
# deferral id -> output ids it still waits for.
_waiting: dict[int, set[int]] = {}
# deferral id -> [(capturer, device slots, host slots)] to copy once it resolves.
_deferred: dict[int, list] = {}
# The deferral that backups inside ``deferring_inflight`` join: (id, device slots).
_active: Optional[tuple[int, torch.Tensor]] = None


def _capturers() -> list[BaseTopkCapturer]:
    from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
    from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

    result = []
    for capturer in (get_global_experts_capturer(), get_global_indexer_capturer()):
        if capturer is None:
            continue
        if capturer.host_cache.buffer is None:
            raise RuntimeError(
                f"{capturer.host_cache.name} capturer has no host buffer"
            )
        result.append(capturer)
    return result


class _Carry:
    """Host-slot-indexed copy of one capturer's rows, plus which rows are valid."""

    def __init__(self, capturer: BaseTopkCapturer, host_size: int):
        cache = capturer.host_cache
        self.store = torch.zeros(
            (host_size, cache.num_layers, cache.topk_size), dtype=cache.buffer.dtype
        )
        # Distinguishes a backed-up all-zero row from one that was never backed up.
        self.valid = torch.zeros(host_size, dtype=torch.bool)
        # Nonzero: the deferral whose capture this host row still waits for.
        self.owner = torch.zeros(host_size, dtype=torch.int64)

    def grow(self, host_size: int) -> None:
        if self.store.shape[0] >= host_size:
            return
        old = self.store.shape[0]
        store = torch.zeros((host_size, *self.store.shape[1:]), dtype=self.store.dtype)
        store[:old].copy_(self.store)
        valid = torch.zeros(host_size, dtype=torch.bool)
        valid[:old].copy_(self.valid)
        owner = torch.zeros(host_size, dtype=torch.int64)
        owner[:old].copy_(self.owner)
        self.store, self.valid, self.owner = store, valid, owner


def _carry(capturer: BaseTopkCapturer, host_size: int) -> _Carry:
    carry = getattr(capturer, "_hicache_carry", None)
    if carry is None:
        carry = capturer._hicache_carry = _Carry(capturer, host_size)
    carry.grow(host_size)
    return carry


def _pair(device_indices, host_indices, device_rows: int, host_rows: int):
    for indices in (device_indices, host_indices):
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("HiCache carry indices must be integer tensors")
    device = device_indices.reshape(-1).to("cpu", torch.int64)
    host = host_indices.reshape(-1).to("cpu", torch.int64)
    if device.numel() != host.numel():
        raise ValueError(
            f"HiCache carry has {device.numel()} device slots for {host.numel()} host slots"
        )
    if ((device < 0) | (device >= device_rows)).any() or (
        (host < 0) | (host >= host_rows)
    ).any():
        raise ValueError("HiCache carry slot index out of bounds")
    return device, host


def _copy_rows(source, source_indices, target, target_indices) -> None:
    """Copy runs of consecutive slots as slices; HiCache pages are mostly contiguous."""
    n = source_indices.numel()
    if n >= 32:
        breaks = (
            torch.nonzero(
                (source_indices[1:] != source_indices[:-1] + 1)
                | (target_indices[1:] != target_indices[:-1] + 1),
                as_tuple=True,
            )[0]
            + 1
        )
        if breaks.numel() + 1 <= n // 32:
            bounds = [0, *breaks.tolist(), n]
            starts = torch.tensor(bounds[:-1], dtype=torch.int64)
            for start, end, src, dst in zip(
                bounds[:-1],
                bounds[1:],
                source_indices[starts].tolist(),
                target_indices[starts].tolist(),
            ):
                target[dst : dst + end - start].copy_(source[src : src + end - start])
            return
    target[target_indices] = source[source_indices].to(target.dtype)


def backup(
    device_indices: torch.Tensor, host_indices: torch.Tensor, host_size: int
) -> None:
    """Copy rows of ``device_indices`` into the carry at ``host_indices``."""
    plans = []
    for capturer in _capturers():
        device, host = _pair(
            device_indices, host_indices, capturer.host_cache.buffer.shape[0], host_size
        )
        if device.numel() == 0:
            continue
        if host.unique().numel() != host.numel():
            raise ValueError("HiCache carry backup host slots must be unique")
        pending = torch.zeros(device.numel(), dtype=torch.bool)
        if _active is not None:
            pending = torch.isin(device, _active[1])
        plans.append((capturer, _carry(capturer, host_size), device, host, pending))
    # Invalidate first so a failed copy never exposes the previous rows; a new
    # backup of a host row also cancels an older deferral still targeting it.
    for capturer, carry, device, host, pending in plans:
        carry.valid[host] = False
        carry.owner[host] = 0
        if pending.any():
            carry.owner[host[pending]] = _active[0]
            _deferred.setdefault(_active[0], []).append(
                (capturer, device[pending], host[pending])
            )
    for capturer, carry, device, host, pending in plans:
        ready = ~pending
        if ready.any():
            _copy_rows(
                capturer.host_cache.buffer, device[ready], carry.store, host[ready]
            )
            carry.valid[host[ready]] = True


def load(host_indices: torch.Tensor, device_indices: torch.Tensor) -> None:
    """Copy the carry at ``host_indices`` into the capturer at ``device_indices``."""
    plans = []
    for capturer in _capturers():
        if device_indices.numel() == 0:
            continue
        carry = getattr(capturer, "_hicache_carry", None)
        if carry is None:
            raise _never_backed_up(capturer)
        device, host = _pair(
            device_indices,
            host_indices,
            capturer.host_cache.buffer.shape[0],
            carry.store.shape[0],
        )
        if device.unique().numel() != device.numel():
            raise ValueError("HiCache carry load device slots must be unique")
        if not carry.valid[host].all():
            raise _never_backed_up(capturer)
        plans.append((capturer, carry, device, host))
    for capturer, carry, device, host in plans:
        _copy_rows(carry.store, host, capturer.host_cache.buffer, device)


def _never_backed_up(capturer) -> RuntimeError:
    return RuntimeError(
        f"HiCache is loading {capturer.host_cache.name} rows that were never "
        "backed up; they would return another request's routing"
    )


def register_output() -> int:
    """Called when a capture output is created ahead of its D2H copy."""
    output_id = next(_output_ids)
    _outstanding.add(output_id)
    return output_id


def on_output_finalized(output_id: int) -> None:
    """Called once a capture output's rows are in the capturer host buffer."""
    _outstanding.discard(output_id)
    for deferral_id, waiting in list(_waiting.items()):
        waiting.discard(output_id)
        if not waiting:
            del _waiting[deferral_id]
            _resolve(deferral_id)


def _resolve(deferral_id: int) -> None:
    for capturer, device, host in _deferred.pop(deferral_id, ()):
        carry = capturer._hicache_carry
        owned = carry.owner[host] == deferral_id
        if not owned.any():
            continue
        device, host = device[owned], host[owned]
        _copy_rows(capturer.host_cache.buffer, device, carry.store, host)
        carry.owner[host] = 0
        carry.valid[host] = True


@contextlib.contextmanager
def deferring_inflight(req, get_req_to_token_pool) -> Iterator[None]:
    """Defer carry backups of ``req``'s rows while capture outputs are in flight."""
    global _active
    if not _outstanding or not _capturers():
        yield
        return
    pool = get_req_to_token_pool()
    slots = pool.req_to_token[req.kv.req_pool_idx, : req.extend_end].to(
        "cpu", torch.int64
    )
    deferral_id = next(_deferral_ids)
    previous, _active = _active, (deferral_id, slots)
    try:
        yield
    finally:
        _active = previous
        if deferral_id in _deferred:
            _waiting[deferral_id] = set(_outstanding)


def reset() -> None:
    """Forget in-flight bookkeeping, e.g. when the capturers are destroyed."""
    global _active
    _outstanding.clear()
    _waiting.clear()
    _deferred.clear()
    _active = None
