"""Reusable transient resources for NVIDIA EAGLE decode rounds."""

from __future__ import annotations

import math

import torch

from sglang.srt.speculative.prepared_next_draft import (
    prepared_next_draft_enabled,
)
from sglang.srt.utils.common import async_d2h, is_pin_memory_available

_ROUND_RING_DEPTH = 8
_COPY_EVENT_RING_DEPTH = 4
_MAX_DEVICE_RING_KEYS = 64
_MAX_HOST_RING_KEYS = 32

_join_events: dict[tuple[int, int], torch.cuda.Event] = {}
_copy_event_ring: list[torch.cuda.Event] = []
_copy_event_pos = 0
_round_tensor_rings: dict[tuple, list] = {}
_host_tensor_rings: dict[tuple, list] = {}
_round_contexts: dict[torch.device, list] = {}


def enabled() -> bool:
    return prepared_next_draft_enabled()


def _device_key(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def begin_round(max_batch_size: int, device) -> None:
    """Start one logical speculative round for the bounded tensor arena.

    Each semantic tag may be checked out once in a round.  A tag therefore keeps
    its storage for eight logical rounds instead of sharing a ring with every
    tensor that happens to have the same exact shape.
    """
    if not enabled():
        _round_contexts.pop(torch.device(device), None)
        return
    device = _device_key(device)
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
    context = _round_contexts.get(device)
    if context is None:
        _round_contexts[device] = [int(max_batch_size), 1, set()]
    else:
        context[0] = int(max_batch_size)
        context[1] += 1
        context[2].clear()


def _power_of_two_bucket(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def wait_stream(waiting_stream, source_stream) -> None:
    """Order two streams, reusing one event for each stream pair when enabled."""
    if not enabled():
        waiting_stream.wait_stream(source_stream)
        return
    key = (waiting_stream.cuda_stream, source_stream.cuda_stream)
    event = _join_events.get(key)
    if event is None:
        event = _join_events[key] = torch.cuda.Event()
    event.record(source_stream)
    waiting_stream.wait_event(event)


def empty(
    shape: tuple,
    *,
    tag: str,
    per_request_numel: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Return an exact-shape view of a bounded, fully overwritten device arena.

    ``tag`` identifies one producer output, not its shape.  Reusing a tag in the
    same logical round is a programming error: without this assertion, eight
    same-key allocations could wrap the ring before the preceding round is done.
    Consumers remain ordered on the forward stream or through the existing stream
    joins; no device-to-host synchronization is added here.
    """
    device = _device_key(device)
    context = _round_contexts.get(device)
    if not enabled() or context is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if not tag:
        raise ValueError("a nonempty speculative arena tag is required")
    if per_request_numel <= 0:
        raise ValueError(f"per_request_numel must be positive, got {per_request_numel}")
    used_tags = context[2]
    if tag in used_tags:
        raise AssertionError(
            f"speculative arena tag {tag!r} was checked out twice in one round"
        )
    used_tags.add(tag)

    max_batch_size = context[0]
    factor = _power_of_two_bucket(per_request_numel)
    capacity = max_batch_size * factor
    numel = math.prod(shape)
    if numel > capacity:
        return torch.empty(shape, dtype=dtype, device=device)
    key = (device, tag, factor, dtype, max_batch_size)
    ring = _round_tensor_rings.get(key)
    if ring is None:
        keys_on_device = sum(k[0] == device for k in _round_tensor_rings)
        if keys_on_device >= _MAX_DEVICE_RING_KEYS:
            return torch.empty(shape, dtype=dtype, device=device)
        ring = _round_tensor_rings[key] = [
            [
                torch.empty(capacity, dtype=dtype, device=device)
                for _ in range(_ROUND_RING_DEPTH)
            ],
            0,
        ]
    position = ring[1]
    ring[1] = (position + 1) % _ROUND_RING_DEPTH
    return ring[0][position][:numel].view(shape)


def copy_done_event(device_module):
    """Return an event that remains unused for at least four result copies."""
    global _copy_event_pos
    if not enabled():
        return device_module.Event()
    if not _copy_event_ring:
        _copy_event_ring.extend(
            device_module.Event() for _ in range(_COPY_EVENT_RING_DEPTH)
        )
    event = _copy_event_ring[_copy_event_pos]
    _copy_event_pos = (_copy_event_pos + 1) % _COPY_EVENT_RING_DEPTH
    return event


def copy_to_host(tensor: torch.Tensor, *, tag: str) -> torch.Tensor:
    """Copy into a reusable pinned buffer when speculative decode is enabled."""
    if (
        not enabled()
        or tensor.device.type == "cpu"
        or not is_pin_memory_available(tensor.device)
    ):
        return async_d2h(tensor)
    if not tag:
        raise ValueError("a nonempty host-ring tag is required")
    numel = tensor.numel()
    capacity = _power_of_two_bucket(numel)
    key = (tensor.device, tag, capacity, tensor.dtype)
    ring = _host_tensor_rings.get(key)
    if ring is None:
        keys_on_device = sum(k[0] == tensor.device for k in _host_tensor_rings)
        if keys_on_device >= _MAX_HOST_RING_KEYS:
            return async_d2h(tensor)
        ring = _host_tensor_rings[key] = [
            [
                torch.empty(capacity, dtype=tensor.dtype, pin_memory=True)
                for _ in range(_ROUND_RING_DEPTH)
            ],
            0,
        ]
    host = ring[0][ring[1]][:numel].view(tensor.shape)
    ring[1] = (ring[1] + 1) % _ROUND_RING_DEPTH
    host.copy_(tensor, non_blocking=True)
    tensor.record_stream(torch.get_device_module(tensor.device).current_stream())
    return host
