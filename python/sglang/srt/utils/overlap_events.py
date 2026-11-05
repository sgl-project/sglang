"""Process-local event reuse for the overlap scheduler's bounded result queue."""

import torch

from sglang.srt.environ import envs

WAIT_EVENTS = envs.SGLANG_OPT_OVERLAP_EVENT_REUSE.get()
_join_events: dict = {}
_copy_done_ring: list = []
_copy_done_pos = 0


def wait_stream(dst, src) -> None:
    """dst.wait_stream(src): the same join, through a cached event for the pair (a
    re-record never changes a wait already enqueued)."""
    if not WAIT_EVENTS:
        dst.wait_stream(src)
        return
    key = (dst.cuda_stream, src.cuda_stream)
    event = _join_events.get(key)
    if event is None:
        event = _join_events[key] = torch.cuda.Event()
    event.record(src)
    dst.wait_event(event)


def copy_done_event(device_module):
    """A fresh event, or with WAIT_EVENTS the next of a ring of 4: a result's event is
    synchronized one pass after its record, long before the ring comes round."""
    global _copy_done_pos
    if not WAIT_EVENTS:
        return device_module.Event()
    if not _copy_done_ring:
        _copy_done_ring.extend(device_module.Event() for _ in range(4))
    event = _copy_done_ring[_copy_done_pos]
    _copy_done_pos = (_copy_done_pos + 1) % 4
    return event
