"""The scheduler's launch counters and engine flags: live objects shared with
the components that read them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclass(slots=True)
class LaunchCounters:
    """What the launches advanced so far."""

    # The steps run so far; the watchdog's progress counter.
    forward_ct: int = 0
    # Prefill tokens processed so far; used as the aging axis for the HRRN
    # scheduling policy. Reqs snapshot this at waiting_queue entry.
    processed_tokens_counter: int = 0
    # The step this iteration launched; None when it launched nothing.
    cur_batch_for_debug: Optional[ScheduleBatch] = None


@dataclass(slots=True)
class EngineFlags:
    """The engine's run state, set by control requests and the event loop."""

    # Set at construction, cleared once the scheduler is built.
    is_initializing: bool = True
    # Set by a pause request, cleared by continue.
    _engine_paused: bool = False
    # Set by the ShutdownReq handler to break the event loop for graceful shutdown.
    gracefully_exit: bool = False
    # Decode steps may skip a request its pending steps finish (overlap loop).
    enable_skip_finishing_decode: bool = False
    # A sleep before each forward, set by a slow-down request.
    forward_sleep_time: Optional[float] = None
    # The GPU memory types a weight update released and did not resume yet.
    offload_tags: set = field(default_factory=set)
