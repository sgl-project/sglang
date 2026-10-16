"""What the drivers run per mode, and what they hand to it.

The scheduler chooses a mode's stages once (again on a PD role switch); the
flat and the ring driver read the same set and run one core schedule,
`Scheduler.schedule_step`, which calls a mode's schedule stages in a fixed
order. Only `commit_extend` is required; an optional stage left None runs the
core default inline, or nothing, so a mode pays no call for a stage it does
not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler_components.pending_steps import (
        PendingStep,
        PendingSteps,
    )
    from sglang.srt.managers.utils import GenerationBatchResult


@dataclass(slots=True)
class ScheduleSlot:
    """What one step is scheduled for: the decoding requests it may run. The
    flat driver has one slot, over every decoding request; the PP ring has
    one per microbatch slot, over the requests that slot launched."""

    # The ring slot; None for the flat driver.
    index: Optional[int] = None

    def filter_decoding(self, decoding: List[Req]) -> List[Req]:
        """The decoding requests a step of this slot may run."""
        if self.index is None:
            return decoding
        return [req for req in decoding if req.ring_slot == self.index]


@dataclass(frozen=True, slots=True)
class ModeStages:
    """One mode's stages, in driver order."""

    # Commit, the host commit of an extend step; returns the requests that go
    # back to the front of the waiting list (PD prefill: the optimistic
    # requests that yield), or None.
    commit_extend: Callable[[ScheduleBatch, GenerationBatchResult], Optional[List[Req]]]
    # Flat driver, before intake while the engine runs: PD decode starts the
    # prefill DP-rank lookups of its pending rooms.
    prefetch: Optional[Callable[[], None]] = None
    # Flat driver, after intake: PD prefill's bootstrapped requests, which
    # join the core waiting list.
    intake: Optional[Callable[[], List[Req]]] = None
    # Schedule, the HiCache drain at its head (a TP collective): PD decode
    # drains its decode HiCache (None: the core drain, with the storage
    # prefetch retries).
    drain_hicache: Optional[Callable[[], None]] = None
    # Schedule, before admission: flat PD decode advances its own lists
    # (resume retracted, preallocate, poll transfers); its ready requests
    # wait for join_ready.
    before_admission: Optional[Callable[[], None]] = None
    # Schedule, before admission: flat PD prefill resolves the waiting
    # requests' bootstrap; it takes and returns the core waiting list.
    resolve_waiting: Optional[Callable[[List[Req]], List[Req]]] = None
    # Schedule, before admission, once per launched step: PD prefill advances
    # the chunked request's bootstrap and KV send; it takes and returns the
    # chunked request and the waiting list, and reads the pending steps.
    advance_chunked: Optional[
        Callable[
            [Optional[Req], List[Req], PendingSteps],
            Tuple[Optional[Req], List[Req]],
        ]
    ] = None
    # Schedule, after the grammar requeue: the requests ready to decode,
    # which join decoding (HiSparse: their host copy landed; PD decode: their
    # KV arrived and their prefill handoff is committed).
    join_ready: Optional[Callable[[ScheduleSlot], List[Req]]] = None
    # Schedule, admission: the step of newly admitted requests (normal:
    # prefill after the prefill/decode interval; PD prefill: prefill; dLLM:
    # its own step).
    admit: Optional[Callable[[ScheduleSlot], Optional[ScheduleBatch]]] = None
    # Schedule, when admission built no step: the decode step.
    select_decode: Optional[Callable[[ScheduleSlot], Optional[ScheduleBatch]]] = None
    # Launch, before the forward: PD prefill sends cached prefix KV early.
    before_forward: Optional[Callable[[ScheduleBatch], None]] = None
    # Launch: dLLM moves its whole step into its staging list.
    on_launch: Optional[Callable[[ScheduleBatch], None]] = None
    # Launch: where the requests whose last prefill chunk ran go (PD
    # prefill: prefilling; HiSparse: staging; None: decoding).
    on_prefilled: Optional[Callable[[List[Req]], None]] = None
    # Overlap flat driver, the commit readiness gate (PD prefill with
    # continuous input polling): whether a pending step's result is ready on
    # every rank; until it is, the driver returns to intake.
    is_result_ready: Optional[Callable[[Any], bool]] = None
    # Commit, as the finished requests leave decoding (it takes them): PD
    # decode with KV offload moves those whose release waits on a copy into
    # its releasing list.
    hold_finished: Optional[Callable[[List[Req]], None]] = None
    # Commit, after the finished requests leave decoding: HiSparse drops
    # those its prefill commit finished from staging.
    leave_finished: Optional[Callable[[], None]] = None
    # Flat driver, after the commits: PD prefill polls its KV transfers; it
    # takes and returns the core waiting list.
    after_commit: Optional[Callable[[List[Req]], List[Req]]] = None
    # PP ring: builds the PD consensus messages around each slot, from the PP
    # channel and the ring size; under PP they replace the flat intake and
    # after_commit.
    ring_consensus: Optional[Callable[[Any, int], Any]] = None


@dataclass(frozen=True, slots=True)
class LoopSchedules:
    """The scheduler's own schedule stages, filled once by the scheduler; each
    mode's stages take theirs from here. Admission and decode selection come
    through the scheduler because the step planner is built after the mode."""

    # Admission: prefill (PD prefill), prefill after the prefill/decode
    # interval (normal), and dLLM's step.
    admit_prefill: Callable[[ScheduleSlot], Optional[ScheduleBatch]]
    admit_prefill_after_interval: Callable[[ScheduleSlot], Optional[ScheduleBatch]]
    admit_dllm: Callable[[ScheduleSlot], Optional[ScheduleBatch]]
    # The decode step of the decoding requests.
    select_decode: Callable[[ScheduleSlot], Optional[ScheduleBatch]]
    # PD prefill under PP: the ring's intake and release around each slot.
    ring_intake_disagg_prefill: Callable[[], None]
    ring_release_disagg_prefill: Callable[..., None]


@dataclass(frozen=True, slots=True)
class DriverOps:
    """The core stages the PP ring runs for each slot, filled once by the
    scheduler; the ring holds them instead of reaching into the scheduler."""

    # Receive this pass's requests.
    ingest_requests: Callable[[], List]
    # Run a slot's step and keep it pending until the ring commits it.
    launch_step: Callable[..., PendingStep]
    # Commit a pending step, with the outputs the ring brought back.
    commit_step: Callable[[PendingStep], None]
    # Run a slot's forward, behind the ring's PP fences.
    run_batch: Callable[..., GenerationBatchResult]
    # Self-check and reset while the server is idle.
    on_idle: Callable[[], None]
    # Note a ring round that launched nothing.
    mark_idle: Callable[[], None]
    # Whether the engine is paused; a paused pass launches nothing.
    paused_pass: Callable[[], bool]
    # Schedule a slot's step.
    schedule_step: Callable[[ScheduleSlot], Optional[ScheduleBatch]]
