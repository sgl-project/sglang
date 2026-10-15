"""What the drivers run per mode, and what they hand to it.

The scheduler chooses a mode's stages once (again on a PD role switch); the
flat and the ring driver read the same set. Only `schedule` is required; an
optional stage left None runs the core default inline, so a mode pays no call
for a stage it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler_components.pending_steps import PendingStep
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

    # Build the step of a slot (flat PD prefill also takes its polling args).
    schedule: Callable[..., Optional[ScheduleBatch]]
    # Flat driver, before intake while the engine runs: PD decode starts the
    # prefill DP-rank lookups of its pending rooms.
    prefetch: Optional[Callable[[], None]] = None
    # Flat driver, after intake: the owner advances its own lists and returns
    # the requests that join the core waiting list (PD prefill: bootstrapped;
    # PD decode: None, its admitted requests wait in its ready list).
    intake: Optional[Callable[[], Optional[List[Req]]]] = None
    # Normal schedule, before admission: HiSparse requests whose host copy
    # landed; they join decoding.
    collect_ready: Optional[Callable[[], List[Req]]] = None
    # Normal schedule, admission: dLLM takes its requests from waiting and
    # builds its step (None: prefill admission, after the prefill/decode
    # interval).
    admit: Optional[Callable[[ScheduleSlot], Optional[ScheduleBatch]]] = None
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
    mode's stages take theirs from here."""

    # The normal schedule; dLLM's too, which admits through admit_dllm.
    schedule: Callable[..., Optional[ScheduleBatch]]
    admit_dllm: Callable[[ScheduleSlot], Optional[ScheduleBatch]]
    # PD prefill: the flat schedule, and the ring's intake, schedule and
    # release around each slot.
    schedule_disagg_prefill: Callable[..., Optional[ScheduleBatch]]
    ring_intake_disagg_prefill: Callable[[], None]
    ring_schedule_disagg_prefill: Callable[..., Optional[ScheduleBatch]]
    ring_release_disagg_prefill: Callable[..., None]
    # PD decode, flat and ring.
    schedule_disagg_decode: Callable[..., Optional[ScheduleBatch]]


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
