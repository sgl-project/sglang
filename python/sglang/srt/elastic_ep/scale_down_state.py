"""Mooncake-native scale-down state machine.

Per-rank finite state machine that drives a Mooncake-native scale-DOWN
across scheduler ticks, symmetric with the grow-direction handler
:func:`ModelRunner.maybe_join_ep_ranks`. Splits the retirement flow into
explicit stages so that:

* Retirees can drain in-flight batches over multiple ticks instead of a
  single busy-loop.
* Any stage can fail and either retry (transient) or escalate to
  ``scale_phase = "failed"`` (permanent) without corrupting the state
  manager.
* Debug tools (``/is_scaling_elastic_ep``) can surface per-tick progress
  via ``ElasticEPStateManager.scale_phase``.

State layout:

    ┌──────────────────────┐              ┌──────────────────────┐
    │ ScaleDownSurvivor    │              │ ScaleDownRetiree     │
    │    PREPARE           │              │    PREPARE           │
    │       │              │              │       │              │
    │    DRAIN             │              │    DRAIN             │
    │       │              │              │       │              │
    │    FLIP_MASK ────────┼── barrier ───┤    FLIP_MASK         │
    │       │              │              │       │              │
    │    RECONFIG          │              │    LOCAL_CLEANUP     │
    │       │              │              │       │              │
    │    COMPLETE          │              │    EXIT (sys.exit 0) │
    └──────────────────────┘              └──────────────────────┘

The ``FLIP_MASK`` stage is a cohort-wide barrier: every rank posts one
:func:`retire_barrier` on WORLD before writing 0 into the mask. This is
the last collective retirees post -- after it the mask says they're
inactive, and any subsequent collective on WORLD skips them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

logger = logging.getLogger(__name__)


class ScaleDownSurvivorState(Enum):
    """Survivor-side lifecycle. Advances one stage per tick when the guard
    condition (see :meth:`ScaleDownStateMachine.tick`) is satisfied."""

    PREPARE = auto()
    DRAIN = auto()
    FLIP_MASK = auto()
    RECONFIG = auto()
    COMPLETE = auto()
    FAILED = auto()


class ScaleDownRetireeState(Enum):
    """Retiree-side lifecycle. Terminal state is :attr:`EXIT`, at which
    point :func:`ModelRunner._retire_and_exit` calls ``sys.exit(0)`` and
    never returns."""

    PREPARE = auto()
    DRAIN = auto()
    FLIP_MASK = auto()
    LOCAL_CLEANUP = auto()
    EXIT = auto()
    FAILED = auto()


@dataclass
class ScaleDownStateMachine:
    """Per-rank driver for the scale-down flow.

    Owned by :class:`ModelRunner`; instantiated lazily on the first tick
    that observes a pending shrink and torn down when the terminal state
    is reached (or ``fail_scale`` fires).
    """

    is_retiree: bool
    target_size: int
    effective_size: int
    ranks_to_retire: List[int]
    my_global_rank: int
    started_at: float = field(default_factory=time.monotonic)
    survivor_state: ScaleDownSurvivorState = ScaleDownSurvivorState.PREPARE
    retiree_state: ScaleDownRetireeState = ScaleDownRetireeState.PREPARE
    last_error: Optional[str] = None
    # Async barrier handle posted at the end of DRAIN and polled on
    # subsequent ticks until every rank has reached the barrier. Stored
    # as ``object`` because :class:`torch.distributed.Work` is not
    # available at import time on every backend. See
    # ``retire_barrier_post`` in ``elastic_ep.py`` for the rationale for
    # posting the barrier asynchronously instead of blocking.
    _drain_barrier_handle: Optional[object] = None
    # Number of consecutive ``check_drain_barrier`` polls that returned
    # ``False`` for the current handle. Used to fall through to a
    # blocking ``wait()`` after :data:`DRAIN_BARRIER_MAX_POLLS` polls to
    # work around :meth:`torch.distributed.Work.is_completed`
    # unreliability on the Mooncake WORLD backend -- see
    # :func:`retire_barrier_check`.
    _drain_barrier_polls: int = 0

    # Number of ``is_completed`` polls before the DRAIN state gives up
    # and calls ``wait()`` unconditionally. Sized so a healthy async
    # barrier that has actually completed will report so within the
    # window (a few ticks are enough), but a stuck ``is_completed``
    # never wedges the FSM for more than a second. Each scheduler tick
    # is on the order of tens of milliseconds; 100 polls = ~1-2s of
    # wall clock, at which point we know every cohort rank has entered
    # ``retire_barrier_post`` (DRAIN's ``is_drained`` guard) so
    # ``wait()`` cannot deadlock.
    DRAIN_BARRIER_MAX_POLLS: int = 100

    # ---------------------------- observability ----------------------------

    @property
    def phase_name(self) -> str:
        """Human-readable phase string mirrored into ElasticEPStateManager."""
        if self.is_retiree:
            return f"retiree:{self.retiree_state.name.lower()}"
        return f"survivor:{self.survivor_state.name.lower()}"

    def is_terminal(self) -> bool:
        if self.is_retiree:
            return self.retiree_state in (
                ScaleDownRetireeState.EXIT,
                ScaleDownRetireeState.FAILED,
            )
        return self.survivor_state in (
            ScaleDownSurvivorState.COMPLETE,
            ScaleDownSurvivorState.FAILED,
        )

    def is_failed(self) -> bool:
        if self.is_retiree:
            return self.retiree_state is ScaleDownRetireeState.FAILED
        return self.survivor_state is ScaleDownSurvivorState.FAILED

    # ---------------------------- transitions ------------------------------

    def _fail(self, error: str) -> None:
        self.last_error = error
        if self.is_retiree:
            self.retiree_state = ScaleDownRetireeState.FAILED
        else:
            self.survivor_state = ScaleDownSurvivorState.FAILED
        logger.error(
            "[Elastic EP][scale-down FSM] rank=%d %s failed: %s",
            self.my_global_rank,
            "retiree" if self.is_retiree else "survivor",
            error,
        )

    def _advance_survivor(self) -> Optional[ScaleDownSurvivorState]:
        """Return the next survivor state (or ``None`` to stay put)."""
        transitions = {
            ScaleDownSurvivorState.PREPARE: ScaleDownSurvivorState.DRAIN,
            ScaleDownSurvivorState.DRAIN: ScaleDownSurvivorState.FLIP_MASK,
            ScaleDownSurvivorState.FLIP_MASK: ScaleDownSurvivorState.RECONFIG,
            ScaleDownSurvivorState.RECONFIG: ScaleDownSurvivorState.COMPLETE,
        }
        return transitions.get(self.survivor_state)

    def _advance_retiree(self) -> Optional[ScaleDownRetireeState]:
        transitions = {
            ScaleDownRetireeState.PREPARE: ScaleDownRetireeState.DRAIN,
            ScaleDownRetireeState.DRAIN: ScaleDownRetireeState.FLIP_MASK,
            ScaleDownRetireeState.FLIP_MASK: ScaleDownRetireeState.LOCAL_CLEANUP,
            ScaleDownRetireeState.LOCAL_CLEANUP: ScaleDownRetireeState.EXIT,
        }
        return transitions.get(self.retiree_state)

    # --------------------------- per-tick driver ---------------------------

    def tick(self, driver: "ScaleDownStateMachineDriver") -> None:
        """Advance the state machine by up to one stage per call.

        The caller (``ModelRunner.maybe_retire_ep_ranks``) is responsible
        for actually running the side effects requested by the driver
        callbacks -- this class only owns the FSM transitions.
        """
        if self.is_terminal():
            return

        try:
            if self.is_retiree:
                self._tick_retiree(driver)
            else:
                self._tick_survivor(driver)
        except Exception as exc:  # noqa: BLE001 -- FSM errors are terminal
            import traceback

            tb = traceback.format_exc()
            logger.error(
                "[Elastic EP][scale-down FSM] rank=%d %s exception in tick "
                "(state=%s):\n%s",
                self.my_global_rank,
                "retiree" if self.is_retiree else "survivor",
                self.phase_name,
                tb,
            )
            self._fail(f"{type(exc).__name__}: {exc}")

    # -------------------------- survivor per-tick --------------------------

    def _tick_survivor(self, driver: "ScaleDownStateMachineDriver") -> None:
        state = self.survivor_state
        if state is ScaleDownSurvivorState.PREPARE:
            driver.on_prepare(self)
            self.survivor_state = ScaleDownSurvivorState.DRAIN
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d survivor PREPARE->DRAIN",
                self.my_global_rank,
            )
            return

        if state is ScaleDownSurvivorState.DRAIN:
            if not driver.is_drained(self):
                return  # wait another tick for in-flight batches
            if self._drain_barrier_handle is None:
                # First tick with an empty batch queue: post the async
                # retire barrier on WORLD and keep looping. See the
                # ``retire_barrier_post`` docstring for why this must be
                # async instead of blocking.
                self._drain_barrier_handle = driver.post_drain_barrier(self)
                self._drain_barrier_polls = 0
                return
            if not driver.check_drain_barrier(self._drain_barrier_handle):
                self._drain_barrier_polls += 1
                if self._drain_barrier_polls < self.DRAIN_BARRIER_MAX_POLLS:
                    return  # wait for lagging cohort ranks to post
                # ``is_completed`` unreliability fallback -- see field docstring.
                logger.info(
                    "[Elastic EP][scale-down FSM] rank=%d survivor DRAIN "
                    "polled %d times without is_completed=True; falling "
                    "through to blocking wait()",
                    self.my_global_rank,
                    self._drain_barrier_polls,
                )
            t0 = time.monotonic()
            driver.consume_drain_barrier(self._drain_barrier_handle)
            self._drain_barrier_handle = None
            self._drain_barrier_polls = 0
            self.survivor_state = ScaleDownSurvivorState.FLIP_MASK
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d survivor DRAIN->FLIP_MASK "
                "(consume=%.2fs)",
                self.my_global_rank,
                time.monotonic() - t0,
            )
            return

        if state is ScaleDownSurvivorState.FLIP_MASK:
            t0 = time.monotonic()
            driver.on_flip_mask(self)
            self.survivor_state = ScaleDownSurvivorState.RECONFIG
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d survivor "
                "FLIP_MASK->RECONFIG (try_retire_ranks took %.2fs)",
                self.my_global_rank,
                time.monotonic() - t0,
            )
            return

        if state is ScaleDownSurvivorState.RECONFIG:
            t0 = time.monotonic()
            driver.on_reconfig(self)
            self.survivor_state = ScaleDownSurvivorState.COMPLETE
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d survivor "
                "RECONFIG->COMPLETE (finalize_scale_down took %.2fs)",
                self.my_global_rank,
                time.monotonic() - t0,
            )
            return

    # --------------------------- retiree per-tick --------------------------

    def _tick_retiree(self, driver: "ScaleDownStateMachineDriver") -> None:
        state = self.retiree_state
        if state is ScaleDownRetireeState.PREPARE:
            driver.on_prepare(self)
            self.retiree_state = ScaleDownRetireeState.DRAIN
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d retiree PREPARE->DRAIN",
                self.my_global_rank,
            )
            return

        if state is ScaleDownRetireeState.DRAIN:
            if not driver.is_drained(self):
                return
            if self._drain_barrier_handle is None:
                self._drain_barrier_handle = driver.post_drain_barrier(self)
                self._drain_barrier_polls = 0
                return
            if not driver.check_drain_barrier(self._drain_barrier_handle):
                self._drain_barrier_polls += 1
                if self._drain_barrier_polls < self.DRAIN_BARRIER_MAX_POLLS:
                    return
                logger.info(
                    "[Elastic EP][scale-down FSM] rank=%d retiree DRAIN "
                    "polled %d times without is_completed=True; falling "
                    "through to blocking wait()",
                    self.my_global_rank,
                    self._drain_barrier_polls,
                )
            t0 = time.monotonic()
            driver.consume_drain_barrier(self._drain_barrier_handle)
            self._drain_barrier_handle = None
            self._drain_barrier_polls = 0
            self.retiree_state = ScaleDownRetireeState.FLIP_MASK
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d retiree DRAIN->FLIP_MASK "
                "(consume=%.2fs)",
                self.my_global_rank,
                time.monotonic() - t0,
            )
            return

        if state is ScaleDownRetireeState.FLIP_MASK:
            t0 = time.monotonic()
            driver.on_flip_mask(self)
            self.retiree_state = ScaleDownRetireeState.LOCAL_CLEANUP
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d retiree "
                "FLIP_MASK->LOCAL_CLEANUP (try_retire_ranks took %.2fs)",
                self.my_global_rank,
                time.monotonic() - t0,
            )
            return

        if state is ScaleDownRetireeState.LOCAL_CLEANUP:
            t0 = time.monotonic()
            driver.on_local_cleanup(self)
            self.retiree_state = ScaleDownRetireeState.EXIT
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d retiree "
                "LOCAL_CLEANUP->EXIT (took %.2fs)",
                self.my_global_rank,
                time.monotonic() - t0,
            )
            driver.on_exit(self)


class ScaleDownStateMachineDriver:
    """Abstract driver protocol -- concrete side effects live in the
    ModelRunner; the FSM only owns state transitions."""

    def on_prepare(self, sm: ScaleDownStateMachine) -> None:
        raise NotImplementedError

    def is_drained(self, sm: ScaleDownStateMachine) -> bool:
        raise NotImplementedError

    def post_drain_barrier(self, sm: ScaleDownStateMachine) -> Optional[object]:
        """Post the async retire barrier and return an opaque handle.

        The FSM stores the handle and polls
        :meth:`check_drain_barrier` on every tick until it reports
        completion. Returning ``None`` means "no async barrier to wait
        on" -- the FSM then advances to FLIP_MASK on the next tick.
        """
        raise NotImplementedError

    def check_drain_barrier(self, handle: object) -> bool:
        """Non-blocking probe: has every cohort rank posted the barrier
        yet? Returning ``True`` means the FSM should call
        :meth:`consume_drain_barrier` and transition to FLIP_MASK."""
        raise NotImplementedError

    def consume_drain_barrier(self, handle: object) -> None:
        """Finalize (``wait()``) an already-completed barrier handle."""
        raise NotImplementedError

    def on_flip_mask(self, sm: ScaleDownStateMachine) -> None:
        raise NotImplementedError

    # Survivor-only.
    def on_reconfig(self, sm: ScaleDownStateMachine) -> None:
        raise NotImplementedError

    # Retiree-only.
    def on_local_cleanup(self, sm: ScaleDownStateMachine) -> None:
        raise NotImplementedError

    def on_exit(self, sm: ScaleDownStateMachine) -> None:
        raise NotImplementedError
