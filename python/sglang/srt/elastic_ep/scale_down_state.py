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
            return

        if state is ScaleDownSurvivorState.DRAIN:
            if not driver.is_drained(self):
                return  # wait another tick for in-flight batches
            driver.on_drain_complete(self)
            self.survivor_state = ScaleDownSurvivorState.FLIP_MASK
            return

        if state is ScaleDownSurvivorState.FLIP_MASK:
            driver.on_flip_mask(self)
            self.survivor_state = ScaleDownSurvivorState.RECONFIG
            return

        if state is ScaleDownSurvivorState.RECONFIG:
            driver.on_reconfig(self)
            self.survivor_state = ScaleDownSurvivorState.COMPLETE
            return

    # --------------------------- retiree per-tick --------------------------

    def _tick_retiree(self, driver: "ScaleDownStateMachineDriver") -> None:
        state = self.retiree_state
        if state is ScaleDownRetireeState.PREPARE:
            driver.on_prepare(self)
            self.retiree_state = ScaleDownRetireeState.DRAIN
            return

        if state is ScaleDownRetireeState.DRAIN:
            if not driver.is_drained(self):
                return
            driver.on_drain_complete(self)
            self.retiree_state = ScaleDownRetireeState.FLIP_MASK
            return

        if state is ScaleDownRetireeState.FLIP_MASK:
            driver.on_flip_mask(self)
            self.retiree_state = ScaleDownRetireeState.LOCAL_CLEANUP
            return

        if state is ScaleDownRetireeState.LOCAL_CLEANUP:
            driver.on_local_cleanup(self)
            self.retiree_state = ScaleDownRetireeState.EXIT
            # Next call to on_exit is expected to never return.
            driver.on_exit(self)


class ScaleDownStateMachineDriver:
    """Abstract driver protocol -- concrete side effects live in the
    ModelRunner; the FSM only owns state transitions."""

    def on_prepare(self, sm: ScaleDownStateMachine) -> None:
        raise NotImplementedError

    def is_drained(self, sm: ScaleDownStateMachine) -> bool:
        raise NotImplementedError

    def on_drain_complete(self, sm: ScaleDownStateMachine) -> None:
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
