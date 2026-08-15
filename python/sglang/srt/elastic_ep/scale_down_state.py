"""Mooncake-native scale-down state machine (per-rank FSM across scheduler ticks).

    Survivor:  PREPARE -> DRAIN --barrier-- NIXL_RETIRE --barrier-- FLIP_MASK -> RECONFIG -> COMPLETE
    Retiree:   PREPARE -> DRAIN --barrier-- NIXL_RETIRE --barrier-- FLIP_MASK -> LOCAL_CLEANUP -> EXIT

Cohort barriers are async (blocking WORLD wait would deadlock mlp_sync on same PG).
FLIP_MASK is local; the preceding barrier already excluded races.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

logger = logging.getLogger(__name__)


class ScaleDownState(Enum):
    PREPARE = auto()
    DRAIN = auto()
    NIXL_RETIRE = auto()
    FLIP_MASK = auto()
    RECONFIG = auto()       # survivor-only
    COMPLETE = auto()       # survivor terminal
    LOCAL_CLEANUP = auto()  # retiree-only
    EXIT = auto()           # retiree terminal
    FAILED = auto()


_TERMINAL = frozenset({ScaleDownState.COMPLETE, ScaleDownState.EXIT, ScaleDownState.FAILED})


@dataclass
class ScaleDownStateMachine:
    """Per-rank scale-down driver owned by ModelRunner; torn down at terminal."""

    is_retiree: bool
    target_size: int
    effective_size: int
    ranks_to_retire: List[int]
    my_global_rank: int
    started_at: float = field(default_factory=time.monotonic)
    state: ScaleDownState = ScaleDownState.PREPARE
    last_error: Optional[str] = None
    _drain_barrier_handle: Optional[object] = None
    _drain_barrier_polls: int = 0
    _nixl_barrier_handle: Optional[object] = None
    _nixl_post_attempts: int = 0
    # Bounded polls -> blocking wait() (Mooncake is_completed flake guard).
    DRAIN_BARRIER_MAX_POLLS: int = 100
    # Bounded re-arms, then fail rather than flip the mask unsynchronized.
    NIXL_POST_MAX_ATTEMPTS: int = 20

    @property
    def role(self) -> str:
        return "retiree" if self.is_retiree else "survivor"

    @property
    def _post_flip(self) -> ScaleDownState:
        return ScaleDownState.LOCAL_CLEANUP if self.is_retiree else ScaleDownState.RECONFIG

    @property
    def phase_name(self) -> str:
        return f"{self.role}:{self.state.name.lower()}"

    def is_terminal(self) -> bool:
        return self.state in _TERMINAL

    def is_failed(self) -> bool:
        return self.state is ScaleDownState.FAILED

    def _fail(self, error: str) -> None:
        self.last_error = error
        self.state = ScaleDownState.FAILED
        logger.error("[Elastic EP][scale-down FSM] rank=%d %s failed: %s",
                     self.my_global_rank, self.role, error)

    def _log(self, msg: str, *args) -> None:
        logger.info("[Elastic EP][scale-down FSM] rank=%d " + msg, self.my_global_rank, *args)

    def tick(self, driver: "ScaleDownStateMachineDriver") -> None:
        """Advance by one stage per call. Side effects live in driver callbacks."""
        if self.is_terminal():
            return
        try:
            self._advance(driver)
        except Exception as exc:  # noqa: BLE001
            import traceback
            logger.error("[Elastic EP][scale-down FSM] rank=%d %s tick exc (state=%s):\n%s",
                         self.my_global_rank, self.role, self.phase_name, traceback.format_exc())
            self._fail(f"{type(exc).__name__}: {exc}")

    def _fold_nixl_to_flip(self, driver: "ScaleDownStateMachineDriver") -> bool:
        """Same-tick NIXL barrier -> FLIP_MASK -> post_flip. Returns True if folded.

        Posts once per cycle: re-posting double-counts the store arrival/ready
        counters, mis-electing the leader and letting a subset of ranks satisfy
        the barrier.
        """
        if self._nixl_barrier_handle is None:
            self._nixl_post_attempts += 1
            if self._nixl_post_attempts > self.NIXL_POST_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"NIXL retire barrier failed to arm after "
                    f"{self._nixl_post_attempts - 1} attempts"
                )
            self._nixl_barrier_handle = driver.post_nixl_retire_barrier(self)
        if not driver.check_nixl_retire_barrier(self._nixl_barrier_handle):
            return False
        driver.consume_nixl_retire_barrier(self._nixl_barrier_handle)
        if not self.is_retiree:
            driver.on_nixl_retire_pre(self)
        self._nixl_barrier_handle = None
        self.state = ScaleDownState.FLIP_MASK
        driver.on_flip_mask(self)
        self.state = self._post_flip
        return True

    def _advance(self, driver: "ScaleDownStateMachineDriver") -> None:
        S = ScaleDownState
        role = self.role
        if self.state is S.PREPARE:
            driver.on_prepare(self)
            self.state = S.DRAIN
            self._log("%s PREPARE->DRAIN", role)
            return

        if self.state is S.DRAIN:
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
                self._log("%s DRAIN polled %d; fall to wait()", role, self._drain_barrier_polls)
            driver.consume_drain_barrier(self._drain_barrier_handle)
            self._drain_barrier_handle = None
            self._drain_barrier_polls = 0
            self.state = S.NIXL_RETIRE
            self._log("%s DRAIN->NIXL_RETIRE", role)
            # Log the state actually reached; an unfolded attempt stays here.
            if self._fold_nixl_to_flip(driver):
                self._log("%s NIXL_RETIRE->FLIP_MASK->%s [folded]", role, self._post_flip.name)
            else:
                self._log("%s NIXL_RETIRE awaiting cohort", role)
            return

        if self.state is S.NIXL_RETIRE:
            if self._fold_nixl_to_flip(driver):
                self._log("%s NIXL_RETIRE->FLIP_MASK->%s [deferred]",
                          role, self._post_flip.name)
            return

        if self.state is S.RECONFIG:
            driver.on_reconfig(self)
            self.state = S.COMPLETE
            self._log("survivor RECONFIG->COMPLETE")
            return

        if self.state is S.LOCAL_CLEANUP:
            driver.on_local_cleanup(self)
            self.state = S.EXIT
            self._log("retiree LOCAL_CLEANUP->EXIT")
            driver.on_exit(self)


class ScaleDownStateMachineDriver:
    """Driver protocol: concrete side effects live in ModelRunner; FSM owns transitions."""

    def on_prepare(self, sm: ScaleDownStateMachine) -> None: ...
    def is_drained(self, sm: ScaleDownStateMachine) -> bool: ...
    def post_drain_barrier(self, sm: ScaleDownStateMachine) -> Optional[object]: ...
    def check_drain_barrier(self, handle: object) -> bool: ...
    def consume_drain_barrier(self, handle: object) -> None: ...
    def on_nixl_retire_pre(self, sm: ScaleDownStateMachine) -> None: ...
    def post_nixl_retire_barrier(self, sm: ScaleDownStateMachine) -> Optional[object]: ...
    def check_nixl_retire_barrier(self, handle: object) -> bool: ...
    def consume_nixl_retire_barrier(self, handle: object) -> None: ...
    def on_flip_mask(self, sm: ScaleDownStateMachine) -> None: ...
    def on_reconfig(self, sm: ScaleDownStateMachine) -> None: ...
    def on_local_cleanup(self, sm: ScaleDownStateMachine) -> None: ...
    def on_exit(self, sm: ScaleDownStateMachine) -> None: ...
