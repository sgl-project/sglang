"""Mooncake-native scale-down state machine (per-rank FSM across scheduler ticks).

    Survivor:  PREPARE -> DRAIN --barrier-- NIXL_RETIRE --barrier-- FLIP_MASK -> RECONFIG -> COMPLETE
    Retiree:   PREPARE -> DRAIN --barrier-- NIXL_RETIRE --barrier-- FLIP_MASK -> LOCAL_CLEANUP -> EXIT

Both cohort barriers rendezvous over the global TCPStore and are polled across ticks, so
a rank waiting on one keeps servicing mlp_sync; a collective here would deadlock instead.
FLIP_MASK is local; the preceding barrier already excluded races.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class ScaleDownState(Enum):
    PREPARE = auto()
    DRAIN = auto()
    NIXL_RETIRE = auto()
    FLIP_MASK = auto()
    RECONFIG = auto()  # survivor-only
    COMPLETE = auto()  # survivor terminal
    LOCAL_CLEANUP = auto()  # retiree-only
    EXIT = auto()  # retiree terminal
    FAILED = auto()


_TERMINAL = {ScaleDownState.COMPLETE, ScaleDownState.EXIT, ScaleDownState.FAILED}


@dataclass
class ScaleDownStateMachine:
    """Per-rank scale-down driver owned by ModelRunner; torn down at terminal."""

    is_retiree: bool
    target_size: int
    effective_size: int
    ranks_to_retire: List[int]
    my_global_rank: int
    state: ScaleDownState = ScaleDownState.PREPARE
    last_error: Optional[str] = None
    _drain_barrier_handle: Optional[object] = None
    _drain_post_attempts: int = 0
    _nixl_barrier_handle: Optional[object] = None
    _nixl_post_attempts: int = 0
    # Bounded re-arms, then fail rather than flip the mask unsynchronized.
    BARRIER_POST_MAX_ATTEMPTS: int = 20

    def is_terminal(self) -> bool:
        return self.state in _TERMINAL

    def is_failed(self) -> bool:
        return self.state is ScaleDownState.FAILED

    def tick(self, driver: Any) -> None:
        """Advance by one stage per call. Side effects live in driver callbacks."""
        if self.is_terminal():
            return
        try:
            self._advance(driver)
            if self.state is ScaleDownState.EXIT:
                driver.on_exit(self)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[Elastic EP][scale-down FSM] rank=%d %s failed in %s",
                self.my_global_rank,
                "retiree" if self.is_retiree else "survivor",
                self.state.name.lower(),
                exc_info=True,
            )
            self.state = ScaleDownState.FAILED

    def _fold_nixl_to_flip(self, driver: Any) -> bool:
        """Same-tick NIXL barrier -> FLIP_MASK -> post-flip. Returns True if folded."""
        # Post once per cycle: re-posting double-counts the store arrival/ready
        # counters, mis-electing the leader and letting a subset satisfy the barrier.
        if self._nixl_barrier_handle is None:
            self._nixl_post_attempts += 1
            if self._nixl_post_attempts > self.BARRIER_POST_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"NIXL retire barrier failed to arm after "
                    f"{self._nixl_post_attempts - 1} attempts"
                )
            self._nixl_barrier_handle = driver.post_nixl_retire_barrier(self)
        if not driver.check_barrier(self._nixl_barrier_handle):
            return False
        driver.consume_barrier(self._nixl_barrier_handle)
        if not self.is_retiree:
            driver.on_nixl_retire_pre(self)
        self._nixl_barrier_handle = None
        self.state = ScaleDownState.FLIP_MASK
        driver.on_flip_mask(self)
        self.state = (
            ScaleDownState.LOCAL_CLEANUP if self.is_retiree else ScaleDownState.RECONFIG
        )
        return True

    def _advance(self, driver: Any) -> None:
        S = ScaleDownState
        if self.state is S.PREPARE:
            driver.on_prepare(self)
            self.state = S.DRAIN
            return

        if self.state is S.DRAIN:
            # Post once per cycle, for the same reason the NIXL barrier does.
            if self._drain_barrier_handle is None:
                self._drain_post_attempts += 1
                if self._drain_post_attempts > self.BARRIER_POST_MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"retire barrier failed to arm after "
                        f"{self._drain_post_attempts - 1} attempts"
                    )
                self._drain_barrier_handle = driver.post_drain_barrier(self)
                return
            if not driver.check_barrier(self._drain_barrier_handle):
                return
            driver.consume_barrier(self._drain_barrier_handle)
            self._drain_barrier_handle = None
            self.state = S.NIXL_RETIRE
            self._fold_nixl_to_flip(driver)
            return

        if self.state is S.NIXL_RETIRE:
            self._fold_nixl_to_flip(driver)
            return

        if self.state is S.RECONFIG:
            driver.on_reconfig(self)
            self.state = S.COMPLETE
            return

        if self.state is S.LOCAL_CLEANUP:
            driver.on_local_cleanup(self)
            self.state = S.EXIT
