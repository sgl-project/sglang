"""Mooncake-native scale-down state machine (per-rank FSM across scheduler ticks).

    Survivor:  PREPARE -> DRAIN --barrier-- NIXL_RETIRE --barrier-- FLIP_MASK -> RECONFIG -> COMPLETE
    Retiree:   PREPARE -> DRAIN --barrier-- NIXL_RETIRE --barrier-- FLIP_MASK -> LOCAL_CLEANUP -> EXIT

Both cohort barriers rendezvous over the global TCPStore and are polled across ticks, so
a rank waiting on one keeps servicing mlp_sync; a collective here would deadlock instead.
That polling is also why leaving DRAIN takes a second signal, carried by the mlp_sync
itself -- see elastic_ep.departure_pending(). FLIP_MASK is local; the preceding barrier
already excluded races.

DRAIN is a rendezvous that a retiree enters only once its own queues are empty, so the
barrier doubles as the drain: survivors cannot advance past a retiree still decoding.
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
    _drain_barrier_done: bool = False
    _nixl_barrier_handle: Optional[object] = None
    _nixl_post_attempts: int = 0
    # Bounded re-arms, then fail rather than flip the mask unsynchronized.
    BARRIER_POST_MAX_ATTEMPTS: int = 20
    # Under the barrier's own 300s timeout, so a genuinely absent peer is still
    # reported there rather than here.
    RETIREE_FOLD_BLOCK_S: float = 120.0

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
        # A retiree waits here in place rather than across ticks. It consumes the drain
        # barrier a moment before its survivors and so arrives first, and re-entering
        # the event loop to retry posts the control-plane collective against peers
        # already inside reconfig, which are no longer answering it. It has no work
        # left to service, so blocking costs nothing and removes the ordering hazard.
        if not driver.check_barrier(
            self._nixl_barrier_handle,
            block_s=self.RETIREE_FOLD_BLOCK_S if self.is_retiree else None,
        ):
            return False
        driver.consume_barrier(self._nixl_barrier_handle)
        if self.is_retiree:
            driver.on_retiree_quiesce(self)
        else:
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
            # Past the barrier, waiting for the cohort to agree on leaving together.
            # Still serving, which is what keeps the mlp_sync carrying that answer.
            if self._drain_barrier_done:
                if not driver.departure_cleared(self):
                    return
                self.state = S.NIXL_RETIRE
                # Gate the batch step here rather than at FLIP_MASK: every rank crossed
                # in this same iteration, so from now on the cohort posts no
                # serving-loop collective and the barriers left are store-only.
                driver.on_depart_drain(self)
                self._fold_nixl_to_flip(driver)
                return
            # Post once per cycle, for the same reason the NIXL barrier does.
            if self._drain_barrier_handle is None:
                # A retiree that arrives with work in flight strands it: its terminal
                # is sys.exit() and the tokenizer gate stops new admissions only.
                # Survivors keep serving while they wait here and that gate is shut,
                # so the queue only drains; the scale timeout refuses the shrink if it
                # never does. Ahead of the counter so a decode cannot burn the budget.
                if self.is_retiree and not driver.local_idle(self):
                    return
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
            self._drain_barrier_done = True
            driver.announce_departure(self)
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
