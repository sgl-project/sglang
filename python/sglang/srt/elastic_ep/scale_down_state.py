"""Mooncake-native scale-down state machine (per-rank FSM across scheduler ticks).

    Survivor:  PREPARE -> DRAIN --barrier-- NIXL_RETIRE --barrier-- FLIP_MASK -> RECONFIG -> COMPLETE
    Retiree:   PREPARE -> DRAIN --barrier-- NIXL_RETIRE --barrier-- FLIP_MASK -> LOCAL_CLEANUP -> EXIT

Both barriers rendezvous over the global TCPStore and poll across ticks, so a waiting
rank keeps servicing mlp_sync where a collective deadlocks. That polling is why leaving
DRAIN needs a second signal, carried by the mlp_sync (elastic_ep.departure_pending()).
A retiree enters DRAIN only once its queues are empty, so the barrier is the drain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional

from sglang.srt.elastic_ep.errors import ElasticLayoutFatal

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
    _departure_announced: bool = False
    _nixl_barrier_handle: Optional[object] = None
    _nixl_post_attempts: int = 0
    # Bounded re-arms, then fail rather than flip the mask unsynchronized.
    BARRIER_POST_MAX_ATTEMPTS: int = 20
    # Under the barrier's own 300s timeout, so an absent peer is reported there.
    RETIREE_FOLD_BLOCK_S: float = 120.0

    def is_terminal(self) -> bool:
        return self.state in _TERMINAL

    def abandon(self, driver) -> None:
        """Release any barrier this rank posted, before the machine is discarded.

        The store barriers are leader-based: whoever posted an epoch first owes the
        ARRIVAL reset. Dropping the machine with a handle outstanding leaves that epoch
        held, and the next cycle's first arrival then never reads 1, so every later
        scale waits on a barrier that can no longer arm. Idempotent."""
        for attr in ("_drain_barrier_handle", "_nixl_barrier_handle"):
            handle = getattr(self, attr)
            if handle is None:
                continue
            setattr(self, attr, None)
            try:
                driver.consume_barrier(handle)
            except Exception:
                logger.warning(
                    "[Elastic EP] releasing %s during teardown failed",
                    attr,
                    exc_info=True,
                )

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
        except ElasticLayoutFatal:
            # Ahead of the handler below, and not caught. FAILED leaves the rank
            # serving, which is the one thing a rank whose map and weights disagree
            # must not do: it answers from one expert's weights under another's label
            # and nothing downstream can tell. Reported as a failure this reads as a
            # scale that did not take, while the rank quietly returns wrong tokens.
            raise
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

    def _fold_nixl_to_flip(self, driver: Any) -> None:
        """Same-tick NIXL barrier -> FLIP_MASK -> post-flip."""
        # Post once per cycle: re-posting double-counts arrivals and lets a subset pass.
        if self._nixl_barrier_handle is None:
            self._nixl_post_attempts += 1
            if self._nixl_post_attempts > self.BARRIER_POST_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"NIXL retire barrier failed to arm after "
                    f"{self._nixl_post_attempts - 1} attempts"
                )
            self._nixl_barrier_handle = driver.post_nixl_retire_barrier(self)
            # A retiree blocks rather than polls: it clears the drain barrier first,
            # so re-ticking would post a control-plane collective against peers
            # already in reconfig. It has no work left to lose.
        if not driver.check_barrier(
            self._nixl_barrier_handle,
            block_s=self.RETIREE_FOLD_BLOCK_S if self.is_retiree else None,
        ):
            return
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

    def _advance(self, driver: Any) -> None:
        S = ScaleDownState
        if self.state is S.PREPARE:
            driver.on_prepare(self)
            self.state = S.DRAIN
            return

        if self.state is S.DRAIN:
            # Past the barrier, awaiting cohort agreement; serving keeps mlp_sync alive.
            if self._drain_barrier_done:
                if not self._departure_announced:
                    # Idle is re-checked here, not just before the barrier post: this
                    # rank keeps serving across the barrier wait, so work admitted in
                    # that window would otherwise ride along to sys.exit(). Holding
                    # here simply stays in DRAIN, and an unannounced retiree keeps
                    # contributing 1 to the mlp_sync, so the cohort waits with it.
                    if self.is_retiree and not driver.local_idle(self):
                        return
                    driver.announce_departure(self)
                    self._departure_announced = True
                if not driver.departure_cleared(self):
                    return
                self.state = S.NIXL_RETIRE
                # Gate the batch step here, not at FLIP_MASK: all ranks crossed this
                # tick, so no serving-loop collective remains, only store barriers.
                driver.on_depart_drain(self)
                self._fold_nixl_to_flip(driver)
                return
            # Post once per cycle, for the same reason the NIXL barrier does.
            if self._drain_barrier_handle is None:
                # A retiree exiting with work in flight strands it: its terminal is
                # sys.exit(). Ahead of the counter, so decodes burn no attempts.
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
            # Probe, never wait: every peer here is still serving, so waiting
            # stalls cohort decode for the window. The NIXL fold does wait --
            # there peers have gated, and the same-tick crossing needs it.
            if not driver.check_barrier(self._drain_barrier_handle, keep_serving=True):
                return
            driver.consume_barrier(self._drain_barrier_handle)
            self._drain_barrier_handle = None
            self._drain_barrier_done = True
            # Departure is announced from the branch above, after a fresh idle check.
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
