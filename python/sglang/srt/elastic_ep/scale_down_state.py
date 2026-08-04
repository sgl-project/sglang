"""Mooncake-native scale-down state machine.

Per-rank FSM driving retirement across scheduler ticks. Split into
explicit stages so retirees drain over multiple ticks, transient
failures retry, and observability tools can report per-tick progress.

    Survivor:  PREPARE -> DRAIN --barrier-- NIXL_RETIRE --barrier--
               FLIP_MASK -> RECONFIG -> COMPLETE
    Retiree:   PREPARE -> DRAIN --barrier-- NIXL_RETIRE --barrier--
               FLIP_MASK -> LOCAL_CLEANUP -> EXIT (os._exit 0)

Both cohort barriers (DRAIN and NIXL_RETIRE) are async because a
blocking WORLD wait would deadlock against mlp_sync (same PG). FLIP_MASK
is pure local work; the preceding barrier already excluded races.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

logger = logging.getLogger(__name__)


class ScaleDownSurvivorState(Enum):
    """Survivor lifecycle; NIXL_RETIRE drives the NIXL peer-disconnect
    + strict-sync store barrier across ticks (mirrors DRAIN)."""

    PREPARE = auto()
    DRAIN = auto()
    NIXL_RETIRE = auto()
    FLIP_MASK = auto()
    RECONFIG = auto()
    COMPLETE = auto()
    FAILED = auto()


class ScaleDownRetireeState(Enum):
    """Retiree lifecycle; terminal EXIT calls os._exit(0). NIXL_RETIRE
    prevents retirees from exiting mid-RDMA op on survivors."""

    PREPARE = auto()
    DRAIN = auto()
    NIXL_RETIRE = auto()
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
    # Async DRAIN barrier handle (typed as object; Work is not
    # available at import on every backend).
    _drain_barrier_handle: Optional[object] = None
    # Consecutive is_completed=False polls; falls through to wait()
    # after DRAIN_BARRIER_MAX_POLLS to work around Mooncake WORLD's
    # unreliable Work.is_completed.
    _drain_barrier_polls: int = 0
    # ~1-2s of wall clock at scheduler tick rate: long enough for a
    # healthy async barrier, short enough not to wedge the FSM.
    DRAIN_BARRIER_MAX_POLLS: int = 100

    # Async NIXL retire store-barrier handle. None after a legitimate
    # skip (non-NIXL, no TCPStore); use _nixl_barrier_posted below to
    # distinguish "posted-but-skipped" from "not yet posted".
    _nixl_barrier_handle: Optional[object] = None
    # Sticky flag: prevents re-post loop on non-NIXL backends.
    _nixl_barrier_posted: bool = False

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
            ScaleDownSurvivorState.DRAIN: ScaleDownSurvivorState.NIXL_RETIRE,
            ScaleDownSurvivorState.NIXL_RETIRE: ScaleDownSurvivorState.FLIP_MASK,
            ScaleDownSurvivorState.FLIP_MASK: ScaleDownSurvivorState.RECONFIG,
            ScaleDownSurvivorState.RECONFIG: ScaleDownSurvivorState.COMPLETE,
        }
        return transitions.get(self.survivor_state)

    def _advance_retiree(self) -> Optional[ScaleDownRetireeState]:
        transitions = {
            ScaleDownRetireeState.PREPARE: ScaleDownRetireeState.DRAIN,
            ScaleDownRetireeState.DRAIN: ScaleDownRetireeState.NIXL_RETIRE,
            ScaleDownRetireeState.NIXL_RETIRE: ScaleDownRetireeState.FLIP_MASK,
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
            self.survivor_state = ScaleDownSurvivorState.NIXL_RETIRE
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d survivor "
                "DRAIN->NIXL_RETIRE (consume=%.2fs)",
                self.my_global_rank,
                time.monotonic() - t0,
            )
            # Fold nixl_retire post+check+consume + FLIP_MASK into
            # this tick. Prevents an admission-gate advance while a
            # peer is still between DRAIN consume and mark_retiring
            # from wedging the peer's mlp_sync on WORLD. Bounded ~5s
            # catch-up busy-poll inside check_nixl_retire_barrier.
            self._nixl_barrier_handle = driver.post_nixl_retire_barrier(self)
            self._nixl_barrier_posted = True
            if not driver.check_nixl_retire_barrier(self._nixl_barrier_handle):
                return
            t1 = time.monotonic()
            driver.consume_nixl_retire_barrier(self._nixl_barrier_handle)
            driver.on_nixl_retire_pre(self)
            self._nixl_barrier_handle = None
            self._nixl_barrier_posted = False
            self.survivor_state = ScaleDownSurvivorState.FLIP_MASK
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d survivor "
                "NIXL_RETIRE->FLIP_MASK (consume=%.2fs) [folded]",
                self.my_global_rank,
                time.monotonic() - t1,
            )
            t2 = time.monotonic()
            driver.on_flip_mask(self)
            self.survivor_state = ScaleDownSurvivorState.RECONFIG
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d survivor "
                "FLIP_MASK->RECONFIG (try_retire_ranks took %.2fs) "
                "[folded]",
                self.my_global_rank,
                time.monotonic() - t2,
            )
            return

        if state is ScaleDownSurvivorState.NIXL_RETIRE:
            # Reached only when the DRAIN-fold check_nixl_retire_barrier
            # returned False -- the post already ran there and set
            # _nixl_barrier_posted=True. mark_retiring closes the
            # admission gate before returning so no get_next_batch
            # sees the mismatched (K-rank NIXL / N-rank expert map).
            if not driver.check_nixl_retire_barrier(self._nixl_barrier_handle):
                return  # wait for lagging cohort ranks to post
            t0 = time.monotonic()
            driver.consume_nixl_retire_barrier(self._nixl_barrier_handle)
            # NIXL peer disconnect. Cohort-aligned: every survivor
            # observes count >= world_size in the same tick.
            driver.on_nixl_retire_pre(self)
            self._nixl_barrier_handle = None
            self._nixl_barrier_posted = False
            self.survivor_state = ScaleDownSurvivorState.FLIP_MASK
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d survivor "
                "NIXL_RETIRE->FLIP_MASK (consume=%.2fs)",
                self.my_global_rank,
                time.monotonic() - t0,
            )
            # Fold FLIP_MASK into this tick: no event-loop iter runs
            # between the NIXL disconnect and mark_retiring, so no
            # dispatch sees the mismatched state.
            t1 = time.monotonic()
            driver.on_flip_mask(self)
            self.survivor_state = ScaleDownSurvivorState.RECONFIG
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d survivor "
                "FLIP_MASK->RECONFIG (try_retire_ranks took %.2fs)",
                self.my_global_rank,
                time.monotonic() - t1,
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
            self.retiree_state = ScaleDownRetireeState.NIXL_RETIRE
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d retiree "
                "DRAIN->NIXL_RETIRE (consume=%.2fs)",
                self.my_global_rank,
                time.monotonic() - t0,
            )
            # Symmetric fold with the survivor branch: post + check +
            # consume + FLIP_MASK in the same tick that consumed the
            # drain barrier. See the survivor's DRAIN branch for the
            # fold rationale.
            self._nixl_barrier_handle = driver.post_nixl_retire_barrier(self)
            self._nixl_barrier_posted = True
            if not driver.check_nixl_retire_barrier(self._nixl_barrier_handle):
                return
            t1 = time.monotonic()
            driver.consume_nixl_retire_barrier(self._nixl_barrier_handle)
            self._nixl_barrier_handle = None
            self._nixl_barrier_posted = False
            self.retiree_state = ScaleDownRetireeState.FLIP_MASK
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d retiree "
                "NIXL_RETIRE->FLIP_MASK (consume=%.2fs) [folded]",
                self.my_global_rank,
                time.monotonic() - t1,
            )
            t2 = time.monotonic()
            driver.on_flip_mask(self)
            self.retiree_state = ScaleDownRetireeState.LOCAL_CLEANUP
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d retiree "
                "FLIP_MASK->LOCAL_CLEANUP (try_retire_ranks took "
                "%.2fs) [folded]",
                self.my_global_rank,
                time.monotonic() - t2,
            )
            return

        if state is ScaleDownRetireeState.NIXL_RETIRE:
            # Reached only when the DRAIN-fold check_nixl_retire_barrier
            # returned False -- symmetric with the survivor path minus
            # the peer disconnect (that's the survivor's side).
            if not driver.check_nixl_retire_barrier(self._nixl_barrier_handle):
                return
            t0 = time.monotonic()
            driver.consume_nixl_retire_barrier(self._nixl_barrier_handle)
            self._nixl_barrier_handle = None
            self._nixl_barrier_posted = False
            self.retiree_state = ScaleDownRetireeState.FLIP_MASK
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d retiree "
                "NIXL_RETIRE->FLIP_MASK (consume=%.2fs)",
                self.my_global_rank,
                time.monotonic() - t0,
            )
            # Fold FLIP_MASK into this tick; keeps survivor + retiree
            # progression in lockstep.
            t1 = time.monotonic()
            driver.on_flip_mask(self)
            self.retiree_state = ScaleDownRetireeState.LOCAL_CLEANUP
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d retiree "
                "FLIP_MASK->LOCAL_CLEANUP (try_retire_ranks took %.2fs)",
                self.my_global_rank,
                time.monotonic() - t1,
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
        """Post the async retire barrier; None means "skip, advance
        immediately next tick"."""
        raise NotImplementedError

    def check_drain_barrier(self, handle: object) -> bool:
        """Non-blocking probe: has every cohort rank posted?"""
        raise NotImplementedError

    def consume_drain_barrier(self, handle: object) -> None:
        """Finalize (wait()) an already-completed barrier handle."""
        raise NotImplementedError

    def on_nixl_retire_pre(self, sm: ScaleDownStateMachine) -> None:
        """Survivor-only NIXL peer disconnect run once, folded into
        the NIXL_RETIRE consume tick just before mark_retiring so no
        run_batch sees asymmetric NIXL state. No-op for retirees /
        non-NIXL backends."""
        raise NotImplementedError

    def post_nixl_retire_barrier(
        self, sm: ScaleDownStateMachine
    ) -> Optional[object]:
        """Post the NIXL retire store barrier; None means skip."""
        raise NotImplementedError

    def check_nixl_retire_barrier(self, handle: object) -> bool:
        """Non-blocking probe: has every rank posted the NIXL barrier?"""
        raise NotImplementedError

    def consume_nixl_retire_barrier(self, handle: object) -> None:
        """Finalize the NIXL retire barrier (log only; no wait())."""
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
