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
    │    DRAIN ────────────┼── barrier ───┤    DRAIN             │
    │       │              │              │       │              │
    │    NIXL_RETIRE ──────┼── barrier ───┤    NIXL_RETIRE       │
    │       │              │              │       │              │
    │    FLIP_MASK         │              │    FLIP_MASK         │
    │       │              │              │       │              │
    │    RECONFIG          │              │    LOCAL_CLEANUP     │
    │       │              │              │       │              │
    │    COMPLETE          │              │    EXIT (sys.exit 0) │
    └──────────────────────┘              └──────────────────────┘

Two cohort-wide barriers punctuate the flow:

  * ``DRAIN``       -- everyone stops accepting new work. Uses the async
    :func:`retire_barrier_post` / ``check`` / ``consume`` primitives on
    WORLD; blocking would deadlock against ``mlp_sync`` because both
    are WORLD collectives and different ranks receive the scale request
    one event-loop iteration apart.
  * ``NIXL_RETIRE`` -- survivors have finished ``NixlEPBuffer.on_retire``
    and every rank has entered the retire boundary. Uses the async
    TCP-store primitives :func:`nixl_retire_barrier_post` /
    ``check`` / ``consume``; blocking here re-introduces the same
    deadlock: a slow joiner (e.g. mid-DeepGEMM-JIT when the shrink
    kicks off) keeps survivors parked in the blocking store wait,
    which starves ``mlp_sync`` on cohort ranks that are still
    processing the request, which in turn prevents the joiner from
    reaching the barrier itself -- circular dependency, 300s timeout.

``FLIP_MASK`` is now pure local work: every rank writes 0 into the
Mooncake ``active_ranks`` mask for the retirees; the NIXL_RETIRE
barrier already guaranteed no survivor is mid-collective against a
retiree. Any subsequent WORLD collective is mask-honored and skips
the retirees, who then proceed to LOCAL_CLEANUP -> ``sys.exit(0)``.
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
    condition (see :meth:`ScaleDownStateMachine.tick`) is satisfied.

    ``NIXL_RETIRE`` sits between DRAIN and FLIP_MASK so the NIXL a2a
    survivor-side peer disconnect + strict-sync store barrier drives
    across scheduler ticks (mirrors the async DRAIN barrier pattern).
    Making this a distinct FSM state -- rather than a blocking call
    inside FLIP_MASK -- avoids deadlocking against ``mlp_sync`` on
    cohort ranks that haven't yet reached the barrier. See
    :func:`sglang.srt.elastic_ep.elastic_ep.nixl_retire_barrier_post`
    for the full trace."""

    PREPARE = auto()
    DRAIN = auto()
    NIXL_RETIRE = auto()
    FLIP_MASK = auto()
    RECONFIG = auto()
    COMPLETE = auto()
    FAILED = auto()


class ScaleDownRetireeState(Enum):
    """Retiree-side lifecycle. Terminal state is :attr:`EXIT`, at which
    point :func:`ModelRunner._retire_and_exit` calls ``sys.exit(0)`` and
    never returns.

    ``NIXL_RETIRE`` mirrors the survivor state so retirees post the
    same store barrier and don't leave until every survivor has
    completed its NIXL peer disconnect (invariant that prevents
    retiree-side sys.exit from racing an in-flight RDMA op)."""

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

    # Async handle for the NIXL retire store barrier. Posted on entry
    # into ``NIXL_RETIRE`` and polled each subsequent tick via
    # :meth:`ScaleDownStateMachineDriver.check_nixl_retire_barrier`
    # until every cohort rank has entered the barrier. ``None`` after
    # a legitimate skip (non-NIXL backend, no global TCPStore, torch.
    # distributed not initialized) -- ``_nixl_barrier_posted`` below
    # then distinguishes "posted-but-skipped" from "not yet posted",
    # so the FSM does not re-run ``on_nixl_retire_pre`` +
    # ``post_nixl_retire_barrier`` on every tick.
    _nixl_barrier_handle: Optional[object] = None
    # Sticky flag flipped to ``True`` after
    # :meth:`ScaleDownStateMachineDriver.post_nixl_retire_barrier` has
    # been called once for this FSM run, regardless of whether the
    # driver returned a handle or ``None``. Guards against the
    # non-NIXL infinite re-post loop that a bare ``handle is None``
    # check would fall into.
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
            # FOLD nixl_retire_barrier post + check + consume +
            # FLIP_MASK into the same tick body. Rationale: any FSM
            # transition where one rank advances (closing the
            # admission gate) while a peer is still between drain
            # consume and the folded ``mark_retiring`` risks wedging
            # the peer's ``mlp_sync`` all-gather on WORLD. Folding
            # here (with a bounded ~5s catch-up busy-poll inside
            # ``check_nixl_retire_barrier``, see ``elastic_ep.py``)
            # ensures every rank leaves this single tick with either
            # FLIP_MASK completed or the barrier state preserved for
            # a delayed check in the NIXL_RETIRE branch below.
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
            # Three-step barrier-mediated handshake across scheduler ticks:
            #
            #   T4a: post the async retire store barrier. NO peer
            #        disconnect yet -- deferred until the cohort-wide
            #        sync point at T4c so every rank flips its NIXL
            #        state simultaneously.
            #   T4b: poll barrier. Every rank stays in the "draining"
            #        phase so ``mlp_sync`` keeps making forward progress
            #        across the cohort.
            #   T4c: barrier consume + NIXL peer disconnect + fold the
            #        FLIP_MASK body (``mark_retiring`` +
            #        ``try_retire_ranks``) into this same tick,
            #        transitioning straight to RECONFIG. Folding
            #        FLIP_MASK closes the ``nixl_ep_ll.cu:178``
            #        ``dst_expert_idx < active_expert_bound`` race
            #        without introducing a separate
            #        ``"nixl_retiring"`` gate phase: ``mark_retiring``
            #        (the admission gate that always fenced the
            #        FLIP_MASK -> RECONFIG window) closes immediately
            #        after ``_pre_nixl_retire`` mutates NIXL, before
            #        control returns to the event loop, so no
            #        ``get_next_batch_to_run`` sees the mismatched
            #        (K-rank NIXL / N-rank expert map) state.
            #
            # The ``_nixl_barrier_posted`` flag (not the handle) gates
            # re-posting: on Mooncake a2a the driver legitimately returns
            # ``None`` and we still must not re-post every tick.
            if not self._nixl_barrier_posted:
                self._nixl_barrier_handle = driver.post_nixl_retire_barrier(self)
                self._nixl_barrier_posted = True
                return
            if not driver.check_nixl_retire_barrier(self._nixl_barrier_handle):
                return  # wait for lagging cohort ranks to post
            t0 = time.monotonic()
            driver.consume_nixl_retire_barrier(self._nixl_barrier_handle)
            # NIXL peer disconnect. Aligned across the cohort by the
            # barrier consume -- every survivor observes
            # ``count >= world_size`` on the shared TCPStore in the
            # same scheduler iteration and disconnects together.
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
            # Fold FLIP_MASK body into the same tick body. NO event-loop
            # iter runs between ``on_nixl_retire_pre`` (which drops
            # :attr:`NixlEPBuffer._dispatch_ep_size` from ``N`` to
            # ``K``) and ``on_flip_mask`` (which flips
            # ``active_ranks[retiree]=0`` and marks phase ``retiring``,
            # closing the scheduler admission gate). This closes the
            # ``nixl_ep_ll.cu:178`` assertion race without introducing
            # a separate ``"nixl_retiring"`` phase.
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

        if state is ScaleDownSurvivorState.FLIP_MASK:
            # Reachable only if the folded ``on_flip_mask`` in the
            # NIXL_RETIRE branch raised after the FLIP_MASK state
            # transition. Left as a safety net -- the normal path
            # folds this body into the NIXL_RETIRE consume tick above.
            t0 = time.monotonic()
            driver.on_flip_mask(self)
            self.survivor_state = ScaleDownSurvivorState.RECONFIG
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d survivor "
                "FLIP_MASK->RECONFIG (try_retire_ranks took %.2fs) "
                "[fallback path]",
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
            # Retirees post the same store barrier and don't leave
            # until every survivor has entered it (invariant: no
            # survivor still holds a live NIXL peer state targeting
            # this retiree). Retiree does NOT call on_nixl_retire_pre
            # -- that's the survivor's peer-disconnect side of the
            # handshake. Symmetric with the survivor branch: post
            # first, poll, then fold FLIP_MASK into the barrier consume
            # tick body so retirees advance to LOCAL_CLEANUP in the
            # same scheduler iter that survivors advance to RECONFIG.
            # Uses ``_nixl_barrier_posted`` (not the handle) to gate
            # re-posting so Mooncake a2a retirees (driver returns
            # ``None``) skip the state in exactly two ticks instead
            # of infinite-looping.
            if not self._nixl_barrier_posted:
                self._nixl_barrier_handle = driver.post_nixl_retire_barrier(self)
                self._nixl_barrier_posted = True
                return
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
            # Fold FLIP_MASK body into the same tick body -- symmetric
            # with the survivor branch above. Keeps survivor + retiree
            # FLIP_MASK progression in lockstep at cohort scale.
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

        if state is ScaleDownRetireeState.FLIP_MASK:
            # Reachable only if the folded ``on_flip_mask`` in the
            # NIXL_RETIRE branch raised after the FLIP_MASK state
            # transition. Safety net -- the normal path folds this
            # body into NIXL_RETIRE consume.
            t0 = time.monotonic()
            driver.on_flip_mask(self)
            self.retiree_state = ScaleDownRetireeState.LOCAL_CLEANUP
            logger.info(
                "[Elastic EP][scale-down FSM] rank=%d retiree "
                "FLIP_MASK->LOCAL_CLEANUP (try_retire_ranks took %.2fs) "
                "[fallback path]",
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

    def on_nixl_retire_pre(self, sm: ScaleDownStateMachine) -> None:
        """Survivor-only: run NIXL peer disconnect on the FSM tick that
        CONSUMES the NIXL retire store barrier. Called exactly once
        per FSM run, immediately BEFORE the folded ``on_flip_mask``
        body runs in the same tick (see the ``NIXL_RETIRE`` branch of
        :meth:`_tick_survivor`). Aligned across the cohort by the
        barrier consume so every survivor drops
        ``NixlEPBuffer._dispatch_ep_size`` from ``N`` to ``K`` in the
        same scheduler iteration; no run_batch elsewhere in the cohort
        observes an asymmetric NIXL peer state. The
        ``nixl_ep_ll.cu:178`` ``dst_expert_idx < active_expert_bound``
        assertion is closed by folding FLIP_MASK's ``mark_retiring``
        into the same FSM tick body -- no scheduler event-loop iter
        runs between this call and the gate close, so a raced
        ``get_next_batch_to_run`` cannot admit a batch with N-rank
        ``topk_ids`` while NIXL is already K-rank. Retiree ticks skip
        this hook. No-op on non-NIXL backends."""
        raise NotImplementedError

    def post_nixl_retire_barrier(
        self, sm: ScaleDownStateMachine
    ) -> Optional[object]:
        """Post the NIXL retire store barrier and return an opaque
        handle. Every rank (survivors + retirees) calls this exactly
        once per FSM run. Returning ``None`` (e.g. non-NIXL backend,
        no global TCPStore) tells the FSM to advance immediately on
        the next tick."""
        raise NotImplementedError

    def check_nixl_retire_barrier(self, handle: object) -> bool:
        """Non-blocking probe: has every cohort rank posted the NIXL
        retire barrier yet? Returning ``True`` means the FSM should
        call :meth:`consume_nixl_retire_barrier` and transition to
        FLIP_MASK."""
        raise NotImplementedError

    def consume_nixl_retire_barrier(self, handle: object) -> None:
        """Finalize the NIXL retire barrier (log record; no wait())."""
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
