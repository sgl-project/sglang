from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional

import torch

from sglang.srt.distributed import get_world_group, parallel_state
from sglang.srt.distributed.utils import get_global_tcp_store
from sglang.srt.eplb.expert_location import broadcast_global_expert_location_metadata
from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cpu, is_cuda

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.eplb.eplb_manager import EPLBManager

logger = logging.getLogger(__name__)

_SCALE_COHORT_KEY_PREFIX = "elastic_ep/scale_cohort"


def register_scale_cohort(rank_offset: int, target_ep_size: int) -> None:
    store = get_global_tcp_store()
    if store is None:
        raise RuntimeError("Elastic EP scale-up requires the global TCPStore.")
    store.set(f"{_SCALE_COHORT_KEY_PREFIX}/{rank_offset}", str(target_ep_size).encode())


def get_scale_cohort_target(rank_offset: int) -> Optional[int]:
    store = get_global_tcp_store()
    if store is None:
        return None
    key = f"{_SCALE_COHORT_KEY_PREFIX}/{rank_offset}"
    if not store.check([key]):
        return None
    return int(store.get(key).decode())


@dataclass
class ElasticEPState:
    active_ranks: Optional[torch.Tensor]
    last_active_ranks: Optional[torch.Tensor]
    active_ranks_cpu: Optional[torch.Tensor]
    effective_ep_size: int = 0
    pending_ep_size: Optional[int] = None
    scale_phase: str = "idle"
    last_error: Optional[str] = None
    pending_since: Optional[float] = None
    original_ep_size: int = 0
    has_scaled: bool = False
    ep_join_rank_offset: int = 0
    # Global ranks whose Mooncake ``active_ranks`` slot must be flipped
    # 0->1 during the pending grow (grow-into-retired-slot). Empty
    # when the pending grow is an append-only scale-up.
    pending_recover_ranks: List[int] = field(default_factory=list)

    def is_active_equal_last(self) -> bool:
        return torch.equal(self.active_ranks, self.last_active_ranks)

    def sync_active_to_cpu(self):
        if self.active_ranks is not None:
            self.active_ranks_cpu = self.active_ranks.detach().cpu().clone()

    def snapshot_active_to_last(self):
        if self.active_ranks is not None:
            self.last_active_ranks = self.active_ranks.clone()

    def reset(self):
        if self.active_ranks is not None:
            # Reserved slots stay inactive until their ranks join.
            self.active_ranks.zero_()
            self.active_ranks[: self.effective_ep_size] = 1
            self.snapshot_active_to_last()
            self.sync_active_to_cpu()

    def _set_active_bits(self, global_ranks: List[int], value: int) -> None:
        if self.active_ranks is None:
            return
        numel = self.active_ranks.numel()
        for global_rank in global_ranks:
            if 0 <= global_rank < numel:
                self.active_ranks[global_rank] = value
        self.snapshot_active_to_last()
        self.sync_active_to_cpu()

    def activate_ranks(self, global_ranks: List[int]) -> None:
        """Flip ``active_ranks[g] = 1`` for each g in ``global_ranks``,
        then update the pinned CPU snapshot + last-known-good mirror.

        Shared entrypoint for every code path that grows the live
        cohort: append-only scale-up
        (:meth:`ModelRunner._finalize_scale_up`), recover-mode grow
        (:meth:`ModelRunner._finalize_scale_recover`), and Mooncake-
        native rejoin (:func:`try_recover_ranks`). Out-of-range ranks
        are silently ignored (mirrors the pre-refactor bounds guards).
        """
        self._set_active_bits(global_ranks, 1)

    def deactivate_ranks(self, global_ranks: List[int]) -> None:
        """Flip ``active_ranks[g] = 0`` for each g in ``global_ranks``.

        Shared entrypoint for :func:`try_retire_ranks` and any other
        retirement/eviction path. Out-of-range ranks are silently
        ignored.
        """
        self._set_active_bits(global_ranks, 0)


class ElasticEPStateManager:
    _instance: Optional[ElasticEPState] = None
    _on_scale: Optional[Callable[[int, int], None]] = None

    @classmethod
    def instance(cls) -> ElasticEPState:
        return cls._instance

    @classmethod
    def init(cls, server_args: ServerArgs):
        if cls._instance is not None:
            return cls._instance

        if server_args.elastic_ep_backend is not None:
            world_size = torch.distributed.get_world_size()
            active_rank_capacity = server_args.max_ep_size or world_size
            assert active_rank_capacity >= world_size, (
                f"--max-ep-size ({active_rank_capacity}) must be >= "
                f"world_size ({world_size})."
            )

            inst = cls._build_state(ep_size=active_rank_capacity, device=None)
            inst.effective_ep_size = world_size
            inst.original_ep_size = world_size
            if active_rank_capacity > world_size:
                inst.active_ranks[world_size:].zero_()
                inst.snapshot_active_to_last()
                inst.sync_active_to_cpu()

            if server_args.moe_a2a_backend == "nixl":
                cls._on_scale = cls._on_scale_nixl

            inst.ep_join_rank_offset = server_args.ep_join_rank_offset
            if server_args.is_ep_joiner:
                cls._init_joiner_state(inst, server_args)

            cls._instance = inst

        return cls._instance

    @classmethod
    def _init_joiner_state(cls, inst: ElasticEPState, server_args: ServerArgs) -> None:
        global_rank = torch.distributed.get_rank()
        inst.active_ranks.zero_()
        inst.active_ranks[global_rank] = 1
        inst.snapshot_active_to_last()
        inst.sync_active_to_cpu()

        if server_args.ep_join_mode == "scale":
            inst.effective_ep_size = (
                server_args.ep_join_rank_offset + server_args.tp_size
            )
            inst.original_ep_size = (
                server_args.elastic_ep_initial_size or server_args.ep_join_rank_offset
            )
            inst.has_scaled = True
        else:
            world_size = torch.distributed.get_world_size()
            inst.effective_ep_size = world_size
            inst.original_ep_size = world_size

    @staticmethod
    def _select_device() -> torch.device:
        if is_cuda():
            return torch.device("cuda")
        elif is_cpu():
            return torch.device("cpu")
        else:
            raise NotImplementedError("Only CUDA and CPU support elastic ep now.")

    @classmethod
    def _build_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> ElasticEPState:
        active = cls.healthy_rank_state(ep_size=ep_size, device=device)
        return ElasticEPState(
            active_ranks=active,
            last_active_ranks=active.clone(),
            active_ranks_cpu=active.detach().cpu().clone(),
        )

    @classmethod
    def healthy_rank_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = ep_size if ep_size is not None else torch.distributed.get_world_size()
        dev = device if device is not None else cls._select_device()

        return torch.ones(size, dtype=torch.int32, device=dev)

    @classmethod
    def request_scale(cls, n: int) -> bool:
        inst = cls._instance
        if inst is None:
            return False
        if (
            inst.pending_ep_size is not None
            or inst.scale_phase == "recovery_unsupported"
        ):
            return False
        # Reject invalid targets synchronously so the tokenizer's
        # admission gate can 4xx to the operator instead of stalling
        # for the outer ``elastic_ep_scale_timeout`` (~600s) on an
        # unreachable target. ``active_ranks_cpu`` is sized to the
        # configured ``--max-ep-size`` capacity at :meth:`init`, so
        # its element count is the max cohort size this deployment
        # can grow to.
        if n <= 0:
            logger.warning(
                "[Elastic EP] request_scale rejected: target ep_size=%d must be > 0",
                n,
            )
            return False
        if n == inst.effective_ep_size:
            logger.info(
                "[Elastic EP] request_scale rejected: target ep_size=%d is a "
                "no-op (already the effective cohort size)",
                n,
            )
            return False
        max_cap = (
            inst.active_ranks_cpu.numel()
            if inst.active_ranks_cpu is not None
            else None
        )
        if max_cap is not None and n > max_cap:
            logger.warning(
                "[Elastic EP] request_scale rejected: target ep_size=%d exceeds "
                "max_ep_size capacity=%d",
                n,
                max_cap,
            )
            return False
        inst.pending_ep_size = n
        inst.scale_phase = "waiting_for_cohort"
        inst.last_error = None
        inst.pending_since = time.monotonic()
        return True

    @classmethod
    def begin_scale(cls) -> bool:
        inst = cls._instance
        if (
            inst is None
            or inst.pending_ep_size is None
            or inst.scale_phase != "waiting_for_cohort"
        ):
            return False
        inst.scale_phase = "pending"
        return True

    @classmethod
    def mark_joining(cls) -> None:
        cls._mark_phase("joining")

    @classmethod
    def mark_configuring_data_plane(cls) -> None:
        cls._mark_phase("configuring_data_plane")

    @classmethod
    def mark_syncing_new_world(cls) -> None:
        cls._mark_phase("syncing_new_world")

    @classmethod
    def mark_draining(cls) -> None:
        """Enter the drain phase for a pending scale-DOWN.

        Survivors + retirees pause admission of new work and wait for
        in-flight batches to complete before the mask flip.
        """
        cls._mark_phase("draining")

    @classmethod
    def mark_retiring(cls) -> None:
        """Enter the retire phase for a pending scale-DOWN.

        Survivors flip ``active_ranks[retiree]=0`` on every launch-time PG;
        retirees run local Mooncake quiesce and ``sys.exit(0)``.
        """
        cls._mark_phase("retiring")

    @classmethod
    def mark_reconfiguring(cls) -> None:
        """Enter the reconfig phase for a pending scale-DOWN.

        Survivors rebuild MoE dispatcher / dp_attention / expert-location
        for the shrunk K-rank cohort. Symmetric with grow's
        ``mark_configuring_data_plane``; kept as a separate phase name so
        observers can distinguish shrink from grow direction.
        """
        cls._mark_phase("reconfiguring")

    @classmethod
    def _mark_phase(cls, phase: str) -> None:
        inst = cls._instance
        if inst is not None and inst.pending_ep_size is not None:
            inst.scale_phase = phase

    @classmethod
    def get_shrink_direction(cls) -> Optional[str]:
        """Return ``"shrink"`` / ``"grow"`` / ``None`` for the pending op.

        Determined by comparing ``pending_ep_size`` against ``effective_ep_size``.
        Returns ``None`` if no scale is pending.
        """
        inst = cls._instance
        if inst is None or inst.pending_ep_size is None:
            return None
        if inst.pending_ep_size < inst.effective_ep_size:
            return "shrink"
        if inst.pending_ep_size > inst.effective_ep_size:
            return "grow"
        return None

    @classmethod
    def get_pending_shrink_ranks(cls) -> List[int]:
        """Return the global-rank list being retired by the pending shrink.

        Retirees are the highest-numbered ranks in the current cohort:
        ``[pending_ep_size, ..., effective_ep_size - 1]``. The retirement
        scope is contiguous whole-TP-group suffixes. Returns an empty
        list if no shrink is pending.
        """
        inst = cls._instance
        if inst is None or inst.pending_ep_size is None:
            return []
        if inst.pending_ep_size >= inst.effective_ep_size:
            return []
        return list(range(inst.pending_ep_size, inst.effective_ep_size))

    @classmethod
    def commit_scale(cls) -> None:
        inst = cls._instance
        if inst is None or inst.pending_ep_size is None:
            return
        direction = cls.get_shrink_direction()
        inst.effective_ep_size = inst.pending_ep_size
        inst.pending_ep_size = None
        inst.has_scaled = True
        inst.scale_phase = "serving_shrunk" if direction == "shrink" else "serving_expanded"
        inst.last_error = None
        inst.pending_since = None
        inst.pending_recover_ranks = []
        inst.reset()

    @classmethod
    def fail_scale(cls, error: str) -> None:
        inst = cls._instance
        if inst is None:
            return
        inst.pending_ep_size = None
        inst.scale_phase = "failed"
        inst.last_error = error
        inst.pending_since = None
        inst.pending_recover_ranks = []
        inst.reset()

    @classmethod
    def fail_recovery(cls, error: str) -> None:
        inst = cls._instance
        if inst is None:
            return
        inst.scale_phase = "recovery_unsupported"
        inst.last_error = error

    @classmethod
    def set_pending_recover_ranks(cls, ranks: List[int]) -> None:
        """Record the global ranks that a pending grow must reactivate."""
        inst = cls._instance
        if inst is None:
            return
        inst.pending_recover_ranks = list(ranks)

    @classmethod
    def get_pending_recover_ranks(cls) -> List[int]:
        """Return the global ranks that the pending grow must reactivate."""
        inst = cls._instance
        if inst is None:
            return []
        return list(inst.pending_recover_ranks)

    @classmethod
    def get_effective_ep_size(cls) -> int:
        inst = cls._instance
        assert inst is not None, "Elastic EP state is not initialized."
        return inst.effective_ep_size

    @classmethod
    def get_pending_ep_size(cls) -> Optional[int]:
        inst = cls._instance
        if inst is None:
            return None
        return inst.pending_ep_size

    @classmethod
    def get_scale_phase(cls) -> str:
        inst = cls._instance
        if inst is None:
            return "disabled"
        return inst.scale_phase

    @classmethod
    def get_last_error(cls) -> Optional[str]:
        inst = cls._instance
        if inst is None:
            return None
        return inst.last_error

    @classmethod
    def get_ep_join_rank_offset(cls) -> int:
        inst = cls._instance
        if inst is None:
            return 0
        return inst.ep_join_rank_offset

    @classmethod
    def on_scale(cls, from_ep_size: int, to_ep_size: int) -> None:
        if cls._on_scale is not None:
            cls._on_scale(from_ep_size, to_ep_size)

    @staticmethod
    def _on_scale_nixl(from_ep_size: int, to_ep_size: int) -> None:
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

        NixlEPBuffer.on_scale(from_ep_size, to_ep_size)

    @classmethod
    def is_scaling(cls) -> bool:
        """Return whether a scale or recovery operation is pending.

        The CPU snapshot is authoritative because rank polling uses it too.
        """
        inst = cls._instance
        if inst is None or inst.active_ranks_cpu is None:
            return False
        if inst.scale_phase == "recovery_unsupported":
            return False
        if inst.pending_ep_size is not None:
            return True
        active_count = int(inst.active_ranks_cpu[: inst.effective_ep_size].sum().item())
        return active_count < inst.effective_ep_size


def elastic_expanded_world_enabled() -> bool:
    """Return whether execution uses ranks admitted after server launch.

    Launch-time TP groups exclude ranks admitted during scale-up.
    """
    inst = ElasticEPStateManager.instance()
    if inst is None:
        return False
    if get_parallel().max_ep_size is None:
        return False
    active_target_size = inst.effective_ep_size
    if inst.pending_ep_size is not None and inst.scale_phase in (
        "configuring_data_plane",
        "syncing_new_world",
    ):
        active_target_size = inst.pending_ep_size

    return active_target_size > inst.original_ep_size


def _refresh_ep_members() -> None:
    from sglang.srt.layers.moe.token_dispatcher.mooncake import EPBuffer

    buffer = EPBuffer.get_existing_buffer()
    if buffer is not None:
        buffer.update_ep_member()


_NIXL_RETIRE_STORE_BARRIER_POLL_INTERVAL_S = 0.05
_NIXL_RETIRE_STORE_BARRIER_TIMEOUT_S = 300.0
# Bounded catch-up busy-poll inside ``nixl_retire_barrier_check`` --
# see that function's docstring for the fold rationale.
#
# Two windows: ``_FIRST`` bounds the fold check (called in the same tick
# as ``post``) where peer ranks are typically <100ms behind and we
# must not surrender the tick to the event loop. ``_FALLBACK`` bounds
# subsequent NIXL_RETIRE fallback checks -- the FSM re-ticks quickly,
# so a shorter poll is enough to catch racing counter updates without
# starving the scheduler for seconds at a time on stuck cohorts. The
# 200ms fallback matches the typical FSM tick cadence, so a counter
# update in-flight during a fallback check is observed within one tick.
_NIXL_RETIRE_STORE_BARRIER_CATCH_UP_FIRST_S = 5.0
_NIXL_RETIRE_STORE_BARRIER_CATCH_UP_FALLBACK_S = 0.2
# Bounded busy-poll for the cycle-id inside ``nixl_retire_barrier_post``.
# A non-first-arriver waits up to this long for the elected cycle
# leader to publish the cycle id before falling back to a legacy
# derivation.
_NIXL_RETIRE_CYCLE_ID_CATCH_UP_S = 5.0
_NIXL_RETIRE_ARRIVAL_COUNTER_KEY = "sglang_nixl_retire_arrival_counter"


def _barrier_target_from_effective_ep(
    rank: int, world_size: int, log_prefix: str
) -> int:
    """Retire-barrier target: number of ranks that will actually post.

    This is the ``effective_ep_size`` at the retire boundary (the full
    cohort BEFORE the impending retirees flip their mask), not the
    launch ``world_size``. For the common case (shrink from the full
    launch cohort or from a fully-recovered cohort) they are equal,
    but for a chained shrink (e.g. ``4 -> 3 -> 2``) the second shrink
    only has 3 live ranks -- waiting for ``world_size == 4`` posts
    would deadlock the survivors for the full 300s barrier timeout.

    Reading ``effective_ep_size`` is safe here because
    :meth:`ElasticEPStateManager.commit_scale` (which lowers it to the
    post-shrink target) runs strictly AFTER the barrier releases; the
    read at post time returns the pre-shrink cohort size.

    Called by both :func:`nixl_retire_barrier_post` and
    :func:`retire_barrier_post`; kept as a shared helper to keep the
    two barriers' cohort-size derivation in lockstep.
    """
    try:
        effective = ElasticEPStateManager.get_effective_ep_size()
        if effective > 0:
            return effective
    except Exception as exc:
        logger.debug(
            "%s rank=%d get_effective_ep_size failed (%s); falling "
            "back to world_size=%d for retire barrier target",
            log_prefix,
            rank,
            exc,
            world_size,
        )
    return world_size


def _rollback_arrival_counter(store, rank: int, key: str) -> None:
    """Reset the shared ARRIVAL_KEY back to 0 after a failed leader.

    Used by the elected-leader epoch derivation when the leader (arrival
    == 1) increments ARRIVAL_KEY but then fails on the CYCLE_KEY write.
    Without this rollback the counter stays permanently elevated and no
    future ``store.add`` returns 1: every rank becomes a follower on the
    next cycle, every rank times out in the catch-up poll, and the
    barrier is silently disabled for the rest of the deployment.

    Only the elected leader for a cycle may call this. Followers observe
    the counter already >= 2 and cannot safely reset without racing a
    peer that has already moved on to the next cycle.
    """
    try:
        store.set(key, "0")
    except Exception as exc:
        logger.debug(
            "[Elastic EP][retire] rank=%d (leader) ARRIVAL_KEY rollback "
            "on %s failed (%s); next cycle may busy-poll before the "
            "counter drains",
            rank,
            key,
            exc,
        )
# Cohort-wide monotonic cycle counter maintained on the shared
# TCPStore. Bumped exactly once per retire boundary by the elected
# first-arriver; consumed by every other rank in the same boundary
# to derive a cohort-consistent ``epoch``. Replaces the previous
# ``epoch = (arrival - 1) // world_size + 1`` derivation, which
# silently split cohort ranks into different epoch buckets on any
# retire boundary whose poster count did not equal ``world_size``
# (e.g. a 3-step chained shrink like ``4 -> 3 -> 2 -> 1``, where
# cycle 3's two arrivals land at counter 8/9 and straddle the
# ``world_size = 4`` bucket boundary).
_NIXL_RETIRE_CYCLE_KEY = "sglang_nixl_retire_cycle_counter"

# Per-process record of the most recent NIXL retire cycle id this
# rank has consumed. Non-first arrivers busy-poll the shared
# ``_NIXL_RETIRE_CYCLE_KEY`` on the TCPStore until they observe a
# value strictly greater than this. Rank-local because it survives
# retiree exits: a joiner subprocess spawned mid-flight starts at
# 0, which is correct (its first observed cycle is whatever the
# incumbents advertise).
_last_local_nixl_retire_cycle_id: int = 0


@dataclass
class _NixlRetireBarrierState:
    """Handle for the async NIXL retire store barrier.

    Mirrors :class:`_RetireBarrierState` but uses only the shared
    TCPStore counter as the source of truth -- there is no
    ``torch.distributed.Work`` associated with a store barrier. Kept
    as a dataclass (not a bare tuple) so the FSM tick has a
    consistent ``rank`` / ``epoch`` / ``ready_key`` view for logging.

    ``arrival`` is this rank's within-cycle sequence number (1 for
    the elected cycle leader, 2..N for the others); ``epoch`` is the
    cohort-wide cycle id read from the shared TCPStore counter. See
    :func:`nixl_retire_barrier_post` for the elected-leader
    derivation.
    """

    epoch: int
    world_size: int
    ready_key: str
    rank: int
    arrival: int
    posted_at: float
    # False once ``nixl_retire_barrier_check`` has run at least once for
    # this state (i.e. the fold-tick check has completed). Subsequent
    # fallback ticks use the shorter ``_CATCH_UP_FALLBACK_S`` window --
    # see ``nixl_retire_barrier_check`` docstring.
    first_check: bool = True


def _pre_nixl_retire(retiree_global_ranks: List[int]) -> None:
    """Survivor-side NIXL peer disconnect run before the retire barrier.

    Extracted from the old blocking ``_refresh_nixl_ep_members`` so the
    FSM's :class:`ScaleDownSurvivorState.NIXL_RETIRE` entry hook can
    call it exactly once (on the tick that transitions from DRAIN into
    NIXL_RETIRE), while the async barrier post/check/consume runs
    across subsequent ticks with the scheduler event loop free to
    keep pumping ``mlp_sync`` and other WORLD collectives.

    Bookkeeping moves ``NixlEPBuffer._connected_ep_size`` /
    ``_scale_to`` / ``_dispatch_ep_size`` to the new size, so the
    lazy-disconnect branch in :meth:`NixlEPBuffer.get_nixl_buffer` is a
    no-op on the first post-shrink dispatch. No-op on the retiree
    itself and when NIXL a2a is not the active backend
    (``NixlEPBuffer._buffer is None``); the Mooncake a2a path performs
    the equivalent handshake via ``EPBuffer.update_ep_member`` in
    :func:`_refresh_ep_members`.
    """
    from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

    if NixlEPBuffer._state().buffer is None:
        return

    if not torch.distributed.is_initialized():
        return

    my_rank = torch.distributed.get_rank()
    if my_rank in retiree_global_ranks:
        return

    t0 = time.monotonic()
    NixlEPBuffer.on_retire(retiree_global_ranks)
    logger.info(
        "[Elastic EP][retire] rank=%d nixl on_retire took %.3fs",
        my_rank,
        time.monotonic() - t0,
    )


def nixl_retire_barrier_post(
    retiree_global_ranks: List[int],
) -> Optional[_NixlRetireBarrierState]:
    """Post the NIXL retire barrier and return a handle for async polling.

    Replaces the blocking ``_nixl_retire_store_barrier`` that used to
    sit inside :func:`try_retire_ranks`. Split into post/check/consume
    so the :class:`ScaleDownStateMachine.NIXL_RETIRE` state can drive
    the barrier across scheduler ticks without stalling the scheduler
    event loop for the whole rendezvous window -- a blocking wait
    holds every survivor out of ``mlp_sync`` (itself a WORLD
    collective on ``dp_cpu_group``), and that mutual dependency
    deadlocks whenever any joiner rank is slow to reach the barrier
    (e.g. mid-DeepGEMM-JIT when the shrink kicks off).

    Under Mooncake ``torch.distributed.barrier(WORLD)`` is not strictly
    synchronous -- observed skews of a few seconds between survivors
    on the same barrier call. During that window retirees can flip
    their mask and ``sys.exit(0)``, corrupting the next WORLD-scope
    broadcast on a lagging survivor. The store barrier is strictly
    synchronous by construction: every rank must post to the same
    key and every rank must observe ``count >= world_size`` before
    proceeding.

    Why TCPStore and not a Gloo sub-group: PyTorch's
    ``dist.new_group(..., use_local_synchronization=True)`` derives the
    subgroup's store prefix from a SHA1 hash that salts the rank list
    with process-local ``len(_world.pg_names)``. Survivors and joiner
    subprocesses have different PG-creation histories at the recover
    boundary, so they compute different subgroup names and never
    rendezvous. The global TCPStore (created for NIXL by
    :func:`sglang.srt.distributed.utils._create_global_tcp_store`) has
    no such dependency and is already the source of truth for the
    OUTER retire barrier
    (:func:`retire_barrier_post` / :func:`retire_barrier_check` /
    :func:`retire_barrier_consume`).

    Cohort-wide epoch derivation (why not a per-process counter): a
    naive module-level ``_EPOCH`` fails whenever cycle-N retirees are
    joiner subprocesses spawned during the cycle-(N-1) recover --
    their local counter is at 0 while the survivors' is at N-1, so
    per-rank keys diverge and no rendezvous happens. We instead
    derive the epoch from an ELECTED-LEADER pattern on the shared
    TCPStore:

      * Every rank calls ``store.add(ARRIVAL_COUNTER_KEY, 1)`` on
        entry and gets its within-cycle arrival number.
      * The rank whose add returned 1 is the elected cycle leader.
        It bumps a second cohort-wide counter
        ``_NIXL_RETIRE_CYCLE_KEY`` via a fresh ``store.add`` and
        uses that return value as the epoch. Every non-leader
        busy-polls the same key until it observes a value strictly
        greater than the last epoch this rank consumed.
      * :func:`nixl_retire_barrier_consume` resets
        ``ARRIVAL_COUNTER_KEY`` to ``0`` (idempotent under racing
        consumes) so the NEXT cycle's ``store.add`` starts at 1
        again and re-elects a leader.
      * Invariant that makes this correct: retire boundaries are
        strictly sequential (the outer retire barrier + FSM ensures
        cycle N completes before cycle N+1 starts on any rank), so
        the reset happens between cycles with no concurrent arrivals.
        The elected-leader pattern is robust to variable cohort
        sizes (chained shrinks like 4 -> 3 -> 2 -> 1) which the old
        divisor-based derivation was not.

    Returns ``None`` when:

      * NIXL a2a is not the active backend (nothing to synchronize on),
      * torch.distributed is not initialized (test / smoke path),
      * the global TCPStore is not available (legacy launch).

    A ``None`` return tells the FSM there is no async barrier to wait
    on -- the NIXL_RETIRE state advances immediately on the next tick.

    Called on every rank (survivors and retirees) on the FSM tick that
    transitions DRAIN -> NIXL_RETIRE. Survivors must have called
    :func:`_pre_nixl_retire` right before this call so the peer
    disconnect completes before the counter post; retirees skip the
    disconnect.
    """
    from sglang.srt.distributed.utils import get_global_tcp_store
    from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

    if NixlEPBuffer._state().buffer is None:
        return None

    if not torch.distributed.is_initialized():
        return None

    my_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Barrier target: the count of ranks that will actually post at THIS
    # Shared helper: for a chained shrink like ``4 -> 3 -> 2`` this
    # yields the pre-shrink cohort size (3), NOT the launch
    # ``world_size`` (4); mirrors :func:`retire_barrier_post`.
    barrier_target = _barrier_target_from_effective_ep(
        my_rank, world_size, "[Elastic EP][retire]"
    )

    store = None
    try:
        store = get_global_tcp_store()
    except Exception as exc:
        logger.warning(
            "[Elastic EP][retire] rank=%d get_global_tcp_store failed (%s); "
            "skipping NIXL retire barrier (no strict sync -- legacy path)",
            my_rank,
            exc,
        )
        return None

    if store is None:
        logger.warning(
            "[Elastic EP][retire] rank=%d no global TCPStore; skipping "
            "NIXL retire barrier (no strict sync -- legacy path)",
            my_rank,
        )
        return None

    global _last_local_nixl_retire_cycle_id

    try:
        arrival = int(store.add(_NIXL_RETIRE_ARRIVAL_COUNTER_KEY, 1))
    except Exception as exc:
        logger.warning(
            "[Elastic EP][retire] rank=%d TCPStore add on arrival counter "
            "failed (%s); skipping NIXL retire barrier",
            my_rank,
            exc,
        )
        return None

    # Cohort-wide cycle id via elected first-arriver. The arrival
    # counter is reset to 0 in :func:`nixl_retire_barrier_consume`
    # (idempotent on all ranks) so cycle N+1's post starts at
    # arrival=1 again. The rank whose ``store.add`` returned 1 is
    # therefore the elected leader for this cycle: it bumps the
    # shared cycle counter; every other rank busy-polls it. This
    # replaces the previous ``(arrival - 1) // world_size + 1``
    # derivation, which straddled bucket boundaries whenever a
    # cycle's poster count did not equal ``world_size``.
    epoch: Optional[int] = None
    if arrival == 1:
        try:
            epoch = int(store.add(_NIXL_RETIRE_CYCLE_KEY, 1))
        except Exception as exc:
            logger.warning(
                "[Elastic EP][retire] rank=%d TCPStore add on cycle counter "
                "failed (%s); falling back to legacy epoch derivation",
                my_rank,
                exc,
            )
    else:
        cycle_deadline = time.monotonic() + _NIXL_RETIRE_CYCLE_ID_CATCH_UP_S
        while time.monotonic() < cycle_deadline:
            try:
                raw = store.get(_NIXL_RETIRE_CYCLE_KEY)
                candidate = int(raw.decode() if isinstance(raw, bytes) else raw)
            except Exception as exc:
                logger.debug(
                    "[Elastic EP][retire] rank=%d TCPStore get on cycle "
                    "counter transient error (%s); reusing local cycle id",
                    my_rank,
                    exc,
                )
                candidate = _last_local_nixl_retire_cycle_id
            if candidate > _last_local_nixl_retire_cycle_id:
                epoch = candidate
                break
            time.sleep(_NIXL_RETIRE_STORE_BARRIER_POLL_INTERVAL_S)

    if epoch is None:
        # TCPStore ``cycle_id`` unavailable or leader's write failed.
        # We cannot safely derive an epoch that every rank in this
        # cohort will agree on -- any local heuristic (e.g. bumping
        # ``_last_local_nixl_retire_cycle_id + 1`` per-rank) risks
        # split epochs across ranks and a cohort-wide FSM stall.
        # Skip the barrier entirely; the FSM treats a ``None``
        # return as "no barrier to wait on" and advances via its
        # outer timeout mechanism (600 s ``elastic_ep_scale_timeout``
        # in ``maybe_retire_ep_ranks``). This is a strict degradation,
        # but a bounded-latency failure beats an indefinite split.
        #
        # Leader rolls back its ARRIVAL_KEY increment so the next cycle
        # can still elect a leader -- see :func:`_rollback_arrival_counter`
        # for the full rationale.
        if arrival == 1:
            _rollback_arrival_counter(
                store, my_rank, _NIXL_RETIRE_ARRIVAL_COUNTER_KEY
            )
        logger.warning(
            "[Elastic EP][retire] rank=%d arrival=%d could not observe "
            "cycle id within %.1fs; skipping NIXL retire barrier for "
            "this cycle (outer scale timeout will catch a real hang)",
            my_rank,
            arrival,
            _NIXL_RETIRE_CYCLE_ID_CATCH_UP_S,
        )
        return None

    _last_local_nixl_retire_cycle_id = epoch
    ready_key = f"sglang_nixl_retire_barrier_e{epoch}_posted"

    try:
        posted = int(store.add(ready_key, 1))
    except Exception as exc:
        logger.warning(
            "[Elastic EP][retire] rank=%d TCPStore add failed for e=%d (%s); "
            "skipping NIXL retire barrier",
            my_rank,
            epoch,
            exc,
        )
        return None

    logger.info(
        "[Elastic EP][retire] rank=%d posting NIXL retire barrier arrival=%d "
        "epoch=%d (count-after-add=%d, target=%d, world_size=%d)",
        my_rank,
        arrival,
        epoch,
        posted,
        barrier_target,
        world_size,
    )

    return _NixlRetireBarrierState(
        epoch=epoch,
        world_size=barrier_target,
        ready_key=ready_key,
        rank=my_rank,
        arrival=arrival,
        posted_at=time.monotonic(),
    )


def nixl_retire_barrier_check(
    state: Optional[_NixlRetireBarrierState],
) -> bool:
    """Non-blocking probe: has every cohort rank posted the NIXL retire
    barrier yet?

    Returns ``True`` when ``count >= world_size`` (all ranks have
    entered :func:`nixl_retire_barrier_post`), which is the invariant
    the FSM cares about before advancing NIXL_RETIRE -> FLIP_MASK.
    ``None`` (no barrier posted, e.g. non-NIXL backend) is also treated
    as ready -- the FSM just skips the wait.

    Bounded catch-up busy-poll (fold rationale, see
    ``scale_down_state.py`` DRAIN branch): the fold pattern calls
    ``check`` in the SAME tick as ``post``, so an early-arriving rank
    typically observes ``count < world_size`` on its first read
    because peer ranks post ~ms later. Returning ``False`` immediately
    would surrender the tick to the event loop, where a peer that HAS
    reached ``count == world_size`` could fold FLIP_MASK, close the
    admission gate, and wedge this rank's next ``mlp_sync`` all-gather
    on WORLD.

    Two-tier catch-up window:
      * First check (fold-tick, ``state.first_check``): spin up to
        ``_CATCH_UP_FIRST_S`` (5s) to close the sub-100ms post race.
        The 5s bound is loose because in the healthy case the poll
        returns early -- once every peer's ``store.add`` completes,
        the very next 50ms read observes ``count == world_size`` and
        returns ``True``.
      * Subsequent checks (NIXL_RETIRE fallback ticks): spin up to
        ``_CATCH_UP_FALLBACK_S`` (200ms). Sufficient to catch a
        counter update that lands during the tick (typical FSM tick
        cadence is 10-50ms), but never burns 5s of scheduler thread
        on a stuck cohort. The prior single-tick 5s spin on every
        fallback risked starving unrelated work (KV admission, DPC
        control) for seconds at a time when the actual counter
        update was already visible microseconds later.

    Always re-runs the poll on every check (no one-shot flag): the
    previous ``catch_up_used`` design skipped the poll on subsequent
    calls, which deadlocked MC09 cycle 2 when the fold's 5s spin
    ended exactly as the remaining ranks completed their
    ``store.add`` -- the very next non-blocking read raced the
    counter update and returned ``False``, leaving the FSM pinned in
    NIXL_RETIRE with no re-poll path.

    Also enforces the 300s deadline so a stuck cohort (e.g. a joiner
    caught in a torch.distributed collective without a timeout) fails
    the FSM fast instead of wedging the scheduler. Raises TimeoutError
    on deadline exceeded, which the FSM catches and marks the state
    machine FAILED.
    """
    if state is None:
        return True

    from sglang.srt.distributed.utils import get_global_tcp_store

    try:
        store = get_global_tcp_store()
    except Exception as exc:
        logger.debug(
            "[Elastic EP][retire] rank=%d get_global_tcp_store() raised (%s) "
            "in NIXL retire barrier check; treating as ready",
            state.rank,
            exc,
        )
        store = None
    if store is None:
        # Store went away since post; treat as ready so we don't wedge.
        return True

    catch_up_window_s = (
        _NIXL_RETIRE_STORE_BARRIER_CATCH_UP_FIRST_S
        if state.first_check
        else _NIXL_RETIRE_STORE_BARRIER_CATCH_UP_FALLBACK_S
    )
    state.first_check = False
    catch_up_deadline = time.monotonic() + catch_up_window_s
    last_seen_count: Optional[int] = None
    while True:
        try:
            raw = store.get(state.ready_key)
            count = int(raw.decode() if isinstance(raw, bytes) else raw)
            last_seen_count = count
            if count >= state.world_size:
                return True
        except Exception as exc:
            logger.debug(
                "[Elastic EP][retire] rank=%d TCPStore get transient error "
                "for e=%d: %s",
                state.rank,
                state.epoch,
                exc,
            )
        if time.monotonic() >= catch_up_deadline:
            break
        time.sleep(_NIXL_RETIRE_STORE_BARRIER_POLL_INTERVAL_S)

    if time.monotonic() - state.posted_at > _NIXL_RETIRE_STORE_BARRIER_TIMEOUT_S:
        raise TimeoutError(
            f"[Elastic EP][retire] rank={state.rank} NIXL retire store barrier "
            f"e={state.epoch} (arrival={state.arrival}) timed out at "
            f"count={last_seen_count}/{state.world_size} after "
            f"{_NIXL_RETIRE_STORE_BARRIER_TIMEOUT_S:.0f}s"
        )
    return False


def nixl_retire_barrier_consume(
    state: Optional[_NixlRetireBarrierState],
) -> None:
    """Finalize the NIXL retire barrier.

    Called after :func:`nixl_retire_barrier_check` has returned
    ``True``. There is no ``torch.distributed.Work`` handle to
    ``wait()`` on for a store barrier -- the check already verified
    every rank has posted -- so the only remaining work is to reset
    the arrival counter (so the NEXT cycle's post starts at
    ``arrival=1`` and re-elects a fresh leader; see
    :func:`nixl_retire_barrier_post`).

    Reset invariant: ONLY the elected leader for this cycle
    (``state.arrival == 1``) writes ``ARRIVAL_KEY = 0``. If every
    rank did the reset, a fast rank that already completed cycle N's
    fold and posted cycle (N+1)'s ``store.add(ARRIVAL_KEY, 1)`` would
    have its ``arrival = 1`` clobbered back to 0 by a slow rank
    still finishing cycle N's consume, letting a THIRD rank get
    ``arrival = 1`` again for cycle N+1 -- two leaders, two
    ``store.add`` calls on ``CYCLE_KEY``, cohort splits across two
    epochs. Restricting the reset to the leader (which is
    monotonically first-in / first-out for its own cycle) avoids
    this race.
    """
    if state is None:
        return
    if state.arrival == 1:
        try:
            from sglang.srt.distributed.utils import get_global_tcp_store

            store = get_global_tcp_store()
            if store is not None:
                store.set(_NIXL_RETIRE_ARRIVAL_COUNTER_KEY, "0")
        except Exception as exc:
            logger.debug(
                "[Elastic EP][retire] rank=%d (leader) TCPStore reset "
                "failed for e=%d (%s); next cycle's arrival counter "
                "may be stale",
                state.rank,
                state.epoch,
                exc,
            )
    logger.info(
        "[Elastic EP][retire] rank=%d NIXL retire barrier consumed "
        "e=%d (arrival=%d, wait=%.3fs)",
        state.rank,
        state.epoch,
        state.arrival,
        time.monotonic() - state.posted_at,
    )


_PEER_STATE_POLL_INTERVAL_SEC = 0.01


def _iter_live_parallel_groups() -> Iterator[parallel_state.GroupCoordinator]:
    groups = []
    for group_ref in parallel_state._groups.values():
        group = group_ref()
        if group is not None:
            groups.append(group)
    yield from sorted(groups, key=lambda group: group.unique_name)


def _map_global_to_group_local_ranks(
    group_ranks: List[int], global_ranks: List[int]
) -> List[int]:
    rank_to_local = {rank: index for index, rank in enumerate(group_ranks)}
    return [rank_to_local[rank] for rank in global_ranks if rank in rank_to_local]


def _wait_for_peer_state(backend, ranks: List[int]) -> None:
    from mooncake.pg import get_peer_state

    while not all(get_peer_state(backend, ranks)):
        time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)


def _maybe_create_message_queue(group) -> None:
    if not group.use_message_queue_broadcaster or group.world_size <= 1:
        return

    from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue

    group.mq_broadcaster = MessageQueue.create_from_process_group(
        group.cpu_group, 1 << 22, 6
    )


def _try_recover_world(
    global_ranks: List[int],
    *,
    include_subgroups: bool = False,
) -> bool:
    """Recover WORLD-scope Mooncake peers for the rejoining ranks.

    ``include_subgroups`` (recover-mode only): also recover the sglang
    ``_WORLD`` group coordinator's own device+cpu backends. In recover-mode
    grow the joiner's ``parallel_state._WORLD`` uses launch-time PG IDs, so
    the survivor must call ``recover_ranks`` on those sub-PGs too to pair
    with the joiner's ``join_group``; otherwise the next
    ``all_gather_object(cpu_group=...)`` unpickles an empty tensor.

    In scale-up-v1 append the joiner is a NEW rank whose ``_WORLD`` sub-PGs
    have DIFFERENT Mooncake IDs than the primary's launch-time sub-PGs.
    Recovering the primary's sub-PGs against such a joiner deadlocks the
    survivor in ``get_peer_state``; the caller (``try_admit_scale_ranks``)
    therefore passes ``include_subgroups=False``.
    """
    from mooncake.pg import get_peer_state, recover_ranks

    world_backend = torch.distributed.group.WORLD
    if not all(get_peer_state(world_backend, global_ranks)):
        return False

    recover_ranks(world_backend, global_ranks)
    logger.debug("[Elastic EP][recover] WORLD recover_ranks(%s) done", global_ranks)

    if not include_subgroups:
        return True

    _WORLD_RECOVER_WAIT_TIMEOUT_S = 60.0
    world_group = parallel_state._WORLD
    if world_group is not None:
        for pg in (world_group.device_group, world_group.cpu_group):
            if pg is None or pg is world_backend:
                continue
            wait_start = time.monotonic()
            while not all(get_peer_state(pg, global_ranks)):
                if time.monotonic() - wait_start > _WORLD_RECOVER_WAIT_TIMEOUT_S:
                    return False
                time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)
            recover_ranks(pg, global_ranks)
    return True


def try_admit_scale_ranks(global_ranks: List[int]) -> bool:
    """Admit append-only ranks into the expandable WORLD group.

    Symmetric with :func:`try_recover_ranks` on the mask-flip side: the
    admitted ranks live in ``[elastic_ep_initial_size, max_ep_size)`` and
    were born with ``active_ranks[i] = 0``. Mooncake's C++ side is
    updated by ``mooncake_ep.recover_ranks(WORLD, ...)`` inside
    :func:`_try_recover_world`, but the Python-owned tensors (top-level
    :class:`ElasticEPStateManager` bitmap, sub-group masks, and the
    default WORLD backend handle) all need an explicit flip to publish
    the same value that :func:`get_backend_active_ranks(WORLD)` and
    :func:`try_retire_ranks` will later read.

    Without these three flips, a subsequent scale-DOWN that retires an
    ex-scale-up-v1-append joiner (e.g. ``4 -> 6 -> 3``) reads a stale
    ``active_ranks[8] = 0`` on the survivor's WORLD backend mask, and
    the WORLD-scope collectives inside ``_finalize_scale_down`` observe
    an inconsistent view relative to Mooncake's C++ mask (which was
    already flipped to 1 during grow). This is the mask-flip half of
    the fix; sub-group iteration is a no-op today because scale-mode
    append never joins an existing sub-group, but it is kept for
    symmetry with the recover path so a future sub-group topology
    change doesn't silently regress the append path.

    ``include_subgroups=False`` on the WORLD recover call: scale-up-v1
    joiners boot with a fresh ``parallel_state._WORLD`` coordinator
    whose ``device_group`` / ``cpu_group`` sub-PGs have Mooncake IDs
    distinct from the primary's launch-time sub-PG IDs (see
    :func:`_try_recover_world` for the deadlock this would cause).
    """
    if not _try_recover_world(global_ranks, include_subgroups=False):
        return False

    inst = ElasticEPStateManager.instance()
    if inst is not None:
        inst.activate_ranks(global_ranks)

    for group in _iter_live_parallel_groups():
        _flip_active_rank_mask(group, global_ranks, value=1)
        _maybe_create_message_queue(group)

    # ``mooncake_ep.recover_ranks(WORLD, ...)`` inside ``_try_recover_
    # world`` flipped the mask on Mooncake's C++ side. Publish the same
    # value on the python-side handle so ``get_backend_active_ranks
    # (WORLD)`` observers agree -- symmetric with ``try_recover_ranks``.
    _flip_world_backend_active_rank_mask(global_ranks, value=1)

    _refresh_ep_members()
    return True


def try_recover_ranks(global_ranks: List[int]) -> bool:
    """Recover ranks in WORLD and flip ``active_ranks`` on every live PG.

    In DP-attention launches every multi-rank sub-group has the same
    rank set as WORLD; the single-rank sub-groups (``attention_tp:0``,
    ``moe_dp:0``, ...) do not contain the retired rank. Waiting on a
    sub-group's ``get_peer_state`` would also deadlock: the joiner
    subprocess runs with local ``tp=1/dp=1``, its own ``tp:0`` PG has
    ``world_size=1``, which :func:`join_process_groups` skips. So the
    joiner never publishes sub-group metadata and
    ``peerConnected[retiree]`` on the survivor's sub-group PGs stays
    False forever.

    Since the retired rank re-establishes its Mooncake connections via
    ``join_group(WORLD)`` (which the survivor's WORLD-scoped
    ``ConnectionPoller`` picks up), it is sufficient to:

      * Flip ``active_ranks[retiree] = 1`` on every live sub-group's
        Python-side mask (so subsequent collectives honour the
        rejoined rank).
      * Refresh the MoE EP peer table (``_refresh_ep_members``).

    This matches :func:`try_retire_ranks` which also only writes the
    mask on sub-groups.

    TODO: there is no cohort-scope barrier around this call today.
    Every incumbent survivor drives its own ``try_recover_ranks``
    from :meth:`ModelRunner.maybe_join_ep_ranks` based on its local
    per-request state, so a slow survivor can still be advertising
    the pre-grow ``active_ranks`` mask on WORLD when a fast survivor
    has already flipped and re-broadcast the expanded expert-
    location map. A naive cohort barrier here would deadlock the
    survivor cohort against in-flight ``mlp_sync`` all-gather
    (survivors call ``try_recover_ranks`` outside the FSM's
    admission gate, so ``mlp_sync`` is still live on WORLD when
    this runs). A proper fix requires either (a) moving the recover
    path inside the elastic FSM's admission gate (like scale-down)
    or (b) using a dedicated grow-side FSM with its own drain
    barrier.
    """
    # ``include_subgroups=True``: recover-mode joiners rebuild the
    # launch-time ``parallel_state._WORLD.device_group`` and
    # ``cpu_group`` sub-PGs on their side too, so the survivor's
    # ``recover_ranks`` on those sub-PGs completes the Mooncake
    # handshake and prevents a stale-retiree ``EOFError`` on the
    # next WORLD-scope ``all_gather_object(cpu_group=...)``.
    if not _try_recover_world(global_ranks, include_subgroups=True):
        return False

    for group in _iter_live_parallel_groups():
        _flip_active_rank_mask(group, global_ranks, value=1)
        _maybe_create_message_queue(group)

    # ``mooncake_ep.recover_ranks(WORLD, ...)`` above flipped the mask on
    # Mooncake's C++ side. Publish the same value on the python-side
    # handle so ``get_backend_active_ranks(WORLD)`` observers agree.
    _flip_world_backend_active_rank_mask(global_ranks, value=1)

    _refresh_ep_members()
    return True


def _join_world_group(*, include_subgroups: bool = False) -> None:
    """Publish the joiner's WORLD-scope Mooncake state.

    Always joins ``torch.distributed.group.WORLD`` (the default Mooncake
    device backend). ``include_subgroups=True`` (recover-mode only): also
    join ``parallel_state._WORLD.device_group`` / ``_WORLD.cpu_group``.
    These pair with the survivor's ``recover_ranks`` on the same sub-PGs
    inside ``_try_recover_world``; without this pairing the next
    ``all_gather_object(cpu_group=...)`` unpickles an empty tensor.

    In scale-up-v1 append the joiner's sub-PGs have DIFFERENT Mooncake IDs
    than the primary's launch-time sub-PGs, so ``include_subgroups=False``
    matches the primary's ``try_admit_scale_ranks`` and avoids a deadlock
    inside ``join_group(sub_pg)``.
    """
    from mooncake.pg import join_group

    world_backend = torch.distributed.group.WORLD
    join_group(world_backend)

    if not include_subgroups:
        return

    world_group = parallel_state._WORLD
    if world_group is not None:
        for pg in (world_group.device_group, world_group.cpu_group):
            if pg is None or pg is world_backend:
                continue
            join_group(pg)


def join_scale_process_group() -> None:
    """Join the expandable WORLD group for a scale-up-v1 append.

    The joiner is a fresh rank appended beyond ``elastic_ep_initial_
    size``; its ``parallel_state._WORLD`` sub-PGs were freshly built
    at boot with different Mooncake IDs than the primary's launch-
    time sub-PGs, so ``include_subgroups=False`` -- only the default
    WORLD Mooncake backend is joined (matched by the survivor's
    :func:`_try_recover_world` with ``include_subgroups=False``).
    """
    _join_world_group(include_subgroups=False)
    _refresh_ep_members()


def join_process_groups() -> None:
    """Rejoin WORLD after a recover-mode grow.

    In DP-attention launches every multi-rank sub-group has the same rank
    set as WORLD, and the joiner subprocess runs with local tp=1/dp=1.
    Iterating sub-groups for ``join_group`` would either re-issue the
    WORLD join or block on a size-1 no-op, so we only rejoin WORLD plus
    the sglang ``_WORLD.device_group`` / ``_WORLD.cpu_group`` sub-PGs
    (via ``include_subgroups=True``). Those sub-PGs still carry the
    retiree's old peer entry on the survivor side, so a matched
    ``recover_ranks(_WORLD.cpu_group, ...)`` inside ``_try_recover_world``
    (also with ``include_subgroups=True``) is required to avoid an
    ``EOFError`` on the next ``all_gather_object(cpu_group=...)``.
    """
    _join_world_group(include_subgroups=True)
    _refresh_ep_members()


def get_healthy_expert_location_src_rank(
    *, invoked_in_elastic_ep_rejoin_path: bool
) -> int:
    world_group = get_world_group()
    # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
    # A rank that was started as a rejoin rank may later act as a healthy
    # rank in a subsequent recovery cycle.
    local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
    gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

    for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
        if not is_rejoin_rank:
            return world_group.ranks[rank_in_group]

    raise RuntimeError(
        "No healthy rank found for broadcasting expert location metadata. "
        "All ranks are marked as elastic_ep_rejoin."
    )


def maybe_recover_ep_ranks(
    *,
    tp_group: parallel_state.GroupCoordinator,
    eplb_manager: EPLBManager,
    model_config: ModelConfig,
    moe_ep_rank: int,
) -> bool:
    # TODO(perf): `active_ranks.all()` on a CUDA tensor triggers host-device
    # synchronization, and this function is on the forward-path.
    # This check only runs when `--elastic-ep-backend` is enabled, so the
    # synchronization overhead does not propagate to other configs.
    # Leave for future optimization of the elastic EP path.
    if tp_group.active_ranks.all() and tp_group.active_ranks_cpu.all():
        return False

    tp_active_ranks = tp_group.active_ranks.detach().cpu().numpy()
    tp_active_ranks_cpu = tp_group.active_ranks_cpu.detach().numpy()
    tp_active_ranks &= tp_active_ranks_cpu
    # NOTE: `ranks_to_recover` uses indices in `tp_group`. For the current
    # Mooncake elastic EP implementation we assume `--pp-size=1`, so the
    # tp-group index is the same as the global rank index.
    ranks_to_recover = [
        i for i in range(len(tp_active_ranks)) if not tp_active_ranks[i]
    ]

    # try_recover_ranks polls peer state via Mooncake EP backend.
    # Mooncake's internal semantics guarantee that all ranks observe
    # consistent peer readiness state, so collective operations below
    # are safe even though polling appears local.
    if ranks_to_recover and try_recover_ranks(ranks_to_recover):
        eplb_manager.reset_generator()
        broadcast_global_expert_location_metadata(
            model_config=model_config,
            moe_ep_rank=moe_ep_rank,
            src_rank=get_healthy_expert_location_src_rank(
                invoked_in_elastic_ep_rejoin_path=False
            ),
        )
        ElasticEPStateManager.instance().reset()
        logger.info(f"recover ranks {ranks_to_recover} done")
        return True

    return False


def maybe_rebalance_after_rank_fault(*, eplb_manager: EPLBManager) -> bool:
    elastic_ep_state = ElasticEPStateManager.instance()
    if elastic_ep_state is None or elastic_ep_state.is_active_equal_last():
        return False
    elastic_ep_state.snapshot_active_to_last()
    elastic_ep_state.sync_active_to_cpu()
    logger.info("EPLB due to rank faults")
    gen = eplb_manager.rebalance()
    while True:
        try:
            next(gen)
        except StopIteration:
            break
    return True


def _flip_world_backend_active_rank_mask(global_ranks: List[int], value: int) -> None:
    """Flip the default ``torch.distributed.group.WORLD`` active_ranks mask.

    Elastic sub-groups (created via :class:`parallel_state.GroupCoordinator`)
    keep a python-side handle to the tensor they passed into
    :class:`MooncakeBackendOptions`. The default WORLD PG has no such
    wrapper -- pybind11's ``ProcessGroup`` refuses arbitrary attribute
    assignment -- so :func:`parallel_state.init_distributed_environment`
    publishes the tensor via
    :func:`parallel_state.get_world_backend_active_ranks`. Without this
    second flip, collectives on ``dist.group.WORLD`` (mlp_sync in
    DP-attention, dist.barrier) keep reading a stale mask and drift out
    of sync with sub-groups after a shrink or grow-back.
    """
    active_ranks = parallel_state.get_world_backend_active_ranks()
    if active_ranks is None:
        return
    ranks = parallel_state.get_world_backend_ranks()
    local_ranks = _map_global_to_group_local_ranks(ranks, global_ranks)
    if not local_ranks:
        return
    for local_rank in local_ranks:
        active_ranks[local_rank] = value


def _flip_active_rank_mask(group, global_ranks: List[int], value: int) -> None:
    """Write ``value`` (0 or 1) into ``group.active_ranks[local_rank]``.

    Mirrors what ``mooncake_ep.recover_ranks`` does for the 0->1 direction
    on the ``MooncakeBackendOptions.activeRanks_`` tensor, except we do the
    write directly since Mooncake exposes no ``deactivate_ranks`` primitive.
    Skips ranks not in ``group.ranks`` (retiree not a member of this
    subgroup).

    The tensor referenced by ``group.active_ranks`` is the same slice that
    was passed into ``MooncakeBackendOptions`` at PG construction (see
    ``parallel_state.py:376-377``), so an in-place write here is observed
    by the Mooncake C++ backend on the next collective read.
    """
    local_ranks = _map_global_to_group_local_ranks(group.ranks, global_ranks)
    if not local_ranks:
        return
    active_ranks = getattr(group, "active_ranks", None)
    active_ranks_cpu = getattr(group, "active_ranks_cpu", None)
    if active_ranks is None:
        return
    for local_rank in local_ranks:
        active_ranks[local_rank] = value
    if active_ranks_cpu is not None:
        for local_rank in local_ranks:
            active_ranks_cpu[local_rank] = value


def try_retire_ranks(global_ranks: List[int]) -> bool:
    """Retire ranks in WORLD and every launch-time parallel group.

    Symmetric counterpart to :func:`try_recover_ranks`. Called
    collectively from every rank in the current cohort (survivors AND
    retirees) after the drain barrier has completed. Each rank:

    1. Flips ``active_ranks[retiree]=0`` on every live parallel group's
       ``active_ranks`` / ``active_ranks_cpu`` tensor.
    2. Also flips the top-level ``ElasticEPStateManager`` bitmap.
    3. Refreshes the Mooncake EPBuffer peer table so subsequent MoE
       dispatch skips the retired slots.

    Unlike :func:`try_recover_ranks`, no ``mooncake_ep`` primitive is
    called: Mooncake exposes ``join_group`` / ``recover_ranks`` (grow
    direction) but no counterpart for retirement. The mask tensor is
    Python-owned (constructed in :func:`sglang.srt.distributed.parallel_
    state.GroupCoordinator.__init__` and passed by reference into
    ``MooncakeBackendOptions``), so a direct in-place write is the
    Mooncake-recommended way to update it -- consistent with how
    :meth:`ModelRunner._finalize_scale_up`-style code paths already
    flip the mask via :meth:`ElasticEPState.activate_ranks` for grow.

    Retirees stay in the WORLD PG's membership list; their slots remain
    reserved so a future :func:`try_recover_ranks` on the same rank can
    reactivate them cleanly (slot-reuse property).
    """
    if not global_ranks:
        return True

    # NIXL-a2a retire-time handshake (survivor peer disconnect +
    # cohort-wide store barrier) is driven asynchronously by the
    # FSM's NIXL_RETIRE state via :func:`_pre_nixl_retire` /
    # :func:`nixl_retire_barrier_post` / :func:`nixl_retire_barrier_check`
    # / :func:`nixl_retire_barrier_consume` -- see
    # :class:`ScaleDownStateMachine._tick_survivor`. Doing the peer
    # disconnect + barrier inline here would block the scheduler
    # event loop for the whole rendezvous window and deadlock
    # against ``mlp_sync`` on cohort ranks that are slow to reach
    # the barrier.

    inst = ElasticEPStateManager.instance()
    if inst is not None:
        inst.deactivate_ranks(global_ranks)

    for group in _iter_live_parallel_groups():
        _flip_active_rank_mask(group, global_ranks, value=0)

    # Retire the default WORLD PG's mask too -- Mooncake has no
    # ``retire_ranks`` primitive to flip its C++ tensor, so any
    # WORLD-scope collective would otherwise keep waiting for the
    # retirees after they exit. This closes the same gap that already
    # exists for GroupCoordinator sub-groups above.
    _flip_world_backend_active_rank_mask(global_ranks, value=0)

    _refresh_ep_members()
    logger.debug("[Elastic EP][retire] retire_ranks(%s) done", global_ranks)
    return True


@dataclass
class _RetireBarrierState:
    """Per-cycle handle for the async retire barrier.

    Wraps the ``torch.distributed.Work`` from ``barrier(async_op=True)``
    with a TCP-store "posted" counter, because on the Mooncake WORLD
    backend ``Work.is_completed()`` is observed to return ``False``
    forever on some ranks even after the barrier has completed on
    others (see :func:`retire_barrier_check` docstring). The store
    counter is polled every tick as a reliable readiness signal so no
    rank races ahead of the cohort into ``try_retire_ranks``.

    ``arrival`` is 1 for the elected cycle leader (see
    :func:`retire_barrier_post`) and ``>=2`` for followers. Used by
    :func:`retire_barrier_consume` to gate the arrival-counter reset:
    a non-leader reset would race a fast rank that already entered
    the next cycle's ``store.add`` and let a THIRD rank get
    ``arrival=1`` again, splitting the cohort across two epochs.
    ``0`` marks a cycle whose epoch fell back to the per-process
    counter (store unavailable), in which case the leader-only
    reset is a no-op and the arrival counter simply grows unbounded
    -- correctness is still preserved because followers under the
    fallback path never consult the arrival counter for epoch
    derivation.
    """

    handle: Optional["torch.distributed.Work"]
    epoch: int
    world_size: int
    ready_key: str
    rank: int
    arrival: int = 0


# Per-process fallback epoch counter for the outer retire barrier.
# Used only when the shared TCPStore-based elected-leader derivation
# (see below) is unreachable -- the WORLD ``async_op`` handle in
# :class:`_RetireBarrierState` is the true synchronization primitive,
# so a per-process epoch is still correctness-safe for a single
# cohort. It becomes incorrect ONLY when a cohort contains ex-joiners
# whose fresh subprocesses started with a local counter of 0 while
# incumbent ranks have already advanced their local counter; a
# TCPStore-shared epoch fixes that fast-path (see below).
_RETIRE_BARRIER_EPOCH = 0

# Shared TCPStore keys for the elected-leader epoch derivation. See
# :func:`nixl_retire_barrier_post` for the same pattern applied to
# the NIXL retire barrier.
_RETIRE_BARRIER_ARRIVAL_COUNTER_KEY = "sglang_retire_barrier_arrival_counter"
_RETIRE_BARRIER_CYCLE_KEY = "sglang_retire_barrier_cycle_counter"
_RETIRE_BARRIER_CYCLE_ID_CATCH_UP_S = 5.0
_last_local_retire_cycle_id: int = 0


def _retire_barrier_ready_key(epoch: int) -> str:
    return f"sglang_retire_barrier_e{epoch}_posted"


def _derive_shared_retire_barrier_epoch(
    store, rank: int
) -> tuple[Optional[int], int]:
    """Elected-leader epoch derivation over the shared TCPStore.

    Returns ``(epoch, arrival)``. ``epoch is None`` signals the caller
    to fall back to the per-process counter (store atomically-increment
    unavailable, or leader window elapsed without a follower observing
    the leader's bump). ``arrival`` is 1 for the elected leader and
    ``>=2`` for followers of a given cycle.

    Why not just keep the per-process counter: for a cohort that
    contains an ex-joiner whose subprocess only started AFTER the
    survivor's first shrink, the ex-joiner's per-process counter is
    lagging by that first shrink's worth of increments. Rank 0
    computes ``e2``; the ex-joiner computes ``e1``; the two write to
    different ``ready_key`` s and the atomic-counter fast-path in
    :func:`retire_barrier_check` never observes ``count >=
    world_size``. The WORLD async barrier still completes and the
    fallback branch in ``retire_barrier_check`` catches it, but the
    fast-path (which was added precisely to guard against Mooncake
    ``is_completed()`` flakes) is silently disabled for exactly the
    cohort configuration that most needs it.
    """
    global _last_local_retire_cycle_id
    try:
        arrival = int(store.add(_RETIRE_BARRIER_ARRIVAL_COUNTER_KEY, 1))
    except Exception as exc:
        logger.debug(
            "[Elastic EP][retire_barrier] rank=%d store.add arrival failed "
            "(%s); falling back to per-process epoch",
            rank,
            exc,
        )
        return None, 0

    if arrival == 1:
        try:
            epoch = int(store.add(_RETIRE_BARRIER_CYCLE_KEY, 1))
            _last_local_retire_cycle_id = epoch
            return epoch, arrival
        except Exception as exc:
            logger.debug(
                "[Elastic EP][retire_barrier] rank=%d (leader) store.add "
                "cycle_id failed (%s); falling back to per-process epoch",
                rank,
                exc,
            )
            # Leader rolls back ARRIVAL_KEY so the next cycle can still
            # elect a leader; symmetric with the reset in
            # :func:`retire_barrier_consume` for the success path. See
            # :func:`nixl_retire_barrier_post` for the full rationale.
            _rollback_arrival_counter(
                store, rank, _RETIRE_BARRIER_ARRIVAL_COUNTER_KEY
            )
            return None, arrival

    deadline = time.monotonic() + _RETIRE_BARRIER_CYCLE_ID_CATCH_UP_S
    while time.monotonic() < deadline:
        try:
            raw = store.get(_RETIRE_BARRIER_CYCLE_KEY)
            candidate = int(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception as exc:
            logger.debug(
                "[Elastic EP][retire] rank=%d TCPStore get on outer retire "
                "cycle counter transient error (%s); reusing local cycle id",
                rank,
                exc,
            )
            candidate = _last_local_retire_cycle_id
        if candidate > _last_local_retire_cycle_id:
            _last_local_retire_cycle_id = candidate
            return candidate, arrival
        time.sleep(0.05)

    logger.warning(
        "[Elastic EP][retire_barrier] rank=%d arrival=%d could not "
        "observe cycle id within %.1fs; falling back to per-process epoch",
        rank,
        arrival,
        _RETIRE_BARRIER_CYCLE_ID_CATCH_UP_S,
    )
    return None, arrival


def retire_barrier_post() -> Optional[_RetireBarrierState]:
    """Post the retire cohort-wide barrier on WORLD in async mode.

    Called by every rank (survivors + retirees) BEFORE the mask flip in
    :func:`try_retire_ranks`. Returns a state handle that the caller
    polls with :func:`retire_barrier_check` until all ranks have posted.

    Why async: the scale request is fanned out to each DP scheduler over
    independent ZMQ push sockets, so different ranks can observe the
    request one event-loop iteration apart. A blocking barrier on WORLD
    deadlocks against ``mlp_sync`` (an all-gather on ``tp_cpu_group``):
    the lagging rank enters ``mlp_sync`` while the leading ranks are
    already parked in the WORLD barrier, and neither collective can
    complete. WORLD and ``tp_cpu_group`` are separate PGs, so posting
    the barrier asynchronously lets the scheduler event loop keep
    iterating and posting the per-tick ``mlp_sync`` all-gather. Every
    rank continues to reach ``mlp_sync`` and unblock it. Eventually the
    lagging rank posts its own barrier and all four handles complete.

    Reliability: we also atomically increment a per-epoch counter on
    the shared TCPStore so :func:`retire_barrier_check` has a signal
    that is reliable across the Mooncake WORLD backend's flaky
    ``is_completed()``. The counter reaching ``world_size`` means every
    rank has entered this function, which is what the FSM cares about
    (all ranks have paused new work and posted the barrier). Without
    this store fast-path, ``is_completed()`` may return ``True`` on
    one rank while others still poll ``False`` forever; the racing
    rank then advances to FLIP_MASK and calls
    :func:`_pre_nixl_retire` (NIXL peer disconnect) while a slow
    peer is still processing the NIXL retire store barrier, starving
    ``mlp_sync`` on WORLD -- the whole cohort deadlocks.

    Epoch derivation uses an elected-leader pattern over the shared
    TCPStore so every rank in the cohort agrees on the same
    ``ready_key`` even when the cohort contains ex-joiners whose
    subprocess-local ``_RETIRE_BARRIER_EPOCH`` counter is lagging
    (mixed launch generations). If the store is unreachable or the
    leader's cycle-id write fails, each rank falls back to its own
    per-process counter; correctness is still guaranteed by the
    WORLD async barrier ``handle.wait()`` and the fallback branch in
    :func:`retire_barrier_check`, but the atomic-counter fast-path
    is disabled for that cycle.

    Two invariants preserved from the original blocking barrier:

    * All in-flight collectives on WORLD have completed (posted after
      the drain phase halts new work).
    * All ranks have reached the mask-flip point before any rank flips,
      so no survivor writes ``active_ranks[retiree]=0`` while another
      rank is still expecting the retiree to participate in a prior
      collective.
    """
    global _RETIRE_BARRIER_EPOCH
    if not torch.distributed.is_initialized():
        return None
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Barrier target: the pre-shrink cohort's ``effective_ep_size``
    # (see :func:`_barrier_target_from_effective_ep`). On chained
    # shrinks only ``effective`` ranks call ``post``; waiting for
    # launch ``world_size`` arrivals would cap the store counter
    # below the target and fall through to the unreliable
    # ``handle.is_completed()`` path in :func:`retire_barrier_check`.
    barrier_target = _barrier_target_from_effective_ep(
        rank, world_size, "[Elastic EP][retire_barrier]"
    )

    from sglang.srt.distributed.utils import get_global_tcp_store

    store = None
    try:
        store = get_global_tcp_store()
    except Exception as exc:
        logger.debug(
            "[Elastic EP][retire] rank=%d get_global_tcp_store() raised (%s) "
            "in outer retire barrier post; using per-process epoch fallback",
            rank,
            exc,
        )
        store = None

    epoch: Optional[int] = None
    arrival = 0
    if store is not None:
        epoch, arrival = _derive_shared_retire_barrier_epoch(store, rank)

    if epoch is None:
        _RETIRE_BARRIER_EPOCH += 1
        epoch = _RETIRE_BARRIER_EPOCH

    logger.info(
        "[Elastic EP][retire_barrier] rank=%d posting async barrier "
        "e=%d (arrival=%d)",
        rank,
        epoch,
        arrival,
    )
    handle = torch.distributed.barrier(
        group=torch.distributed.group.WORLD, async_op=True
    )

    ready_key = _retire_barrier_ready_key(epoch)
    if store is not None:
        try:
            store.add(ready_key, 1)
        except Exception as exc:
            logger.warning(
                "[Elastic EP][retire_barrier] rank=%d TCP store ready-key "
                "add failed for e=%d (%s); falling back to is_completed "
                "polling",
                rank,
                epoch,
                exc,
            )

    return _RetireBarrierState(
        handle=handle,
        epoch=epoch,
        world_size=barrier_target,
        ready_key=ready_key,
        rank=rank,
        arrival=arrival,
    )


def retire_barrier_check(
    state: Optional[_RetireBarrierState],
) -> bool:
    """Non-blocking check on the handle returned by
    :func:`retire_barrier_post`.

    Returns ``True`` when every rank has posted its barrier (or if there
    is nothing to wait on, e.g. torch.distributed is not initialized).

    The check runs two probes in order:

    1. TCP-store atomic counter -- reliable across the Mooncake WORLD
       backend and the primary source of truth. Every rank increments
       this counter inside :func:`retire_barrier_post`; when the poll
       observes ``count >= world_size`` every rank has entered the
       barrier, so calling ``wait()`` on the ``Work`` handle is
       guaranteed to return promptly.
    2. ``handle.is_completed()`` -- kept as a defensive fallback if the
       store is unavailable (rare; the global TCPStore is created
       during distributed init and lives for the process lifetime).

    Historical NOTE on reliability: on the Mooncake WORLD backend
    ``Work.is_completed()`` was observed to occasionally return
    ``False`` forever even after the peer's ``wait()`` had returned
    successfully (~25% frequency, both nixl-a2a and mooncake-a2a). A
    rank that got ``False`` here polled indefinitely; before the store
    probe was added this manifested as an MC02A + NIXL-a2a hang where
    rank 1 raced ahead into FLIP_MASK while rank 0/2/3 stayed in DRAIN
    polling forever.
    """
    if state is None:
        return True

    try:
        from sglang.srt.distributed.utils import get_global_tcp_store

        store = get_global_tcp_store()
        if store is not None:
            raw = store.get(state.ready_key)
            count = int(raw.decode() if isinstance(raw, bytes) else raw)
            if count >= state.world_size:
                return True
    except Exception as exc:
        logger.debug(
            "[Elastic EP][retire] rank=%d outer retire drain-barrier "
            "TCPStore probe transient error (%s); falling back to "
            "handle.is_completed()",
            state.rank,
            exc,
        )

    if state.handle is None:
        return True
    return state.handle.is_completed()


_RETIRE_BARRIER_CONSUME_TIMEOUT_S = 30.0


def retire_barrier_consume(
    state: Optional[_RetireBarrierState],
) -> None:
    """Finalize the retire barrier's ``Work`` handle.

    Calls ``handle.wait()`` after :func:`retire_barrier_check` has
    returned ``True`` so the caller can safely proceed to the mask flip.
    ``wait()`` is expected to return immediately since every rank has
    posted the barrier (either the store counter or is_completed said
    so). We nevertheless enforce a bounded ``timeout`` on ``wait()``
    so a rank whose peer entered ``sys.exit`` (retiree local cleanup)
    between the check and the consume cannot wedge the caller
    indefinitely: the ``Work`` handle is on WORLD, and Mooncake's
    WORLD backend has been observed to hang ``wait()`` forever if a
    peer disappeared mid-flight. A ``TimeoutError`` here is
    converted to a warning + best-effort proceed; the store counter
    already confirmed every rank posted, so the invariant the
    caller cares about (all ranks reached the barrier) is
    preserved.
    """
    if state is None:
        return
    t0 = time.monotonic()
    if state.handle is not None:
        try:
            # ``Work.wait`` accepts a ``timedelta`` on modern
            # torch.distributed; guard against older backends that
            # ignore the arg by wrapping in a wall-clock check.
            state.handle.wait(
                timeout=timedelta(seconds=_RETIRE_BARRIER_CONSUME_TIMEOUT_S)
            )
        except TypeError:
            # Backend does not accept the ``timeout`` kwarg. Fall
            # back to a blocking wait -- the store counter already
            # confirmed the barrier so this SHOULD return promptly.
            state.handle.wait()
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                "[Elastic EP][retire_barrier] rank=%d handle.wait(e=%d) "
                "raised after %.1fs (%s); store counter already "
                "confirmed all posts -- proceeding",
                state.rank,
                state.epoch,
                elapsed,
                exc,
            )
    # Leader-only reset of the shared arrival counter for the elected-
    # leader epoch derivation. Same rationale as
    # :func:`nixl_retire_barrier_consume`: allowing any rank to reset
    # the counter races with a fast peer that already entered the next
    # cycle's ``store.add``, so a slow rank finishing the previous
    # cycle can clobber the peer's fresh ``arrival=1`` back to 0 and
    # let a third rank re-elect itself leader on top of the peer's
    # cycle. We restrict the reset to the rank whose ``arrival == 1``
    # for THIS cycle -- that rank is monotonically first-in / first-out
    # for its own cycle, so no other rank has yet moved on. Cycles
    # whose epoch fell back to the per-process counter (``arrival ==
    # 0``) do not touch the counter; they never advertised on the
    # shared TCPStore.
    if state.arrival == 1:
        try:
            from sglang.srt.distributed.utils import get_global_tcp_store

            store = get_global_tcp_store()
            if store is not None:
                store.set(_RETIRE_BARRIER_ARRIVAL_COUNTER_KEY, "0")
        except Exception as exc:
            logger.debug(
                "[Elastic EP][retire_barrier] rank=%d (leader) TCPStore "
                "reset failed for e=%d (%s); next cycle's arrival counter "
                "may be stale",
                state.rank,
                state.epoch,
                exc,
            )
    logger.info(
        "[Elastic EP][retire_barrier] rank=%d barrier consumed e=%d "
        "(wait=%.3fs)",
        state.rank,
        state.epoch,
        time.monotonic() - t0,
    )


def retire_barrier() -> None:
    """Blocking retire barrier -- kept for standalone tests and any
    caller that has no scheduler event loop to drive async polling.

    The scheduler tick path in :class:`ScaleDownStateMachine` uses
    :func:`retire_barrier_post` / :func:`retire_barrier_check` /
    :func:`retire_barrier_consume` instead so the event loop can keep
    pumping ``mlp_sync`` on ``tp_cpu_group`` while cohort ranks that
    receive the scale request one iteration late catch up.
    """
    state = retire_barrier_post()
    retire_barrier_consume(state)


def retiree_local_cleanup() -> None:
    """Local Mooncake / CUDA quiesce, run by retirees just before exit.

    Called on the retiree AFTER :func:`retire_barrier` and BEFORE
    ``sys.exit(0)``. Deliberately narrow scope:

    * ``torch.cuda.synchronize()`` -- drain any pending GPU kernels
      launched by the last forward pass.
    * ``torch.cuda.empty_cache()`` -- release the PyTorch caching
      allocator's reservations so process teardown returns memory to the
      OS promptly.

    We deliberately do NOT call ``torch.distributed.destroy_process_
    group()``, and we do NOT explicitly tear down NIXL / Mooncake
    RDMA endpoints on the retiree side. The launch-time process
    groups (both torch default WORLD and every sglang parallel group)
    stay at their launch-time membership; destroying them here would
    trigger the well-known "destroy blocks on live NCCL/Mooncake
    comms" hang: ``torch.distributed.destroy_process_group()`` blocks
    until every in-flight collective on every PG completes, which
    never happens when peers are still holding live Mooncake QPs
    against this retiree.

    RDMA endpoint teardown is deferred to kernel-driven process
    cleanup for two reasons:

    1. The survivor already dropped its RDMA QP state to this
       retiree via
       :meth:`layers.moe.token_dispatcher.nixl.EPBuffer.on_retire`
       (called inside :func:`try_retire_ranks` under the retire
       barrier). The survivor no longer waits for
       ``FINISHED_SUM_TAG`` contributions from the retiree, so an
       abrupt retiree exit cannot wedge a survivor collective.
    2. Explicit ``ep_buffer.destroy()`` on the retiree could hang
       if the Mooncake backend's WORLD-scope teardown races the
       survivors' post-shrink Mooncake state, since Mooncake C++
       peer arrays are fixed-size and expect every launch-time
       slot to still be reachable during destroy. Waiting for a
       destroy timeout on the retiree would delay process exit
       and leave the survivor-side ``ElasticScaleUpdateReq``
       ``direction=shrink`` reporting a longer end-to-end shrink
       time.

    Process exit reclaims all Mooncake / NCCL / CUDA / NIXL state
    for this process at the kernel level (page tables, GPU BAR
    mappings, RDMA verbs QPs, TCPStore file descriptors). The
    reserved-but-inactive slot stays available for a later
    ``recover_ranks`` reactivation because the survivors keep the
    slot's rank id in their peer arrays with ``active_ranks[slot] =
    0`` -- Mooncake's mask read protocol skips inactive slots
    without probing the (now-dead) endpoint.
    """
    if torch.cuda.is_available():
        t0 = time.monotonic()
        torch.cuda.synchronize()
        t_sync = time.monotonic() - t0
        torch.cuda.empty_cache()
        logger.info(
            "[Elastic EP] retiree_local_cleanup done (sync=%.3fs, "
            "empty_cache=%.3fs)",
            t_sync,
            time.monotonic() - t0 - t_sync,
        )
