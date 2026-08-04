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
        # admission gate 4xx's instead of stalling for the outer
        # elastic_ep_scale_timeout.
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
# Catch-up busy-poll windows for nixl_retire_barrier_check:
# _FIRST for the same-tick fold, _FALLBACK for FSM re-ticks.
_NIXL_RETIRE_STORE_BARRIER_CATCH_UP_FIRST_S = 5.0
_NIXL_RETIRE_STORE_BARRIER_CATCH_UP_FALLBACK_S = 0.2
# Bounded wait for the elected cycle leader to publish the cycle id.
_NIXL_RETIRE_CYCLE_ID_CATCH_UP_S = 5.0
_NIXL_RETIRE_ARRIVAL_COUNTER_KEY = "sglang_nixl_retire_arrival_counter"


def _barrier_target_from_effective_ep(
    rank: int, world_size: int, log_prefix: str
) -> int:
    """Retire-barrier target = effective_ep_size at retire boundary.

    Uses the pre-shrink cohort size (not launch world_size) so chained
    shrinks like 4->3->2 don't deadlock waiting on already-retired
    ranks. commit_scale runs strictly AFTER the barrier so this read
    is safe. Shared by both retire-barrier flavours."""
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
# Cohort-wide monotonic cycle counter on the shared TCPStore. Bumped
# once per retire boundary by the elected leader; replaces a legacy
# (arrival-1)//world_size derivation that split cohort ranks across
# epochs on chained shrinks like 4->3->2->1.
_NIXL_RETIRE_CYCLE_KEY = "sglang_nixl_retire_cycle_counter"

# Rank-local record of the most recent cycle id consumed; joiner
# subprocesses correctly start at 0.
_last_local_nixl_retire_cycle_id: int = 0


@dataclass
class _NixlRetireBarrierState:
    """Handle for the async NIXL retire store barrier.

    Store-counter-only (no torch.distributed.Work). arrival is this
    rank's within-cycle sequence number (1=leader); epoch is the
    cohort-wide cycle id (see nixl_retire_barrier_post)."""

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
    """Survivor-side NIXL peer disconnect before the retire barrier.

    Called once on DRAIN -> NIXL_RETIRE. Updates NixlEPBuffer's
    _connected_ep_size / _scale_to / _dispatch_ep_size to the new
    size so the lazy-disconnect on the next dispatch is a no-op.
    No-op on the retiree itself and when NIXL a2a is not active
    (Mooncake a2a uses EPBuffer.update_ep_member instead)."""
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

    Split from a blocking implementation into post/check/consume so the
    ScaleDownStateMachine.NIXL_RETIRE state can drive the rendezvous
    across scheduler ticks; a blocking wait would hold survivors out of
    the mlp_sync WORLD collective and deadlock on slow joiners.

    Uses the global TCPStore rather than a Gloo sub-group because
    dist.new_group's SHA1-salted subgroup names diverge between
    survivors and joiners with different PG-creation histories.

    Cohort-wide epoch is derived by ELECTED-LEADER on the shared store:
    every rank calls store.add(ARRIVAL_KEY, 1); the arrival=1 rank
    bumps a second cycle-key and publishes its value as the epoch;
    non-leaders busy-poll for it. consume() resets ARRIVAL_KEY so the
    next cycle re-elects a leader. Robust to variable cohort sizes.

    Returns None when NIXL is not active, torch.distributed is not
    initialized, or the global TCPStore is unavailable (FSM advances
    immediately on the next tick). Called on every rank on the DRAIN
    -> NIXL_RETIRE tick; survivors must _pre_nixl_retire first."""
    from sglang.srt.distributed.utils import get_global_tcp_store
    from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

    if NixlEPBuffer._state().buffer is None:
        return None

    if not torch.distributed.is_initialized():
        return None

    my_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Barrier target = pre-shrink cohort size for chained shrinks
    # (not the launch world_size); mirrors retire_barrier_post.
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
        # Cannot derive a cohort-consistent epoch; skip the barrier so
        # the FSM falls back to its outer scale timeout (bounded failure
        # beats indefinite epoch split). Leader rolls back ARRIVAL_KEY
        # so the next cycle can still elect a leader.
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
    """Non-blocking probe: True when every cohort rank has posted.

    Two-tier bounded catch-up busy-poll: ``_CATCH_UP_FIRST_S`` (5s) on
    the fold tick to close the sub-100ms post race without surrendering
    the tick; ``_CATCH_UP_FALLBACK_S`` (200ms) on subsequent ticks to
    catch counter updates racing an FSM re-tick without starving the
    scheduler on stuck cohorts.

    Always re-runs the poll (no one-shot flag) -- an earlier catch_up
    _used design skipped subsequent polls and deadlocked MC09.

    Enforces a 300s deadline; TimeoutError bubbles to the FSM which
    marks state FAILED. None state (no barrier posted) returns True."""
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
    """Finalize the NIXL retire barrier -- only the leader resets
    ARRIVAL_KEY so the next cycle re-elects a leader.

    Non-leader reset would race a fast peer's next-cycle store.add
    and re-elect a third rank on top of the peer's cycle, splitting
    the cohort across two epochs."""
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
    """Recover WORLD-scope Mooncake peers for rejoining ranks.

    include_subgroups (recover-mode only) also recovers the sglang
    _WORLD coordinator's device+cpu Mooncake sub-PGs so the survivor's
    next all_gather_object(cpu_group=...) doesn't EOFError. Scale-up-v1
    append passes include_subgroups=False -- the joiner's fresh _WORLD
    has different sub-PG IDs than the primary's launch-time PGs, so
    recovering the primary's sub-PGs against it deadlocks."""
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

    Admitted ranks live in [elastic_ep_initial_size, max_ep_size).
    Mooncake C++ mask is flipped inside _try_recover_world; this
    function flips the three Python-side mirrors (ElasticEPStateManager
    bitmap, sub-group masks, WORLD backend handle) so a subsequent
    scale-down does not read a stale active_ranks[joiner]=0.

    include_subgroups=False: scale-up-v1 joiners have fresh sub-PG IDs
    that would deadlock with recover on the primary's launch-time sub-PGs."""
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
    """Recover ranks in WORLD and flip active_ranks on every live PG.

    The retired rank re-establishes Mooncake connections via
    join_group(WORLD); it is sufficient here to flip active_ranks[r]=1
    on every live sub-group's Python mask and refresh MoE EP peers.
    Waiting on sub-group get_peer_state would deadlock (single-rank
    sub-groups on the tp=1 joiner never publish metadata).

    TODO: no cohort-scope barrier -- a slow survivor can still be on
    the pre-grow mask on WORLD when a fast survivor has re-broadcast."""
    # include_subgroups=True lets recover-mode joiners rebuild the
    # launch-time WORLD sub-PGs and complete the Mooncake handshake
    # so the next WORLD all_gather_object doesn't EOFError.
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

    Always joins torch.distributed.group.WORLD. include_subgroups=True
    (recover-mode only) also joins _WORLD.device_group / cpu_group so
    the survivor's next all_gather_object(cpu_group=...) doesn't
    EOFError. Scale-up-v1 append must pass include_subgroups=False
    (fresh joiner's sub-PGs have different Mooncake IDs)."""
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

    include_subgroups=False -- fresh joiner's sub-PGs have different
    Mooncake IDs than the primary's launch-time sub-PGs."""
    _join_world_group(include_subgroups=False)
    _refresh_ep_members()


def join_process_groups() -> None:
    """Rejoin WORLD + launch-time _WORLD sub-PGs after a recover-mode
    grow (include_subgroups=True; paired with survivor's
    try_recover_ranks to prevent EOFError on the next
    all_gather_object(cpu_group=...))."""
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

    Called collectively from every cohort rank after the drain barrier.
    Flips active_ranks[retiree]=0 on every live PG's mask + the top-level
    ElasticEPStateManager bitmap + the WORLD backend handle, then
    refreshes the MoE EP peer table.

    Mooncake has no retire primitive (only join/recover); the mask
    tensor is Python-owned and passed by reference into
    MooncakeBackendOptions, so a direct in-place write is sufficient.

    Retirees stay in WORLD's membership list; slots stay reserved for
    later try_recover_ranks reactivation."""
    if not global_ranks:
        return True

    # NIXL retire handshake is driven asynchronously by the FSM's
    # NIXL_RETIRE state (see ScaleDownStateMachine._tick_survivor);
    # doing it inline would block against mlp_sync.

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

    Wraps the async Work with a TCP-store posted counter (Mooncake
    WORLD's Work.is_completed() is flaky). arrival=1 marks the elected
    cycle leader (see retire_barrier_post); retire_barrier_consume uses
    it to gate the leader-only arrival-counter reset. arrival=0 marks
    a fallback cycle (store unavailable)."""

    handle: Optional["torch.distributed.Work"]
    epoch: int
    world_size: int
    ready_key: str
    rank: int
    arrival: int = 0


# Per-process fallback epoch counter, used only when the shared
# TCPStore is unreachable. WORLD async barrier is the true sync
# primitive, so this is correctness-safe for a single cohort;
# ex-joiners with lagging local counters need the shared-store fast
# path below.
_RETIRE_BARRIER_EPOCH = 0

# Shared TCPStore keys for elected-leader epoch derivation (mirror of
# the pattern in nixl_retire_barrier_post).
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

    Returns (epoch, arrival). epoch=None means fall back to the
    per-process counter (store unavailable / leader-window timeout).
    arrival=1 is the elected leader, >=2 followers.

    Shared derivation is required for cohorts with ex-joiners whose
    per-process counters lag; without it the fast-path in
    retire_barrier_check silently disables on the very configuration
    that most needs the Mooncake is_completed() flake guard."""
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

    Called on every rank BEFORE the mask flip in try_retire_ranks;
    caller polls retire_barrier_check until all ranks have posted.

    Async because a blocking WORLD barrier deadlocks against mlp_sync
    (WORLD and tp_cpu_group are separate PGs); async lets the event
    loop keep servicing per-tick mlp_sync all-gathers while the
    barrier drains.

    A per-epoch TCPStore counter provides a reliable across-Mooncake
    completion signal (WORLD backend's is_completed() is flaky).
    Epoch is derived via elected-leader on the store so ex-joiners
    with a lagging subprocess-local counter still agree on the
    ready_key. Store-unreachable falls back to handle.wait().

    Invariants: all in-flight WORLD collectives complete before mask
    flip, and no rank flips active_ranks[retiree]=0 while another rank
    still expects the retiree in a prior collective."""
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
    """Non-blocking probe: True when every rank has posted the barrier.

    Checks the TCPStore atomic counter first (reliable across Mooncake
    WORLD's flaky is_completed()), then falls back to is_completed()
    if the store is unavailable."""
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
    """Finalize the retire barrier's Work handle after check() = True.

    handle.wait() should return immediately (all ranks have posted).
    Enforces a bounded timeout because Mooncake WORLD's wait() has
    been observed to hang forever if a peer exited mid-flight; a
    timeout warns + proceeds since the store counter already
    confirmed every rank posted."""
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
            # Backend doesn't accept the timeout kwarg; store counter
            # already confirmed the barrier so this returns promptly.
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
    # Leader-only arrival-counter reset -- letting followers reset
    # would race a fast peer's next-cycle store.add and re-elect a
    # third rank as leader on top of the peer's cycle. arrival=0
    # cycles (fallback path) never touched the store.
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


def retiree_local_cleanup() -> None:
    """Local CUDA quiesce, run by retirees just before os._exit(0).

    Only torch.cuda.synchronize() + empty_cache(). Deliberately does
    NOT call destroy_process_group() or tear down NIXL/Mooncake
    endpoints: those would block on live peer collectives. Survivors
    have already dropped RDMA state via EPBuffer.on_retire(); kernel
    exit reclaims the rest. The slot stays available for later
    recover_ranks (survivors keep active_ranks[slot]=0)."""
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
