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
    # Ranks a pending grow must reactivate (grow-into-retired-slot). Empty for append.
    pending_recover_ranks: List[int] = field(default_factory=list)
    # True once mask flips are committed; blocks reset() from clobbering Mooncake state.
    mask_dirty: bool = False

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
            self.mask_dirty = False

    def _set_active_bits(self, global_ranks: List[int], value: int) -> None:
        if self.active_ranks is None:
            return
        numel = self.active_ranks.numel()
        for g in global_ranks:
            if 0 <= g < numel:
                self.active_ranks[g] = value
        self.snapshot_active_to_last()
        self.sync_active_to_cpu()
        self.mask_dirty = True

    def activate_ranks(self, global_ranks: List[int]) -> None:
        self._set_active_bits(global_ranks, 1)

    def deactivate_ranks(self, global_ranks: List[int]) -> None:
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

            if server_args.elastic_ep_backend == "mooncake":
                cls._shrink_mooncake_fault_reconciliation_window()

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
    def _shrink_mooncake_fault_reconciliation_window(cls) -> None:
        # Mooncake PR #2455 buffers positive link events for 30s after any
        # negative event, stalling grow-into-retired-slot in ``_try_recover_world``.
        # Retire/regrow is locally sequenced here, so shrink to 1s. Process-local.
        try:
            from mooncake.pg import set_fault_reconciliation_window_us
        except ImportError:
            return
        set_fault_reconciliation_window_us(1_000_000)

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
        # Sync-reject invalid targets so admission gate can 4xx.
        if n <= 0:
            logger.warning("[Elastic EP] request_scale rejected: ep_size=%d must be > 0", n)
            return False
        if n == inst.effective_ep_size:
            logger.info("[Elastic EP] request_scale rejected: ep_size=%d is no-op", n)
            return False
        max_cap = inst.active_ranks_cpu.numel() if inst.active_ranks_cpu is not None else None
        if max_cap is not None and n > max_cap:
            logger.warning("[Elastic EP] request_scale rejected: ep_size=%d > max=%d", n, max_cap)
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
        cls._mark_phase("draining")

    @classmethod
    def mark_retiring(cls) -> None:
        cls._mark_phase("retiring")

    @classmethod
    def mark_reconfiguring(cls) -> None:
        cls._mark_phase("reconfiguring")

    @classmethod
    def _mark_phase(cls, phase: str) -> None:
        inst = cls._instance
        if inst is not None and inst.pending_ep_size is not None:
            inst.scale_phase = phase

    @classmethod
    def get_shrink_direction(cls) -> Optional[str]:
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
        inst = cls._instance
        if inst is None or inst.pending_ep_size is None or inst.pending_ep_size >= inst.effective_ep_size:
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
        # Skip reset() if mask has already been partially flipped (mirrors Mooncake ground truth).
        if not inst.mask_dirty:
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
        if cls._instance is not None:
            cls._instance.pending_recover_ranks = list(ranks)

    @classmethod
    def get_pending_recover_ranks(cls) -> List[int]:
        return list(cls._instance.pending_recover_ranks) if cls._instance is not None else []

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
# Catch-up windows: 5s on first check (same-tick fold), 200ms on FSM re-ticks.
_NIXL_RETIRE_STORE_BARRIER_CATCH_UP_FIRST_S = 5.0
_NIXL_RETIRE_STORE_BARRIER_CATCH_UP_FALLBACK_S = 0.2
_NIXL_RETIRE_CYCLE_ID_CATCH_UP_S = 5.0
_NIXL_RETIRE_ARRIVAL_COUNTER_KEY = "sglang_nixl_retire_arrival_counter"
_NIXL_RETIRE_CYCLE_KEY = "sglang_nixl_retire_cycle_counter"

# Per-namespace last-observed cycle id (NIXL retire barrier + WORLD retire barrier).
_last_local_cycle_id: dict[str, int] = {"nixl": 0, "world": 0}


def _barrier_target_from_effective_ep(rank: int, world_size: int, log_prefix: str) -> int:
    """Retire barrier target = pre-shrink effective_ep_size (chained shrinks like 4->3->2)."""
    try:
        effective = ElasticEPStateManager.get_effective_ep_size()
        if effective > 0:
            return effective
    except Exception as exc:
        logger.debug("%s rank=%d get_effective_ep_size failed (%s)", log_prefix, rank, exc)
    return world_size


def _rollback_arrival_counter(store, rank: int, key: str) -> None:
    try:
        store.set(key, "0")
    except Exception as exc:
        logger.debug("[Elastic EP][retire] rank=%d ARRIVAL reset on %s failed (%s)", rank, key, exc)


def _derive_epoch_via_store(
    store, rank: int, *,
    arrival_key: str, cycle_key: str, catch_up_s: float,
    cycle_ns: str, log_prefix: str,
) -> tuple[Optional[int], int]:
    """Elected-leader epoch derivation over shared TCPStore. Returns (epoch, arrival).
    epoch=None -> caller decides fallback and whether to rollback arrival_key.
    arrival=1 = leader, >=2 = follower."""
    try:
        arrival = int(store.add(arrival_key, 1))
    except Exception as exc:
        logger.debug("%s rank=%d arrival add failed (%s)", log_prefix, rank, exc)
        return None, 0
    if arrival == 1:
        try:
            epoch = int(store.add(cycle_key, 1))
            _last_local_cycle_id[cycle_ns] = epoch
            return epoch, arrival
        except Exception as exc:
            logger.debug("%s rank=%d cycle add failed (%s)", log_prefix, rank, exc)
            return None, arrival
    prev = _last_local_cycle_id[cycle_ns]
    deadline = time.monotonic() + catch_up_s
    while time.monotonic() < deadline:
        try:
            raw = store.get(cycle_key)
            candidate = int(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            candidate = prev
        if candidate > prev:
            _last_local_cycle_id[cycle_ns] = candidate
            return candidate, arrival
        time.sleep(0.05)
    logger.warning("%s rank=%d arrival=%d cycle id timeout %.1fs",
                   log_prefix, rank, arrival, catch_up_s)
    return None, arrival


@dataclass
class _NixlRetireBarrierState:
    """NIXL retire barrier handle (store-counter only, no Work)."""
    epoch: int
    world_size: int
    ready_key: str
    rank: int
    arrival: int
    posted_at: float
    first_check: bool = True


def _pre_nixl_retire(
    retiree_global_ranks: List[int],
    my_elastic_global_rank: Optional[int] = None,
) -> None:
    """Survivor-side NIXL peer disconnect before retire barrier. No-op on retiree."""
    from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

    if NixlEPBuffer._state().buffer is None or not torch.distributed.is_initialized():
        return
    my_rank = (my_elastic_global_rank if my_elastic_global_rank is not None
               else torch.distributed.get_rank())
    if my_rank in retiree_global_ranks:
        return
    t0 = time.monotonic()
    NixlEPBuffer.on_retire(retiree_global_ranks)
    logger.info("[Elastic EP][retire] rank=%d nixl on_retire took %.3fs",
                my_rank, time.monotonic() - t0)


def nixl_retire_barrier_post(
    retiree_global_ranks: List[int],
) -> Optional[_NixlRetireBarrierState]:
    """Post async NIXL retire barrier over the global TCPStore (elected-leader epoch)."""
    from sglang.srt.distributed.utils import get_global_tcp_store
    from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

    if NixlEPBuffer._state().buffer is None or not torch.distributed.is_initialized():
        return None

    my_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    barrier_target = _barrier_target_from_effective_ep(my_rank, world_size, "[Elastic EP][retire]")

    try:
        store = get_global_tcp_store()
    except Exception as exc:
        logger.warning("[Elastic EP][retire] rank=%d TCPStore lookup failed (%s)", my_rank, exc)
        return None
    if store is None:
        logger.warning("[Elastic EP][retire] rank=%d no TCPStore; skipping", my_rank)
        return None

    epoch, arrival = _derive_epoch_via_store(
        store, my_rank,
        arrival_key=_NIXL_RETIRE_ARRIVAL_COUNTER_KEY,
        cycle_key=_NIXL_RETIRE_CYCLE_KEY,
        catch_up_s=_NIXL_RETIRE_CYCLE_ID_CATCH_UP_S,
        cycle_ns="nixl",
        log_prefix="[Elastic EP][retire]",
    )
    if epoch is None:
        if arrival in (1, 2):
            _rollback_arrival_counter(store, my_rank, _NIXL_RETIRE_ARRIVAL_COUNTER_KEY)
        return None

    ready_key = f"sglang_nixl_retire_barrier_e{epoch}_posted"
    try:
        posted = int(store.add(ready_key, 1))
    except Exception as exc:
        logger.warning("[Elastic EP][retire] rank=%d ready_key add failed e=%d (%s)", my_rank, epoch, exc)
        if arrival == 1:
            try:
                store.add(_NIXL_RETIRE_CYCLE_KEY, -1)
            except Exception:
                pass
            _rollback_arrival_counter(store, my_rank, _NIXL_RETIRE_ARRIVAL_COUNTER_KEY)
            _last_local_cycle_id["nixl"] = epoch - 1
        return None

    logger.info("[Elastic EP][retire] rank=%d posted arr=%d e=%d %d/%d ws=%d",
                my_rank, arrival, epoch, posted, barrier_target, world_size)
    return _NixlRetireBarrierState(
        epoch=epoch, world_size=barrier_target, ready_key=ready_key,
        rank=my_rank, arrival=arrival, posted_at=time.monotonic(),
    )


def nixl_retire_barrier_check(state: Optional[_NixlRetireBarrierState]) -> bool:
    """Non-blocking probe: True when every cohort rank posted. TimeoutError at 300s."""
    if state is None:
        return True

    from sglang.srt.distributed.utils import get_global_tcp_store

    try:
        store = get_global_tcp_store()
    except Exception:
        store = None
    if store is None:
        if time.monotonic() - state.posted_at > _NIXL_RETIRE_STORE_BARRIER_TIMEOUT_S:
            raise TimeoutError(
                f"[Elastic EP][retire] rank={state.rank} store unavailable "
                f"{_NIXL_RETIRE_STORE_BARRIER_TIMEOUT_S:.0f}s at e={state.epoch}"
            )
        return False

    catch_up = (_NIXL_RETIRE_STORE_BARRIER_CATCH_UP_FIRST_S if state.first_check
                else _NIXL_RETIRE_STORE_BARRIER_CATCH_UP_FALLBACK_S)
    state.first_check = False
    deadline = time.monotonic() + catch_up
    last_seen: Optional[int] = None
    while True:
        try:
            raw = store.get(state.ready_key)
            count = int(raw.decode() if isinstance(raw, bytes) else raw)
            last_seen = count
            if count >= state.world_size:
                return True
        except Exception:
            pass
        if time.monotonic() >= deadline:
            break
        time.sleep(_NIXL_RETIRE_STORE_BARRIER_POLL_INTERVAL_S)

    if time.monotonic() - state.posted_at > _NIXL_RETIRE_STORE_BARRIER_TIMEOUT_S:
        raise TimeoutError(
            f"[Elastic EP][retire] rank={state.rank} e={state.epoch} timeout "
            f"{last_seen}/{state.world_size} @ {_NIXL_RETIRE_STORE_BARRIER_TIMEOUT_S:.0f}s"
        )
    return False


def nixl_retire_barrier_consume(state: Optional[_NixlRetireBarrierState]) -> None:
    """Finalize the barrier; only the leader resets ARRIVAL_KEY."""
    if state is None:
        return
    if state.arrival == 1:
        try:
            from sglang.srt.distributed.utils import get_global_tcp_store
            store = get_global_tcp_store()
            if store is not None:
                store.set(_NIXL_RETIRE_ARRIVAL_COUNTER_KEY, "0")
        except Exception as exc:
            logger.debug("[Elastic EP][retire] rank=%d ARRIVAL reset failed e=%d (%s)",
                         state.rank, state.epoch, exc)
    logger.info("[Elastic EP][retire] rank=%d consumed e=%d arr=%d wait=%.3fs",
                state.rank, state.epoch, state.arrival, time.monotonic() - state.posted_at)


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
    """Recover WORLD-scope Mooncake peers. include_subgroups also recovers _WORLD sub-PGs
    (recover-mode only; scale-up-v1 must pass False to avoid sub-PG ID mismatch)."""
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


def _activate(global_ranks: List[int], include_subgroups: bool) -> bool:
    if not _try_recover_world(global_ranks, include_subgroups=include_subgroups):
        return False
    inst = ElasticEPStateManager.instance()
    if inst is not None:
        inst.activate_ranks(global_ranks)
    for group in _iter_live_parallel_groups():
        _flip_active_rank_mask(group, global_ranks, value=1)
        _maybe_create_message_queue(group)
    _flip_world_backend_active_rank_mask(global_ranks, value=1)
    _refresh_ep_members()
    return True


def try_admit_scale_ranks(global_ranks: List[int]) -> bool:
    """Scale-up-v1 append (no sub-PG join)."""
    return _activate(global_ranks, include_subgroups=False)


def try_recover_ranks(global_ranks: List[int]) -> bool:
    """Recover ranks in WORLD + sub-PGs."""
    return _activate(global_ranks, include_subgroups=True)


def _join_world_group(*, include_subgroups: bool = False) -> None:
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
    """Scale-up-v1 append join (no sub-PG join)."""
    _join_world_group(include_subgroups=False)
    _refresh_ep_members()


def join_process_groups() -> None:
    """Recover-mode grow join (includes sub-PGs)."""
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


def _flip_mask(ranks: List[int], active_ranks, active_ranks_cpu, global_ranks, value: int) -> None:
    """Flip active_ranks[local_r] = value for every global rank in this group."""
    if active_ranks is None:
        return
    for lr in _map_global_to_group_local_ranks(ranks, global_ranks):
        active_ranks[lr] = value
        if active_ranks_cpu is not None:
            active_ranks_cpu[lr] = value


def _flip_world_backend_active_rank_mask(global_ranks: List[int], value: int) -> None:
    _flip_mask(parallel_state.get_world_backend_ranks(),
               parallel_state.get_world_backend_active_ranks(), None,
               global_ranks, value)


def _flip_active_rank_mask(group, global_ranks: List[int], value: int) -> None:
    _flip_mask(group.ranks, getattr(group, "active_ranks", None),
               getattr(group, "active_ranks_cpu", None), global_ranks, value)


def try_retire_ranks(global_ranks: List[int]) -> bool:
    """Retire ranks in WORLD + every sub-PG. Mooncake has no retire (direct in-place write)."""
    if not global_ranks:
        return True

    # NIXL retire handshake is driven asynchronously by ScaleDownStateMachine.NIXL_RETIRE.

    inst = ElasticEPStateManager.instance()
    if inst is not None:
        inst.deactivate_ranks(global_ranks)

    for group in _iter_live_parallel_groups():
        _flip_active_rank_mask(group, global_ranks, value=0)
    _flip_world_backend_active_rank_mask(global_ranks, value=0)
    _refresh_ep_members()
    logger.debug("[Elastic EP][retire] retire_ranks(%s) done", global_ranks)
    return True


@dataclass
class _RetireBarrierState:
    """Async retire barrier handle: async Work + TCPStore counter (Work.is_completed flaky)."""

    handle: Optional["torch.distributed.Work"]
    epoch: int
    world_size: int
    ready_key: str
    rank: int
    arrival: int = 0
    # False if ready_key increment failed; skip store fast-path (never reaches world_size).
    store_confirmed: bool = True
    # True when shared-store leader election failed; ARRIVAL_KEY needs a cleaner reset.
    used_fallback_epoch: bool = False
    # True when store fast-path observed count >= world_size (safe to swallow wait() errors).
    check_confirmed_via_store: bool = False


# Per-process fallback counter (only when shared TCPStore is unreachable).
_RETIRE_BARRIER_EPOCH = 0

# Shared TCPStore keys for elected-leader epoch derivation.
_RETIRE_BARRIER_ARRIVAL_COUNTER_KEY = "sglang_retire_barrier_arrival_counter"
_RETIRE_BARRIER_CYCLE_KEY = "sglang_retire_barrier_cycle_counter"
_RETIRE_BARRIER_CYCLE_ID_CATCH_UP_S = 5.0


def _retire_barrier_ready_key(epoch: int) -> str:
    return f"sglang_retire_barrier_e{epoch}_posted"


def retire_barrier_post() -> Optional[_RetireBarrierState]:
    """Async WORLD barrier posted before mask flip; poll via retire_barrier_check.

    Async so mlp_sync (separate PG) doesn't deadlock. Store-backed epoch is authoritative;
    handle.wait() only if store unreachable.
    """
    global _RETIRE_BARRIER_EPOCH
    if not torch.distributed.is_initialized():
        return None
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    barrier_target = _barrier_target_from_effective_ep(rank, world_size, "[Elastic EP][retire_barrier]")

    from sglang.srt.distributed.utils import get_global_tcp_store
    try:
        store = get_global_tcp_store()
    except Exception:
        store = None

    epoch, arrival, used_fallback_epoch = None, 0, False
    if store is not None:
        epoch, arrival = _derive_epoch_via_store(
            store, rank,
            arrival_key=_RETIRE_BARRIER_ARRIVAL_COUNTER_KEY,
            cycle_key=_RETIRE_BARRIER_CYCLE_KEY,
            catch_up_s=_RETIRE_BARRIER_CYCLE_ID_CATCH_UP_S,
            cycle_ns="world",
            log_prefix="[Elastic EP][retire_barrier]",
        )
        if epoch is None and arrival == 1:
            # Leader failed cycle add: undo ARRIVAL so next cycle isn't stuck.
            _rollback_arrival_counter(store, rank, _RETIRE_BARRIER_ARRIVAL_COUNTER_KEY)
    if epoch is None:
        used_fallback_epoch = True
        _RETIRE_BARRIER_EPOCH += 1
        epoch = _RETIRE_BARRIER_EPOCH

    logger.info("[Elastic EP][retire_barrier] rank=%d post async e=%d arr=%d", rank, epoch, arrival)
    handle = torch.distributed.barrier(group=torch.distributed.group.WORLD, async_op=True)

    ready_key = _retire_barrier_ready_key(epoch)
    store_confirmed = store is not None
    if store is not None:
        try:
            store.add(ready_key, 1)
        except Exception as exc:
            logger.warning("[Elastic EP][retire_barrier] rank=%d ready_key add fail e=%d (%s)",
                           rank, epoch, exc)
            store_confirmed = False

    return _RetireBarrierState(
        handle=handle, epoch=epoch, world_size=barrier_target,
        ready_key=ready_key, rank=rank, arrival=arrival,
        store_confirmed=store_confirmed, used_fallback_epoch=used_fallback_epoch,
    )


def retire_barrier_check(state: Optional[_RetireBarrierState]) -> bool:
    """Non-blocking probe: True when every rank posted (TCPStore fast-path)."""
    if state is None:
        return True
    if state.store_confirmed:
        try:
            from sglang.srt.distributed.utils import get_global_tcp_store
            store = get_global_tcp_store()
            if store is not None:
                raw = store.get(state.ready_key)
                if int(raw.decode() if isinstance(raw, bytes) else raw) >= state.world_size:
                    state.check_confirmed_via_store = True
                    return True
        except Exception:
            pass
    if state.handle is None:
        return True
    return state.handle.is_completed()


_RETIRE_BARRIER_CONSUME_TIMEOUT_S = 30.0


def retire_barrier_consume(state: Optional[_RetireBarrierState]) -> None:
    """Finalize the barrier after check()=True. Bounded wait() (Mooncake WORLD hang guard)."""
    if state is None:
        return
    t0 = time.monotonic()
    if state.handle is not None:
        try:
            state.handle.wait(timeout=timedelta(seconds=_RETIRE_BARRIER_CONSUME_TIMEOUT_S))
        except TypeError:
            state.handle.wait()
        except Exception as exc:
            elapsed = time.monotonic() - t0
            if state.check_confirmed_via_store:
                logger.warning("[Elastic EP][retire_barrier] rank=%d wait(e=%d) %.1fs (%s); store OK",
                               state.rank, state.epoch, elapsed, exc)
            else:
                raise RuntimeError(
                    f"[Elastic EP][retire_barrier] rank={state.rank} wait(e={state.epoch}) "
                    f"failed after {elapsed:.1f}s ({exc}); store NOT confirmed"
                ) from exc

    # Leader (arr==1) or fallback second-follower (arr==2) resets ARRIVAL_KEY; safe (lockstep).
    if state.arrival == 1 or (state.used_fallback_epoch and state.arrival == 2):
        try:
            from sglang.srt.distributed.utils import get_global_tcp_store
            store = get_global_tcp_store()
            if store is not None:
                store.set(_RETIRE_BARRIER_ARRIVAL_COUNTER_KEY, "0")
        except Exception as exc:
            logger.debug("[Elastic EP][retire_barrier] rank=%d ARRIVAL reset fail e=%d (%s)",
                         state.rank, state.epoch, exc)
    logger.info("[Elastic EP][retire_barrier] rank=%d consumed e=%d wait=%.3fs",
                state.rank, state.epoch, time.monotonic() - t0)


def retiree_local_cleanup() -> None:
    """CUDA quiesce before sys.exit(0) (no destroy_process_group; peers still live)."""
    if torch.cuda.is_available():
        t0 = time.monotonic()
        torch.cuda.synchronize()
        t_sync = time.monotonic() - t0
        torch.cuda.empty_cache()
        logger.info("[Elastic EP] retiree_local_cleanup sync=%.3fs empty=%.3fs",
                    t_sync, time.monotonic() - t0 - t_sync)
