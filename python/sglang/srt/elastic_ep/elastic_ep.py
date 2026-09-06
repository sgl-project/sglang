from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Union

import torch

from sglang.srt.distributed import get_world_group, parallel_state
from sglang.srt.distributed.utils import get_global_tcp_store
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda

if TYPE_CHECKING:
    from sglang.srt.eplb.eplb_manager import EPLBManager

logger = logging.getLogger(__name__)

_SCALE_COHORT_KEY_PREFIX = "elastic_ep/scale_cohort"

# How long a grow stays in warming_up without serving. Sized over the prefill-shape
# DeepGEMM JIT a fresh rank compiles on its first forward (tens of seconds, and worse
# when concurrent jobs share DG_JIT_CACHE_DIR, whose per-kernel lock serializes them).
_WARMUP_SETTLE_TIMEOUT_S = 60.0


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
    # Deadline for the post-grow warmup window; see _WARMUP_SETTLE_TIMEOUT_S.
    warmup_deadline: Optional[float] = None

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

        if get_exec().moe.elastic_ep_backend is not None:
            world_size = torch.distributed.get_world_size()
            active_rank_capacity = get_parallel().max_ep_size or world_size
            assert active_rank_capacity >= world_size, (
                f"--max-ep-size ({active_rank_capacity}) must be >= "
                f"world_size ({world_size})."
            )

            if server_args.elastic_ep_backend == "mooncake":
                cls._require_graceful_membership()

            inst = cls._build_state(ep_size=active_rank_capacity, device=None)
            inst.effective_ep_size = world_size
            inst.original_ep_size = world_size
            if active_rank_capacity > world_size:
                inst.active_ranks[world_size:].zero_()
                inst.snapshot_active_to_last()
                inst.sync_active_to_cpu()

            if get_exec().moe.moe_a2a_backend == "nixl":
                cls._on_scale = cls._on_scale_nixl

            inst.ep_join_rank_offset = get_parallel().ep_join_rank_offset
            if server_args.is_ep_joiner:
                cls._init_joiner_state(inst, server_args)

            cls._instance = inst

        return cls._instance

    @classmethod
    def _require_graceful_membership(cls) -> None:
        # A planned shrink without this looks like a link fault (~10s/peer timeout).
        try:
            from mooncake.pg import deactivate_ranks  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--elastic-ep-backend mooncake requires a mooncake build providing "
                "mooncake.pg.deactivate_ranks; please upgrade mooncake."
            ) from exc

    @classmethod
    def _init_joiner_state(cls, inst: ElasticEPState, server_args: ServerArgs) -> None:
        global_rank = torch.distributed.get_rank()
        inst.active_ranks.zero_()
        inst.active_ranks[global_rank] = 1
        inst.snapshot_active_to_last()
        inst.sync_active_to_cpu()

        if get_exec().moe.ep_join_mode == "scale":
            inst.effective_ep_size = (
                get_parallel().ep_join_rank_offset + get_parallel().tp_size
            )
            inst.original_ep_size = (
                get_parallel().elastic_ep_initial_size
                or get_parallel().ep_join_rank_offset
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
        # Bounds, feasibility and the no-op target are the caller's to reject: it can
        # answer the client with a reason, and its bound is never looser than ours.
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
        cls.mark_phase("joining")

    @classmethod
    def mark_configuring_data_plane(cls) -> None:
        cls.mark_phase("configuring_data_plane")

    @classmethod
    def mark_syncing_new_world(cls) -> None:
        cls.mark_phase("syncing_new_world")

    @classmethod
    def mark_phase(cls, phase: str) -> None:
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
    def commit_scale(cls) -> None:
        inst = cls._instance
        if inst is None or inst.pending_ep_size is None:
            return
        direction = cls.get_shrink_direction()
        inst.effective_ep_size = inst.pending_ep_size
        inst.pending_ep_size = None
        inst.has_scaled = True
        if direction == "shrink":
            inst.scale_phase = "serving_shrunk"
        else:
            cls.mark_warming_up()
        inst.last_error = None
        inst.pending_since = None
        inst.pending_recover_ranks = []
        inst.reset()

    @classmethod
    def mark_warming_up(cls) -> None:
        inst = cls._instance
        if inst is not None:
            inst.scale_phase = "warming_up"
            inst.warmup_deadline = time.monotonic() + _WARMUP_SETTLE_TIMEOUT_S

    @classmethod
    def settle_warmup(cls, *, served: bool) -> None:
        """Leave warming_up once the cohort has served, or when the window expires.

        A joiner cannot warm itself: its remaining startup cost is prefill-shape JIT
        that only compiles when a kernel runs, and in EP that needs the a2a with its
        peers. So one completed forward at the new width proves it warm -- dispatch is
        collective, so the forward could not have returned without it. The deadline
        covers the idle cohort, where no forward is coming and waiting on one would
        strand a caller that gates traffic on this phase.
        """
        inst = cls._instance
        if inst is None or inst.scale_phase != "warming_up":
            return
        if served or time.monotonic() >= inst.warmup_deadline:
            inst.scale_phase = "serving_expanded"
            inst.warmup_deadline = None

    @classmethod
    def fail_scale(cls, error: str) -> None:
        inst = cls._instance
        if inst is None:
            return
        inst.pending_ep_size = None
        inst.scale_phase = "failed"
        inst.warmup_deadline = None
        inst.last_error = error
        inst.pending_since = None
        inst.pending_recover_ranks = []
        # Skip reset() if mask has already been partially flipped (mirrors Mooncake ground truth).
        if not inst.mask_dirty:
            inst.reset()
        elif inst.active_ranks_cpu is not None:
            # The flip is ground truth and the retirees are already gone, so trust it
            # over the uncommitted width: barrier targets come from effective_ep_size,
            # and leaving it pre-shrink makes every later cohort rendezvous wait on
            # departed ranks forever.
            active_count = int(inst.active_ranks_cpu.sum().item())
            if 0 < active_count != inst.effective_ep_size:
                logger.warning(
                    "[Elastic EP] failed mid-flip; reconciling effective_ep_size %d -> %d "
                    "to the active mask",
                    inst.effective_ep_size,
                    active_count,
                )
                inst.effective_ep_size = active_count

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
        inst = cls._instance
        return list(inst.pending_recover_ranks) if inst is not None else []

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
        # Membership is committed here, but a grown cohort cannot serve at full speed
        # until the joiner has warmed; settle_warmup() closes this on the first
        # forward or at the deadline.
        if inst.scale_phase == "warming_up":
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


# Shared across all three store barriers (NIXL retire, WORLD retire, scale_ready).
_BARRIER_STORE_POLL_S = 0.05
_BARRIER_EPOCH_CATCH_UP_S = 5.0
_RETIRE_BARRIER_TIMEOUT_S = 300.0
_BARRIER_RECHECK_S = 0.2

# TCPStore stands in for dist.barrier(WORLD): Mooncake's bitmap lags our active_ranks
# flip while it digests retirees' link events. Never rename these keys.
_BARRIER_NS: dict[str, tuple[str, str, str, str]] = {
    # ns: (arrival counter key, cycle counter key, ready key format, log prefix)
    "nixl": (
        "sglang_nixl_retire_arrival_counter",
        "sglang_nixl_retire_cycle_counter",
        "sglang_nixl_retire_barrier_e{}_posted",
        "[Elastic EP][retire]",
    ),
    "world": (
        "sglang_retire_barrier_arrival_counter",
        "sglang_retire_barrier_cycle_counter",
        "sglang_retire_barrier_e{}_posted",
        "[Elastic EP][retire_barrier]",
    ),
    "scale_ready": (
        "sglang_scale_ready_arrival_counter",
        "sglang_scale_ready_cycle_counter",
        "sglang_scale_ready_e{}_posted",
        "[Elastic EP][scale_ready]",
    ),
}
_last_local_cycle_id: dict[str, int] = {ns: 0 for ns in _BARRIER_NS}

_EXPERT_MAP_INBOX_KEY = "sglang_expert_map_to_r{}"


def clear_expert_map_inbox(group_rank: int) -> None:
    """Drop residue a previous occupant of this slot never consumed, which would
    otherwise read as this scale's map. Only safe ahead of the cohort announce,
    past which the source may already have written."""
    store = _store_or_none("[Elastic EP][expert map]")
    if store is not None:
        store.delete_key(_EXPERT_MAP_INBOX_KEY.format(group_rank))


def share_expert_map_via_store(
    tensor: torch.Tensor, *, is_src: bool, cohort_size: int, group_rank: int
) -> bool:
    """Hand the expert location map to the cohort over the store, one inbox per
    reader. False = no store, and the caller falls back to the collective.

    Not a broadcast on the group: Mooncake drives a device broadcast through
    putTaskCuda, which synchronizes the stream from inside the collective, so a
    cohort part-way through a grow wedges there until the watchdog fires. Its host
    path copies device-to-device and rejects host memory outright.

    An inbox rather than a shared round: any round derived from a participant count
    desyncs the moment the cohort changes width, which here is every call.
    """
    store = _store_or_none("[Elastic EP][expert map]")
    if store is None or cohort_size <= 1:
        return False
    if is_src:
        blob = tensor.cpu().numpy().tobytes()
        for peer in range(cohort_size):
            if peer != group_rank:
                store.set(_EXPERT_MAP_INBOX_KEY.format(peer), blob)
        return True
    key = _EXPERT_MAP_INBOX_KEY.format(group_rank)
    staged = torch.frombuffer(bytearray(store.get(key)), dtype=tensor.dtype)
    # Leave it empty so the next round blocks rather than being served this one.
    store.delete_key(key)
    tensor.copy_(staged.view(tensor.shape))
    return True


def seed_barrier_epochs() -> None:
    """Adopt the cohort's current cycle ids. Call on a joiner once it is admitted.

    A follower takes the first cycle id above the last one it used, and a process new
    to a namespace has no floor -- so it accepts a round that already closed, whose
    ready key is still satisfied, and crosses alone while the cohort waits in the
    round in flight. Seed at admission: by barrier time the mint may have landed.

    ``add(key, 0)`` reads without creating a value the electing path reads as a round.
    """
    store = _store_or_none("[Elastic EP][barrier seed]")
    if store is None:
        return
    for ns, (_, cycle_key, _, log) in _BARRIER_NS.items():
        try:
            _last_local_cycle_id[ns] = int(store.add(cycle_key, 0))
        except Exception as exc:
            logger.warning("%s cycle id seed failed (%s)", log, exc)


class _BarrierSkip:
    """No barrier needed. Distinct from None (post failed), which must not flip masks."""


BARRIER_SKIP = _BarrierSkip()


def _store_int(store, key: str) -> int:
    raw = store.get(key)
    return int(raw.decode() if isinstance(raw, bytes) else raw)


def _store_or_none(what: Optional[str] = None):
    """Global TCPStore or None; the lookup itself can raise. ``what`` = warning prefix."""
    try:
        store = get_global_tcp_store()
    except Exception as exc:
        if what:
            logger.warning("%s TCPStore lookup failed (%s)", what, exc)
        return None
    if store is None and what:
        logger.warning("%s no TCPStore", what)
    return store


def _barrier_target_from_effective_ep(world_size: int) -> int:
    """Retire barrier target = pre-shrink effective_ep_size (chained shrinks 4->3->2)."""
    try:
        effective = ElasticEPStateManager.get_effective_ep_size()
        if effective > 0:
            return effective
    except Exception:
        pass
    return world_size


@dataclass
class _StoreBarrier:
    """One in-flight store barrier.

    The epoch is elected rather than computed: arrival 1 mints the next cycle id and
    later arrivals wait to read it. Chained scales therefore cannot land on one key.
    """

    ns: str
    rank: int
    epoch: int
    world_size: int
    ready_key: str
    arrival: int = 0
    posted_at: float = 0.0
    first_check: bool = True
    last_poll: float = 0.0

    @property
    def tag(self) -> str:
        return _BARRIER_NS[self.ns][3]

    @classmethod
    def post(
        cls, store, rank: int, ns: str, target: int, *, rewind_on_fail: bool = False
    ) -> Optional[_StoreBarrier]:
        """Elect an epoch, then announce this rank on its ready key. ``rewind_on_fail``
        frees the leader's epoch on a failed announce; a sync driver must not reuse a
        live key."""
        epoch, arrival = cls._elect_epoch(store, rank, ns)
        if epoch is None:
            return None
        ready_key = _BARRIER_NS[ns][2].format(epoch)
        try:
            store.add(ready_key, 1)
        except Exception as exc:
            logger.warning(
                "%s rank=%d ready_key add fail e=%d (%s)",
                _BARRIER_NS[ns][3],
                rank,
                epoch,
                exc,
            )
            if rewind_on_fail and arrival == 1:
                try:
                    store.add(_BARRIER_NS[ns][1], -1)
                except Exception:
                    pass
                _last_local_cycle_id[ns] = epoch - 1
            # Always unwind: ARRIVAL must return to a base where a later cycle reads 1.
            cls._unwind(store, rank, ns)
            return None
        return cls(ns, rank, epoch, target, ready_key, arrival, time.monotonic())

    def check(self, store, secs: float) -> tuple[bool, int]:
        """Poll ready key to target or ``secs`` -> (reached, count); secs=0 probes once."""
        deadline = time.monotonic() + secs
        seen = 0
        while True:
            try:
                seen = _store_int(store, self.ready_key)
                if seen >= self.world_size:
                    return True, seen
            except Exception:
                pass
            if time.monotonic() >= deadline:
                return False, seen
            time.sleep(_BARRIER_STORE_POLL_S)

    def consume(self) -> None:
        """Leader-only ARRIVAL reset, so the next cycle's first arrival reads 1 again.
        Warned, not swallowed: a counter left high costs elasticity for the process."""
        if self.arrival != 1:
            return
        try:
            store = get_global_tcp_store()
            if store is not None:
                store.set(_BARRIER_NS[self.ns][0], "0")
        except Exception as exc:
            logger.warning(
                "%s rank=%d ARRIVAL reset failed e=%d (%s)",
                self.tag,
                self.rank,
                self.epoch,
                exc,
            )

    @staticmethod
    def _unwind(store, rank: int, ns: str) -> None:
        """Undo this arrival. Decrement: set(0) stomps peers, minting a 2nd leader."""
        try:
            store.add(_BARRIER_NS[ns][0], -1)
        except Exception:
            pass

    @classmethod
    def _elect_epoch(cls, store, rank: int, ns: str) -> tuple[Optional[int], int]:
        """arrival 1 = leader and mints the cycle id, >=2 = follower and reads it.
        On epoch=None our ARRIVAL is already unwound; the caller only falls back."""
        arrival_key, cycle_key, _, log = _BARRIER_NS[ns]
        try:
            arrival = int(store.add(arrival_key, 1))
        except Exception as exc:
            logger.warning("%s rank=%d arrival add failed (%s)", log, rank, exc)
            return None, 0
        if arrival == 1:
            try:
                epoch = int(store.add(cycle_key, 1))
                _last_local_cycle_id[ns] = epoch
                return epoch, arrival
            except Exception as exc:
                logger.warning("%s rank=%d cycle add failed (%s)", log, rank, exc)
        else:
            prev = _last_local_cycle_id[ns]
            deadline = time.monotonic() + _BARRIER_EPOCH_CATCH_UP_S
            while time.monotonic() < deadline:
                try:
                    candidate = _store_int(store, cycle_key)
                except Exception:
                    candidate = prev
                if candidate > prev:
                    _last_local_cycle_id[ns] = candidate
                    return candidate, arrival
                time.sleep(_BARRIER_STORE_POLL_S)
            logger.warning("%s rank=%d arrival=%d cycle id timeout", log, rank, arrival)
        cls._unwind(store, rank, ns)
        return None, arrival


_BarrierHandle = Union[_StoreBarrier, _BarrierSkip, None]


def retire_barrier_check(
    state: _BarrierHandle, *, block_s: Optional[float] = None
) -> bool:
    """Non-blocking probe: True when every cohort rank posted. TimeoutError at 300s.
    Namespace-agnostic -- the handle knows which barrier it belongs to.

    ``block_s`` waits in-place instead of across ticks, for a caller that must not
    re-enter its event loop while peers may already have left the collectives it
    would post there."""
    if isinstance(state, _BarrierSkip):
        return True
    if state is None:
        return False  # post failed: fail closed so the FSM re-posts

    store = _store_or_none()
    seen = "no store"
    if store is not None:
        now = time.monotonic()
        # Full catch-up on the same-tick fold, then probe without blocking and
        # throttle by wall clock instead. Sleeping 200ms per re-tick is free on an
        # idle cohort but serializes into the forward loop of a rank still draining
        # a request, which is when this barrier is held longest: it cost a 7x decode
        # slowdown and pushed a legitimate drain past the barrier timeout.
        if block_s is not None:
            state.first_check = False
            window = block_s
        elif state.first_check:
            state.first_check = False
            window = _BARRIER_EPOCH_CATCH_UP_S
        elif now - state.last_poll < _BARRIER_RECHECK_S:
            return False
        else:
            window = 0.0
        state.last_poll = now
        reached, seen = state.check(store, window)
        if reached:
            return True

    if time.monotonic() - state.posted_at > _RETIRE_BARRIER_TIMEOUT_S:
        # Hand ARRIVAL back before giving up, or no later cycle can elect a leader.
        state.consume()
        raise TimeoutError(
            f"{state.tag} rank={state.rank} e={state.epoch} timeout "
            f"{seen}/{state.world_size} @ {_RETIRE_BARRIER_TIMEOUT_S:.0f}s"
        )
    return False


def retire_barrier_consume(state: _BarrierHandle) -> None:
    """Finalize the barrier; only the leader resets ARRIVAL_KEY."""
    if state is None or isinstance(state, _BarrierSkip):
        return
    state.consume()


def _pre_nixl_retire(
    retiree_global_ranks: List[int], my_elastic_global_rank: int
) -> None:
    """Survivor-side NIXL peer disconnect before retire barrier. No-op on retiree.
    Takes the elastic global rank: an offset joiner's torch rank differs from it."""
    from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

    if NixlEPBuffer._state().buffer is None or not torch.distributed.is_initialized():
        return
    if my_elastic_global_rank in retiree_global_ranks:
        return
    NixlEPBuffer.on_retire(retiree_global_ranks)


def nixl_retire_barrier_post() -> _BarrierHandle:
    """Post async NIXL retire barrier over the global TCPStore (elected-leader epoch).
    BARRIER_SKIP = nothing to synchronize; None = post failed, caller must retry."""
    # Keyed off the backend, not off whether this rank built its buffer yet: that is
    # lazy on first dispatch, so a joiner whose cohort shrinks before it serves has
    # none. Skipping is a cohort decision -- one rank opting out crosses without
    # arriving and leaves the rest at the barrier.
    if (
        not torch.distributed.is_initialized()
        or get_exec().moe.moe_a2a_backend != "nixl"
    ):
        return BARRIER_SKIP

    my_rank = torch.distributed.get_rank()
    target = _barrier_target_from_effective_ep(torch.distributed.get_world_size())

    store = _store_or_none(f"[Elastic EP][retire] rank={my_rank}")
    if store is None:
        return None
    return _StoreBarrier.post(store, my_rank, "nixl", target, rewind_on_fail=True)


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


def _is_mooncake_pg(pg) -> bool:
    try:
        return torch.distributed.get_backend(pg) in ("mooncake", "mooncake-cpu")
    except Exception:
        return False


def _lowest_survivor(retiring: set) -> int:
    """Lowest active rank outside ``retiring``, from the local mask (no collective)."""
    active_cpu = getattr(ElasticEPStateManager.instance(), "active_ranks_cpu", None)
    if active_cpu is None:
        return 0
    return next((g for g, a in enumerate(active_cpu) if a and g not in retiring), 0)


def _maybe_create_message_queue(group) -> None:
    if not group.use_message_queue_broadcaster or group.world_size <= 1:
        return

    from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue

    group.mq_broadcaster = MessageQueue.create_from_process_group(
        group.cpu_group, 1 << 22, 6
    )


def mooncake_all_reduce_transient_retry(
    tensor, *, op, group, total_budget_s: float = 30.0
) -> None:
    """WORLD all_reduce retrying Mooncake's transient ``rank is not active in this group``:
    its bitmap lags our ``active_ranks`` flip by seconds after a scale. Others raise."""
    deadline = time.monotonic() + total_budget_s
    last = None
    i = 0
    while True:
        try:
            torch.distributed.all_reduce(tensor, op=op, group=group)
            return
        except RuntimeError as exc:
            msg = str(exc)
            if "invalid state" not in msg or "rank is not active" not in msg:
                raise
            last = exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep_s = min(0.05 * (1 << i), 2.0, remaining)
        time.sleep(sleep_s)
        i += 1
    raise RuntimeError(
        f"Mooncake WORLD all_reduce still transient after {total_budget_s:.1f}s: {last}"
    )


def mooncake_world_settle_probe(*, total_budget_s: float = 30.0) -> None:
    """WORLD all_reduce off the hot path so Mooncake's bitmap converges before the next
    tick. Warn-only (aborting strands ``effective_ep_size``); dp_attn retries anyway."""
    if not torch.distributed.is_initialized():
        return
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    probe = torch.zeros(1, dtype=torch.int64, device=device)
    try:
        mooncake_all_reduce_transient_retry(
            probe,
            op=torch.distributed.ReduceOp.SUM,
            group=None,  # WORLD
            total_budget_s=total_budget_s,
        )
    except RuntimeError as exc:
        logger.warning("[Elastic EP] probe timeout %.1fs: %s", total_budget_s, exc)


def scale_ready_barrier_via_store(target_size: int, *, timeout_s: float = 60.0) -> None:
    """Post-scale WORLD sync via TCPStore, Mooncake-free. ``target_size`` = live ranks."""
    if not torch.distributed.is_initialized():
        return

    rank = torch.distributed.get_rank()
    store = _store_or_none(f"[Elastic EP][scale_ready] rank={rank}")
    if store is None:
        return

    state = _StoreBarrier.post(store, rank, "scale_ready", target_size)
    if state is None:
        logger.warning("[Elastic EP][scale_ready] rank=%d post failed; skipping", rank)
        return

    reached, count = state.check(store, timeout_s)
    # Consume before raising: a timed-out leader still owes ARRIVAL, else a 2nd leader.
    state.consume()
    if not reached:
        raise RuntimeError(
            f"[Elastic EP][scale_ready] rank={rank} e={state.epoch} timeout after "
            f"{timeout_s}s (count={count} / target={target_size})"
        )


def cohort_vote_via_store(
    ok: bool, cohort_size: int, *, tag: str, timeout_s: float = 60.0
) -> bool:
    """Unanimous go/no-go over the TCPStore. Not a device all_reduce(MIN): a departed
    slot reduces in as a zero, so the fastest rank reads a "no" nobody voted while the
    rest block forever. Keyed per width, else consecutive shrinks split a round."""
    if cohort_size <= 1:
        return ok
    store = _store_or_none(f"[Elastic EP][vote] {tag}")
    if store is None:
        return ok

    ns = f"sglang_cohort_vote_{tag}_n{cohort_size}"
    seq_key, cast_key = f"{ns}_seq", f"{ns}_cast"
    rnd = (int(store.add(seq_key, 1)) - 1) // cohort_size
    yes_key = f"{ns}_r{rnd}_yes"
    # Tally before announcing, or the last arrival reads a tally a peer has yet to add to.
    store.add(yes_key, 1 if ok else 0)
    store.add(cast_key, 1)

    target = (rnd + 1) * cohort_size
    deadline = time.monotonic() + timeout_s
    while (cast := _store_int(store, cast_key)) < target:
        if time.monotonic() >= deadline:
            # Realign to this round's boundary: a short round (peer died before voting)
            # offsets every later vote at this width. Idempotent: all writers agree.
            for key in (seq_key, cast_key):
                try:
                    store.set(key, str(target))
                except Exception:
                    pass
            raise RuntimeError(
                f"[Elastic EP][vote] {tag} r{rnd} timeout after {timeout_s}s "
                f"({cast}/{target} voted)"
            )
        time.sleep(_BARRIER_STORE_POLL_S)
    return _store_int(store, yes_key) == cohort_size


def _try_recover_world(
    global_ranks: List[int], *, include_subgroups: bool = False
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
            # Do not gate on get_peer_state: the joiner blocks in join_group (deadlock).
            deadline = time.monotonic() + _WORLD_RECOVER_WAIT_TIMEOUT_S
            while True:
                try:
                    recover_ranks(pg, global_ranks)
                    break
                except Exception as exc:
                    if time.monotonic() > deadline:
                        logger.warning(
                            "[Elastic EP][recover] admit %s to sub-PG failed: %s",
                            global_ranks,
                            exc,
                        )
                        return False
                    time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)
    return True


def _activate(global_ranks: List[int], include_subgroups: bool) -> bool:
    if not _try_recover_world(global_ranks, include_subgroups=include_subgroups):
        return False
    inst = ElasticEPStateManager.instance()
    if inst is not None:
        inst.activate_ranks(global_ranks)
    for group in _iter_live_parallel_groups():
        _flip_active_rank_mask(global_ranks, 1, group)
        # Only where a rank actually came back. Rebuilding elsewhere drops a live
        # queue's shm segment for nothing and adds a blocking collective, with no
        # timeout, to the scale path -- GroupCoordinator skips the same work at
        # construction for a recovered rank precisely so this can do it here.
        if _map_global_to_group_local_ranks(group.ranks, global_ranks):
            _maybe_create_message_queue(group)
    _flip_active_rank_mask(global_ranks, 1)
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


def _flip_active_rank_mask(global_ranks: List[int], value: int, group=None) -> None:
    """Flip mask bits for ``global_ranks``. ``group=None`` targets the WORLD backend,
    which carries no CPU mirror of the mask."""
    if group is None:
        ranks = parallel_state.get_world_backend_ranks()
        active = parallel_state.get_world_backend_active_ranks()
        active_cpu = None
    else:
        ranks = group.ranks
        active = getattr(group, "active_ranks", None)
        active_cpu = getattr(group, "active_ranks_cpu", None)
    if active is None:
        return
    for lr in _map_global_to_group_local_ranks(ranks, global_ranks):
        active[lr] = value
        if active_cpu is not None:
            active_cpu[lr] = value


def _mooncake_membership_targets(global_ranks: List[int]) -> List[tuple]:
    """Live Mooncake groups holding ``global_ranks``, mapped to member indices."""
    targets = []
    world_backend = torch.distributed.group.WORLD
    world_ranks = parallel_state.get_world_backend_ranks()
    if world_ranks and _is_mooncake_pg(world_backend):
        local = _map_global_to_group_local_ranks(world_ranks, global_ranks)
        if local:
            targets.append(("WORLD", world_backend, local))
    for group in _iter_live_parallel_groups():
        local = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not local:
            continue
        for pg in (group.device_group, group.cpu_group):
            if pg is not None and pg is not world_backend and _is_mooncake_pg(pg):
                targets.append((group.unique_name, pg, local))
    return targets


# 60s, not 15s: a retiree that drained a request right up to the barrier still has
# queued GPU work and live RDMA slots, so its deactivation lands well after one that
# was already idle. 15s was sized on the idle case and expired on the busy one.
def await_retirees_departed(global_ranks: List[int], *, budget_s: float = 60.0) -> None:
    """Hold the first post-flip device collective until Mooncake drops the retirees: our
    flip is local, so a collective posted before the retiree proposes its departure still
    expects it and Mooncake's spin-wait kernel pegs the GPU with no fault to break out.
    Must be ``get_peer_state``; ``get_active_ranks`` only reads back our own flip."""
    if not global_ranks or not torch.distributed.is_initialized():
        return
    targets = _mooncake_membership_targets(global_ranks)
    if not targets:
        return

    from mooncake.pg import get_peer_state

    deadline = time.monotonic() + budget_s
    for label, pg, local in targets:
        while True:
            try:
                if not any(get_peer_state(pg, local)):
                    break
            except Exception as exc:
                logger.warning(
                    "[Elastic EP][retire] %s get_peer_state failed: %s", label, exc
                )
                break
            if time.monotonic() >= deadline:
                # Raise, do not proceed. Posting the collective anyway is the exact
                # case this function exists to prevent, and its cost is the spin-wait
                # above: no fault, no timeout, a pegged GPU until the watchdog. A
                # failed scale reconciles the width and keeps serving instead.
                raise RuntimeError(
                    f"[Elastic EP][retire] {label} still holds {local} after "
                    f"{budget_s:.1f}s; refusing to post a collective it still expects"
                )
            time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)


def _mooncake_deactivate_self() -> None:
    """Announce our own departure, mirroring ``recover_ranks`` on grow. The flip alone
    never reaches the coordinator, so a shrink looks like a link fault (~10s p2p timeout,
    rejoin queued behind it), and ``joinGroup`` needs the slot left inactive. Self-only,
    on the way out only: a live event loop keeps broadcasting once its view drops it."""
    from mooncake.pg import deactivate_ranks

    if not torch.distributed.is_initialized():
        return

    my_rank = torch.distributed.get_rank()
    targets = _mooncake_membership_targets([my_rank])
    if not targets:
        return

    for label, pg, local in targets:
        try:
            # A Rejected proposal is not an error here: a peer already carried the
            # same deactivation, which is the outcome we wanted either way.
            deactivate_ranks(pg, local)
        except Exception as exc:
            logger.warning(
                "[Elastic EP][retire] deactivate %s(%s) failed: %s", label, local, exc
            )


def try_retire_ranks(global_ranks: List[int]) -> None:
    """Retire ranks in WORLD + every sub-PG (Mooncake has no retire: in-place write).
    The NIXL retire handshake runs async in ScaleDownStateMachine.NIXL_RETIRE."""
    inst = ElasticEPStateManager.instance()
    if inst is not None:
        inst.deactivate_ranks(global_ranks)

    for group in _iter_live_parallel_groups():
        _flip_active_rank_mask(global_ranks, 0, group)
    _flip_active_rank_mask(global_ranks, 0)
    _refresh_ep_members()


# Departure from DRAIN, as opposed to arrival at it. The store barrier below proves
# every rank reached DRAIN; it cannot prove every rank has acted on that, and each rank
# learns it at its own next poll. A rank that folds into reconfig one poll early then
# blocks in a reconfig collective waiting for peers that are themselves blocked in the
# serving loop's cohort mlp_sync waiting for it: a circular wait, whether or not the two
# land on the same communicator, and a graceful retire raises no fault to break it. A
# second store counter only moves the race one level deeper, because the exit has to be
# decided from something every rank reads at the same instant. That something is the
# mlp_sync itself, which the cohort already runs every iteration: ranks announce
# readiness into it and cross only once it answers unanimously.
_departure_announced_at: Optional[float] = None
_departure_cleared = False
# Backstop, not a schedule: the answer arrives on the next iteration when the gather is
# live, so waiting this long means it is not carrying the flag (gather skipped for a
# degenerate dp size, say) and the pre-existing staggered exit is better than a hang.
_DEPARTURE_ALIGN_BUDGET_S = 30.0


def departure_announce() -> None:
    """Declare this rank done with DRAIN. Idempotent: the deadline is set once."""
    global _departure_announced_at
    if _departure_announced_at is None:
        _departure_announced_at = time.monotonic()


def departure_pending() -> bool:
    """One element of the mlp_sync gather: True while this rank still holds the cohort."""
    return _departure_announced_at is None


def departure_observe(cleared: bool) -> None:
    """Record the gather's verdict. Called from the mlp_sync, once per iteration."""
    global _departure_cleared
    _departure_cleared = cleared


def departure_cleared() -> bool:
    if _departure_cleared:
        return True
    if _departure_announced_at is None:
        return False
    if time.monotonic() - _departure_announced_at < _DEPARTURE_ALIGN_BUDGET_S:
        return False
    logger.warning(
        "[Elastic EP] departing DRAIN unaligned after %.0fs: the mlp_sync gather never "
        "reported cohort readiness, so peers may still be in the serving loop",
        _DEPARTURE_ALIGN_BUDGET_S,
    )
    return True


def departure_reset() -> None:
    global _departure_announced_at, _departure_cleared
    _departure_announced_at = None
    _departure_cleared = False


def retire_barrier_post() -> _BarrierHandle:
    """Post the WORLD retire barrier over the global TCPStore (elected-leader epoch).
    BARRIER_SKIP = nothing to synchronize; None = post failed, caller must retry.

    Deliberately not a collective. Mooncake orders collective issue behind the previous
    collective's completion, so an async barrier left outstanding across FSM ticks stalls
    the next mlp_sync at its issue, and ranks that have not yet ticked into DRAIN never
    arrive. The store carries the rendezvous instead; the GPU quiesce that actually
    guards the retiree is local (on_nixl_retire_pre), and NIXL_RETIRE right after us
    already fails closed when the store is unreachable."""
    if not torch.distributed.is_initialized():
        return BARRIER_SKIP
    rank = torch.distributed.get_rank()
    target = _barrier_target_from_effective_ep(torch.distributed.get_world_size())
    store = _store_or_none(f"[Elastic EP][retire_barrier] rank={rank}")
    if store is None:
        return None
    return _StoreBarrier.post(store, rank, "world", target, rewind_on_fail=True)


def retiree_local_cleanup() -> None:
    """CUDA quiesce, deactivate, then tear the backends down, all before sys.exit(0)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    # Last Mooncake call: no tick follows, so nothing broadcasts once we are dropped.
    _mooncake_deactivate_self()
    _destroy_local_process_groups()


def _destroy_local_process_groups() -> None:
    """Tear down our backends here, not at interpreter exit: Mooncake's destructors would
    run against an unloading CUDA driver, the error escapes a destructor, C++ calls
    std::terminate, and SIGABRT takes down any survivor still mid-reconfig. Safe once we
    are out of the view. Subgroups first, WORLD last, failures never block."""
    if not torch.distributed.is_initialized():
        return
    world = torch.distributed.group.WORLD
    seen = {id(world)}
    groups = []
    for group in _iter_live_parallel_groups():
        for pg in (group.device_group, group.cpu_group):
            if pg is not None and id(pg) not in seen:
                seen.add(id(pg))
                groups.append((group.unique_name, pg))
    for label, pg in groups + [("WORLD", None)]:
        try:
            torch.distributed.destroy_process_group(pg)
        except Exception as exc:
            logger.warning("[Elastic EP][retire] destroy %s failed: %s", label, exc)
