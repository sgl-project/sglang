from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Sequence, Union

import msgspec
import torch

from sglang.srt.distributed import parallel_state
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
_RECOVER_COHORT_KEY_PREFIX = "elastic_ep/recover_cohort"
# Last inactive-rank set reported by is_scaling(), which polls: re-log only on change.
_fault_reported_ranks: tuple = ()
# Max warming_up without a forward; covers a fresh rank's first-forward DeepGEMM JIT.
_WARMUP_SETTLE_TIMEOUT_S = 60.0


class ScaleCohort(msgspec.Struct, frozen=True, kw_only=True):
    target_ep_size: int
    cuda_graph_enabled: bool
    attn_tp_size: int
    attn_cp_size: int


def register_scale_cohort(
    rank_offset: int, target_ep_size: int, cuda_graph_enabled: bool
) -> None:
    store = get_global_tcp_store()
    if store is None:
        raise RuntimeError("Elastic EP scale-up requires the global TCPStore.")
    parallel = get_parallel()
    payload = msgspec.json.encode(
        ScaleCohort(
            target_ep_size=target_ep_size,
            cuda_graph_enabled=cuda_graph_enabled,
            attn_tp_size=parallel.attn_tp_size,
            attn_cp_size=parallel.attn_cp_size,
        )
    )
    store.set(f"{_SCALE_COHORT_KEY_PREFIX}/{rank_offset}", payload)


def get_scale_cohort(rank_offset: int) -> Optional[ScaleCohort]:
    store = get_global_tcp_store()
    if store is None:
        return None
    key = f"{_SCALE_COHORT_KEY_PREFIX}/{rank_offset}"
    if not store.check([key]):
        return None
    return msgspec.json.decode(store.get(key), type=ScaleCohort)


def validate_scale_cohort_topology(cohort: ScaleCohort) -> None:
    from sglang.srt.elastic_ep.topology import validate_matching_attention_topology

    parallel = get_parallel()
    validate_matching_attention_topology(
        primary_attn_tp_size=parallel.attn_tp_size,
        primary_attn_cp_size=parallel.attn_cp_size,
        joining_attn_tp_size=cohort.attn_tp_size,
        joining_attn_cp_size=cohort.attn_cp_size,
    )


def update_dp_attention_for_elastic_ep(
    *, physical_ep_size: int, physical_ep_rank: int
) -> int:
    """Apply a physical EP topology to the logical DP-attention runtime."""
    from sglang.srt.elastic_ep.topology import (
        attn_replica_size,
        physical_ep_rank_to_dp_rank,
        physical_ep_size_to_dp_size,
    )
    from sglang.srt.layers.dp_attention import update_dp_attention_post_scale

    replica_size = attn_replica_size()
    dp_size = physical_ep_size_to_dp_size(physical_ep_size, replica_size)
    update_dp_attention_post_scale(
        new_dp_size=dp_size,
        new_dp_rank=physical_ep_rank_to_dp_rank(physical_ep_rank, replica_size),
    )
    get_parallel().override_permanently(num_dp_ranks=dp_size)
    return dp_size


def register_recover_cohort(
    rank_offset: int, effective_ep_size: int, tp_size: int = 1
) -> None:
    """Announce a recover joiner's expert-map width, one key per covered slot. Not the
    scale cohort's keys: an offset equal to the width would answer get_scale_cohort."""
    store = get_global_tcp_store()
    if store is None:
        raise RuntimeError("Elastic EP recover-mode join requires the global TCPStore.")
    for slot in range(rank_offset, rank_offset + tp_size):
        store.set(f"{_RECOVER_COHORT_KEY_PREFIX}/{slot}", str(effective_ep_size))


def clear_recover_cohort(rank_offset: int) -> None:
    """Drop a stale announce so a later grow cannot be planned off a dead joiner."""
    store = get_global_tcp_store()
    key = f"{_RECOVER_COHORT_KEY_PREFIX}/{rank_offset}"
    if store is not None and store.check([key]):
        store.delete_key(key)


def required_recover_width(recover_slots: List[int]) -> Optional[int]:
    """Narrowest width fitting every joiner on these slots, None until all announced
    (a guess misfits a wider or narrower joiner). Never sleeps (scheduler loop)."""
    store = get_global_tcp_store()
    if store is None:
        return None
    keys = [f"{_RECOVER_COHORT_KEY_PREFIX}/{slot}" for slot in recover_slots]
    if not all(store.check([key]) for key in keys):
        return None
    return max(int(store.get(key)) for key in keys)


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
        for g in global_ranks:
            if 0 <= g < self.active_ranks.numel():
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
    _poll_faults: Optional[Callable[[], None]] = None

    @classmethod
    def instance(cls) -> ElasticEPState:
        return cls._instance

    @classmethod
    def init(cls, server_args: ServerArgs):
        if cls._instance is not None:
            return cls._instance

        if get_exec().moe.elastic_ep_backend is not None:
            world_size = torch.distributed.get_world_size()
            active_rank_capacity = get_parallel().max_world_size
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

            if get_exec().moe.moe_a2a_backend == "nixl":
                cls._on_scale = cls._on_scale_nixl
                cls._poll_faults = cls._poll_faults_nixl

            inst.ep_join_rank_offset = get_parallel().ep_join_rank_offset
            if get_exec().moe.is_ep_joiner:
                cls._init_joiner_state(inst)

            cls._instance = inst

        return cls._instance

    @classmethod
    def _init_joiner_state(cls, inst: ElasticEPState) -> None:
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
    def request_scale(cls, n: int, recover_ranks: Sequence[int] = ()) -> bool:
        inst = cls._instance
        if inst is None:
            return False
        if inst.pending_ep_size is not None:
            return False
        # The caller checks bounds and feasibility, so it can tell the client why.
        inst.pending_recover_ranks = list(recover_ranks)
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
    def is_shrink_pending(cls) -> bool:
        inst = cls._instance
        return cls.is_scale_pending() and inst.pending_ep_size < inst.effective_ep_size

    @classmethod
    def is_scale_pending(cls) -> bool:
        inst = cls._instance
        return inst is not None and inst.pending_ep_size is not None

    @classmethod
    def commit_scale(cls) -> None:
        inst = cls._instance
        if inst is None or inst.pending_ep_size is None:
            return
        if inst.pending_ep_size < inst.effective_ep_size:
            inst.scale_phase = "serving_shrunk"
        else:
            cls.mark_warming_up()
        inst.effective_ep_size = inst.pending_ep_size
        inst.pending_ep_size = None
        inst.has_scaled = True
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
        """Leave warming_up after one forward at the new width (the remaining JIT needs
        the a2a, so a joiner cannot warm itself), or at the deadline if idle."""
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
        # Skip reset() once the mask is partially flipped: it is Mooncake ground truth.
        if not inst.mask_dirty:
            inst.reset()
        elif inst.active_ranks_cpu is not None:
            # Barrier targets read the width: size it to the mask's contiguous prefix.
            bits = inst.active_ranks_cpu.tolist()
            active_count = sum(bits)
            prefix = next((i for i, a in enumerate(bits) if not a), len(bits))
            if prefix != active_count:
                logger.error(
                    "[Elastic EP] active mask is not a contiguous prefix after a failed "
                    "scale (prefix=%d active=%d mask=%s); sizing to the prefix, so the "
                    "active ranks above it stay unreachable until a recover repairs them.",
                    prefix,
                    active_count,
                    bits,
                )
            # Narrow only: widening after a failed grow fans out to unactivated slots.
            if 0 < prefix < inst.effective_ep_size:
                inst.effective_ep_size = prefix
            elif prefix > inst.effective_ep_size:
                logger.warning(
                    "[Elastic EP] mask is wider than the committed width after a "
                    "failed grow (prefix=%d width=%d); keeping the width, so the "
                    "unjoined slots stay inactive until a recover admits them.",
                    prefix,
                    inst.effective_ep_size,
                )

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
    def get_data_plane_ep_size(cls) -> int:
        inst = cls._instance
        assert inst is not None, "Elastic EP state is not initialized."
        if inst.pending_ep_size is not None and inst.scale_phase in (
            "configuring_data_plane",
            "syncing_new_world",
        ):
            return inst.pending_ep_size
        return inst.effective_ep_size

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
    def poll_faults(cls) -> None:
        if cls._poll_faults is not None:
            cls._poll_faults()

    @staticmethod
    def _poll_faults_nixl() -> None:
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

        NixlEPBuffer.poll_rank_faults()

    @classmethod
    def get_inactive_ranks(cls) -> tuple[int, ...]:
        """Ranks in [0, effective_ep_size) whose mask bit is clear."""
        inst = cls._instance
        if inst is None or inst.active_ranks_cpu is None:
            return ()
        mask = inst.active_ranks_cpu
        return tuple(i for i in range(inst.effective_ep_size) if not int(mask[i]))

    @classmethod
    def is_scaling(cls) -> bool:
        """Return whether a scale or recovery operation is pending.

        The CPU snapshot is authoritative because rank polling uses it too.
        """
        inst = cls._instance
        if inst is None or inst.active_ranks_cpu is None:
            return False
        # warming_up: a grown cohort is slow until the joiner warms (see settle_warmup).
        if inst.pending_ep_size is not None or inst.scale_phase == "warming_up":
            return True
        missing = cls.get_inactive_ranks()
        if not missing:
            return False
        # Only a recover clears this, so a poller would wait forever: name the ranks.
        global _fault_reported_ranks
        if missing != _fault_reported_ranks:
            _fault_reported_ranks = missing
            logger.warning(
                "[Elastic EP] %d of %d ranks are inactive with no scale pending: %s. "
                "This is a post-scale fault, not a scale in progress; is_scaling() "
                "stays true until those ranks are recovered.",
                len(missing),
                inst.effective_ep_size,
                list(missing),
            )
        return True


def elastic_expanded_world_enabled() -> bool:
    """Return whether execution uses ranks admitted after server launch.

    Launch-time TP groups exclude ranks admitted during scale-up.
    """
    inst = ElasticEPStateManager.instance()
    if inst is None:
        return False
    if get_parallel().max_ep_size is None:
        return False
    return ElasticEPStateManager.get_data_plane_ep_size() > inst.original_ep_size


def _refresh_ep_members() -> None:
    from sglang.srt.layers.moe.token_dispatcher.mooncake import EPBuffer

    buffer = EPBuffer.get_existing_buffer()
    if buffer is not None:
        buffer.update_ep_member()


_BARRIER_STORE_POLL_S = 0.05
_BARRIER_EPOCH_CATCH_UP_S = 5.0
_RETIRE_BARRIER_TIMEOUT_S = 300.0
_BARRIER_RECHECK_S = 0.2
# Store, not dist.barrier(WORLD): Mooncake's bitmap lags our active_ranks flip.
# ns -> (arrival counter key, cycle counter key, ready key format, log prefix)
_BARRIER_NS: dict[str, tuple[str, str, str, str]] = {
    ns: (
        f"sglang_{stem}_arrival_counter",
        f"sglang_{stem}_cycle_counter",
        f"sglang_{ready}_e{{}}_posted",
        f"[Elastic EP][{tag}]",
    )
    for ns, stem, ready, tag in (
        ("nixl", "nixl_retire", "nixl_retire_barrier", "retire"),
        ("world", "retire_barrier", "retire_barrier", "retire_barrier"),
        ("scale_ready", "scale_ready", "scale_ready", "scale_ready"),
        ("nixl_wired", "nixl_wired", "nixl_wired", "nixl_wired"),
    )
}
_last_local_cycle_id: dict[str, int] = {ns: 0 for ns in _BARRIER_NS}
_EXPERT_MAP_INBOX_KEY = "sglang_expert_map_to_r{}"


def clear_expert_map_inbox(group_rank: int) -> None:
    """Drop residue a previous occupant never consumed, which would read as this scale's
    map. Only safe ahead of the announce, past which the source may have written."""
    store = _store_or_none("[Elastic EP][expert map]")
    if store is not None:
        store.delete_key(_EXPERT_MAP_INBOX_KEY.format(group_rank))


def share_expert_map_via_store(
    tensor: torch.Tensor, *, is_src: bool, cohort_ranks: Sequence[int], group_rank: int
) -> bool:
    """Hand the expert map over the store, one inbox per reader rank id (sparse after a
    fault); False = no store. Not a broadcast (Mooncake's putTaskCuda syncs the stream)
    nor a round keyed on participant count (desyncs when the width changes)."""
    store = _store_or_none("[Elastic EP][expert map]")
    if store is None or len(cohort_ranks) <= 1:
        return False
    if is_src:
        blob = tensor.cpu().numpy().tobytes()
        for peer in cohort_ranks:
            if peer != group_rank:
                store.set(_EXPERT_MAP_INBOX_KEY.format(peer), blob)
        return True
    key = _EXPERT_MAP_INBOX_KEY.format(group_rank)
    staged = torch.frombuffer(bytearray(store.get(key)), dtype=tensor.dtype)
    # Leave it empty so the next round blocks rather than being served this one.
    store.delete_key(key)
    tensor.copy_(staged.view(tensor.shape))
    return True


def sync_random_seed_via_store(is_source: bool) -> None:
    """Broadcast random_seed via TCPStore, not broadcast_pyobj: retirees may have left
    WORLD. Readers wait() rather than poll check() (libuv stalls it 30s) and seed the
    RNGs directly, since server_args is read-only."""
    import datetime

    from sglang.srt.runtime_context import get_device
    from sglang.srt.utils import set_random_seed

    key = "sglang_elastic_ep_random_seed"
    store = _store_or_none("[Elastic EP] random_seed sync:")
    if store is None:
        return
    if is_source:
        try:
            # get_device(): server_args.random_seed is None until the device sets it.
            store.set(key, str(int(get_device().random_seed)))
        except Exception as exc:
            logger.warning("[Elastic EP] random_seed source write failed (%s)", exc)
        return
    try:
        store.wait([key], datetime.timedelta(seconds=30))
        set_random_seed(_store_int(store, key))
    except Exception as exc:
        logger.warning(
            "[Elastic EP] random_seed sync failed (%s); keeping boot-time value", exc
        )


def assert_shrink_supported() -> None:
    """Fail a shrink this deployment cannot serve. Checked per request, not at init, so
    fault-tolerance-only deployments boot; parallel_hook asserts these only when
    --max-ep-size > tp_size."""
    from sglang.srt.runtime_context import get_disagg, get_serving

    parallel = get_parallel()
    disagg = get_disagg()
    # Only the base event loops tick the FSM; the others drop a scale silently.
    for ok, requirement in (
        (
            parallel.pp_size == 1,
            f"--pp-size 1 (got {parallel.pp_size}); WORLD must not span PP stages",
        ),
        (not disagg.enable_pdmux, "--enable-pdmux to be off"),
        (
            disagg.disaggregation_mode == "null",
            f"no disaggregation (got {disagg.disaggregation_mode})",
        ),
        (
            get_serving().tokenizer_worker_num == 1,
            f"--tokenizer-worker-num 1 (got {get_serving().tokenizer_worker_num})",
        ),
        (
            parallel.load_balance_method == "round_robin",
            "--load-balance-method round_robin, so a retired slot stops being "
            f"handed work (got {parallel.load_balance_method})",
        ),
        (
            # Not enable_dp_attention: parallel_hook consumes it and leaves it False.
            parallel.attn_dp_enabled,
            "--attn-dp-size > 1: the shrink retires EP ranks as DP slots",
        ),
        (
            # Not dp_size vs tp_size: at runtime tp_size is per-worker attention TP.
            parallel.attn_tp_size == 1,
            "attn_tp_size == 1 (got "
            f"attn_tp_size={parallel.attn_tp_size}): the shrink reports retired EP "
            "ranks as DP slots. Retiring whole logical replicas is not implemented yet",
        ),
        (
            # The launch hook's "scalable" predicate.
            parallel.max_ep_size is not None,
            "--max-ep-size. Without it this deployment "
            "was launched as fault-tolerance only, and the launch checks were relaxed "
            "on the understanding that it never retires a rank",
        ),
    ):
        if not ok:
            raise RuntimeError(f"Elastic EP scale-down requires {requirement}.")
    if get_exec().moe.elastic_ep_backend != "mooncake":
        return
    try:
        from mooncake.pg import deactivate_ranks  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Elastic EP scale-down requires a mooncake build providing "
            "mooncake.pg.deactivate_ranks; upgrade mooncake."
        ) from exc


def live_cohort_ranks() -> tuple[int, ...]:
    """Live rank ids: a fault can clear any mask bit, not just the top slots."""
    inst = ElasticEPStateManager.instance()
    if inst is None or inst.active_ranks_cpu is None or not inst.effective_ep_size:
        if torch.distributed.is_initialized():
            return tuple(range(torch.distributed.get_world_size()))
        return (0,)
    mask = inst.active_ranks_cpu
    return tuple(i for i in range(inst.effective_ep_size) if int(mask[i]))


def live_cohort_size() -> int:
    """Live rank count; WORLD still counts retirees that have exited."""
    return len(live_cohort_ranks())


def seed_barrier_epochs() -> None:
    """Adopt the cohort's cycle ids on an admitted joiner, which with no floor would
    accept a closed round and cross alone. ``add(key, 0)`` reads without minting."""
    store = _store_or_none("[Elastic EP][barrier seed]")
    if store is None:
        return
    for ns, (_, cycle_key, _, log) in _BARRIER_NS.items():
        try:
            _last_local_cycle_id[ns] = int(store.add(cycle_key, 0))
        except Exception as exc:
            logger.warning("%s cycle id seed failed (%s)", log, exc)


class _BarrierSkip:
    """No barrier needed; unlike None (post failed), which must not flip masks."""


BARRIER_SKIP = _BarrierSkip()


def _store_int(store, key: str) -> int:
    raw = store.get(key)
    return int(raw.decode() if isinstance(raw, bytes) else raw)


def _store_or_none(what: Optional[str] = None):
    store = get_global_tcp_store()
    if store is None and what:
        logger.warning("%s no TCPStore", what)
    return store


def _barrier_target_from_effective_ep() -> int:
    """Retire barrier target: the pre-shrink effective_ep_size (chained 4->3->2)."""
    return ElasticEPStateManager.instance().effective_ep_size


def _post_retire_barrier(ns: str) -> _BarrierHandle:
    rank = torch.distributed.get_rank()
    store = _store_or_none(f"{_BARRIER_NS[ns][3]} rank={rank}")
    if store is None:
        return None
    target = _barrier_target_from_effective_ep()
    return _StoreBarrier.post(store, rank, ns, target, rewind_on_fail=True)


@dataclass
class _StoreBarrier:
    """One in-flight store barrier. Its epoch is elected: arrival 1 mints the next cycle
    id and later arrivals read it, so chained scales cannot share a key."""

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
        """Elect an epoch, then announce on its ready key. ``rewind_on_fail`` frees a
        leader's epoch on a failed announce (a sync driver must not reuse one)."""
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
                with contextlib.suppress(Exception):
                    store.add(_BARRIER_NS[ns][1], -1)
                _last_local_cycle_id[ns] = epoch - 1
            # Always unwind: ARRIVAL must return to a base where a later cycle reads 1.
            cls._unwind(store, rank, ns)
            return None
        return cls(ns, rank, epoch, target, ready_key, arrival, time.monotonic())

    def check(self, store, secs: float) -> tuple[bool, int]:
        """Poll until target or ``secs`` (0 = one probe); returns (reached, count)."""
        deadline = time.monotonic() + secs
        seen = 0
        while True:
            with contextlib.suppress(Exception):
                seen = _store_int(store, self.ready_key)
                if seen >= self.world_size:
                    return True, seen
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
        with contextlib.suppress(Exception):
            store.add(_BARRIER_NS[ns][0], -1)

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
    state: _BarrierHandle,
    *,
    block_s: Optional[float] = None,
    keep_serving: bool = False,
) -> bool:
    """Non-blocking probe: True when every cohort rank posted. TimeoutError at 300s.
    ``block_s`` waits in place, for a caller that must not re-enter the event loop.
    ``keep_serving`` skips the catch-up window, which would stall a serving cohort."""
    if isinstance(state, _BarrierSkip):
        return True
    if state is None:
        return False  # post failed: fail closed so the FSM re-posts
    store = _store_or_none()
    seen = "no store"
    if store is not None:
        now = time.monotonic()
        # One catch-up window, then non-blocking polls: sleeping per tick slows decode.
        if block_s is not None:
            window = block_s
        elif state.first_check and not keep_serving:
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
    if isinstance(state, _StoreBarrier):
        state.consume()


def _pre_nixl_retire(retiree_global_ranks: List[int]) -> None:
    """Survivor-only NIXL peer disconnect before the retire barrier."""
    from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

    if NixlEPBuffer._state().buffer is None or not torch.distributed.is_initialized():
        return
    NixlEPBuffer.on_retire(retiree_global_ranks)


def nixl_retire_barrier_post() -> _BarrierHandle:
    """The NIXL retire barrier; same handle contract as ``retire_barrier_post``."""
    # Keyed off the backend, not the lazily built buffer: skipping must be cohort-wide.
    if (
        not torch.distributed.is_initialized()
        or get_exec().moe.moe_a2a_backend != "nixl"
    ):
        return BARRIER_SKIP
    return _post_retire_barrier("nixl")


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


def is_mooncake_pg(pg) -> bool:
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


def _is_mooncake_inactive_transient(exc: BaseException) -> bool:
    """Mooncake's "my bitmap has not caught up with active_ranks yet" rejection."""
    msg = str(exc)
    return "invalid state" in msg and "rank is not active" in msg


def mooncake_all_reduce_strict(tensor, *, op, group) -> None:
    """One all_reduce, no retry: a rank retrying a collective its peers completed
    deadlocks the cohort, and ``mooncake_world_settle_probe`` already converged."""
    try:
        torch.distributed.all_reduce(tensor, op=op, group=group)
    except RuntimeError as exc:
        if not _is_mooncake_inactive_transient(exc):
            raise
        raise RuntimeError(
            "[Elastic EP] Mooncake rejected a hot-path all_reduce as inactive after "
            "the post-scale settle probe should have converged its bitmap. Retrying "
            "here would deadlock the cohort, so this is fatal by design."
        ) from exc


# Well under --watchdog-timeout, so a view that never settles fails loudly before it.
_GATHER_SETTLE_BUDGET_S = 20.0
_GATHER_SETTLE_POLL_S = 0.05


def mooncake_all_gather_settling(output, input_, *, group, gather) -> None:
    """One mlp_sync gather, retried only while Mooncake refuses it as inactive (a bare
    fault has no settle probe). Safe: the refusal precedes posting, so no peer completed
    a gather this rank never sent. ``gather`` has ``all_gather_single``'s signature."""
    deadline = time.monotonic() + _GATHER_SETTLE_BUDGET_S
    waited_from = None
    while True:
        try:
            gather(output, input_, group=group)
            if waited_from is not None:
                logger.warning(
                    "[Elastic EP] mlp_sync gather settled after %.2fs",
                    time.monotonic() - waited_from,
                )
            return
        except RuntimeError as exc:
            if not _is_mooncake_inactive_transient(exc):
                raise
            if waited_from is None:
                waited_from = time.monotonic()
                logger.warning(
                    "[Elastic EP] Mooncake refused the mlp_sync gather as inactive, "
                    "waiting up to %.0fs for its membership view to settle",
                    _GATHER_SETTLE_BUDGET_S,
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "[Elastic EP] Mooncake refused the mlp_sync gather as inactive for "
                    f"{_GATHER_SETTLE_BUDGET_S:.0f}s, so its membership view never "
                    "settled after a rank fault."
                ) from exc
            time.sleep(_GATHER_SETTLE_POLL_S)


# Sleep before round i (~15s total). No wall-clock bound: every rank must run the same
# number of rounds; ranks reading a deadline either side would leave one voting alone.
_SETTLE_BACKOFF_S = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0, 2.0, 2.0, 2.0)


def mooncake_world_settle_probe(*, cohort_size: int) -> None:
    """Converge Mooncake's bitmap off the hot path; retries are store-voted, so ranks
    that succeeded re-attempt with the rest. Warn-only: aborting strands the width, and
    a cohort that cannot converge fails loudly on the next hot-path tick."""
    if not torch.distributed.is_initialized():
        return
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    last = None
    for backoff_s in _SETTLE_BACKOFF_S:
        if backoff_s:
            time.sleep(backoff_s)
        probe = torch.zeros(1, dtype=torch.int64, device=device)
        ok = True
        try:
            torch.distributed.all_reduce(probe, op=torch.distributed.ReduceOp.SUM)
        except RuntimeError as exc:
            if not _is_mooncake_inactive_transient(exc):
                raise
            ok, last = False, exc
        # A vote timeout means a peer stopped voting: treat it as non-convergence.
        try:
            if cohort_vote_via_store(
                ok, cohort_size, tag="mooncake_settle", timeout_s=30.0
            ):
                return
        except RuntimeError as exc:
            logger.warning("[Elastic EP] settle vote failed: %s", exc)
            return
    logger.warning(
        "[Elastic EP] settle probe did not converge in %d rounds (cohort=%d); last=%s",
        len(_SETTLE_BACKOFF_S),
        cohort_size,
        last,
    )


def scale_ready_barrier_via_store(target_size: int, *, timeout_s: float = 60.0) -> None:
    """Mooncake-free post-scale WORLD sync over the store; target_size = live ranks."""
    if not torch.distributed.is_initialized():
        return
    rank = torch.distributed.get_rank()
    store = _store_or_none(f"[Elastic EP][scale_ready] rank={rank}")
    if store is None:
        return
    state = _StoreBarrier.post(store, rank, "scale_ready", target_size)
    if state is None:
        # Fail closed: a rank skipping this runs on while its peers are still counting.
        raise RuntimeError(
            f"[Elastic EP][scale_ready] rank={rank} failed to arm the barrier "
            f"(target={target_size})"
        )
    reached, count = state.check(store, timeout_s)
    # Consume before raising: a timed-out leader still owes ARRIVAL, else a 2nd leader.
    state.consume()
    if not reached:
        raise RuntimeError(
            f"[Elastic EP][scale_ready] rank={rank} e={state.epoch} timeout after "
            f"{timeout_s}s (count={count} / target={target_size})"
        )


def nixl_wired_barrier_via_store(cohort_size: int, *, timeout_s: float = 30.0) -> None:
    """Hold a rank that has just wired itself to a width until its peers have too, as
    ``connect_ranks`` is no rendezvous. Warn-only and short: this runs inside a forward
    and under graph capture, where raising is fatal."""
    if cohort_size <= 1 or not torch.distributed.is_initialized():
        return
    rank = torch.distributed.get_rank()
    store = _store_or_none(f"[Elastic EP][nixl_wired] rank={rank}")
    if store is None:
        return
    state = _StoreBarrier.post(store, rank, "nixl_wired", cohort_size)
    if state is None:
        logger.warning(
            "[Elastic EP][nixl_wired] rank=%d could not arm the barrier at width %d; "
            "proceeding unsynchronized",
            rank,
            cohort_size,
        )
        return
    reached, count = state.check(store, timeout_s)
    # Consume before returning: a timed-out leader still owes ARRIVAL.
    state.consume()
    if not reached:
        logger.warning(
            "[Elastic EP][nixl_wired] rank=%d e=%d timeout after %.0fs "
            "(count=%d / target=%d); proceeding unsynchronized",
            rank,
            state.epoch,
            timeout_s,
            count,
            cohort_size,
        )


_HEARTBEAT_KEY = "sglang_elastic_heartbeat_r{}"


def beat_heartbeat() -> None:
    """Publish one liveness tick for this rank."""
    if not torch.distributed.is_initialized():
        return
    store = _store_or_none()
    if store is None:
        return
    rank = torch.distributed.get_rank()
    try:
        store.add(_HEARTBEAT_KEY.format(rank), 1)
    except Exception:
        logger.debug(
            "[Elastic EP][heartbeat] rank=%d could not beat", rank, exc_info=True
        )


def read_heartbeats(ranks: Sequence[int]) -> dict:
    """Liveness tick of each rank. Zero means a rank that has never beaten."""
    store = _store_or_none()
    if store is None:
        return {}
    ticks = {}
    for rank in ranks:
        try:
            # add(key, 0), not get: an absent key reads as zero instead of blocking.
            ticks[rank] = int(store.add(_HEARTBEAT_KEY.format(rank), 0))
        except Exception:
            logger.debug(
                "[Elastic EP][heartbeat] could not read rank=%d", rank, exc_info=True
            )
    return ticks


def cohort_vote_via_store(
    ok: bool, cohort_size: int, *, tag: str, timeout_s: float = 60.0
) -> bool:
    """Unanimous go/no-go over the store; a device all_reduce(MIN) would read a departed
    slot as a "no". Keyed per width, else consecutive shrinks split a round."""
    if cohort_size <= 1:
        return ok
    store = _store_or_none(f"[Elastic EP][vote] {tag}")
    if store is None:
        return ok
    ns = f"sglang_cohort_vote_{tag}_n{cohort_size}"
    seq_key, cast_key = f"{ns}_seq", f"{ns}_cast"
    rnd = (int(store.add(seq_key, 1)) - 1) // cohort_size
    yes_key = f"{ns}_r{rnd}_yes"
    # Tally before announcing, else the last arrival may read a tally missing a peer.
    store.add(yes_key, 1 if ok else 0)
    store.add(cast_key, 1)
    target = (rnd + 1) * cohort_size
    deadline = time.monotonic() + timeout_s
    while (cast := _store_int(store, cast_key)) < target:
        if time.monotonic() >= deadline:
            # Realign to the round boundary, or a short round offsets every later vote.
            for key in (seq_key, cast_key):
                with contextlib.suppress(Exception):
                    store.set(key, str(target))
            raise RuntimeError(
                f"[Elastic EP][vote] {tag} r{rnd} timeout after {timeout_s}s "
                f"({cast}/{target} voted)"
            )
        time.sleep(_BARRIER_STORE_POLL_S)
    return _store_int(store, yes_key) == cohort_size


def _wait_for_peer_state(backend, ranks: List[int], *, budget_s: float = 60.0) -> bool:
    """Poll until Mooncake sees ``ranks`` as peers on ``backend``. Bounded, so a joiner
    that never arrives costs one failed recovery rather than the event loop."""
    from mooncake.pg import get_peer_state

    deadline = time.monotonic() + budget_s
    while not all(get_peer_state(backend, ranks)):
        if time.monotonic() >= deadline:
            return False
        time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)
    return True


def _recover_parallel_groups(global_ranks: List[int]) -> bool:
    """Readmit ``global_ranks`` to every live parallel group's Mooncake PGs (bare fault
    only): Mooncake sizes each group's collectives from its own bitmap."""
    from mooncake.pg import recover_ranks

    world_backend = torch.distributed.group.WORLD
    for group in _iter_live_parallel_groups():
        # _try_recover_world took _WORLD's PGs; the joiner joins no group of one.
        if group is parallel_state._WORLD or group.world_size <= 1:
            continue
        local_ranks = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not local_ranks:
            continue
        for pg in (group.device_group, group.cpu_group):
            if pg is None or pg is world_backend:
                continue
            if not _wait_for_peer_state(pg, local_ranks):
                logger.warning(
                    "[Elastic EP][recover] %s peers %s never arrived; leaving the "
                    "recovery for a later tick",
                    group.unique_name,
                    local_ranks,
                )
                return False
            recover_ranks(pg, local_ranks)
    return True


def _try_recover_world(global_ranks: List[int], include_subgroups: bool) -> bool:
    """Recover WORLD-scope Mooncake peers, plus _WORLD sub-PGs if include_subgroups
    (recover-mode only; scale-up-v1 must pass False to avoid sub-PG ID mismatch)."""
    from mooncake.pg import get_peer_state, recover_ranks

    world_backend = torch.distributed.group.WORLD
    if not all(get_peer_state(world_backend, global_ranks)):
        return False

    recover_ranks(world_backend, global_ranks)
    logger.debug("[Elastic EP][recover] WORLD recover_ranks(%s) done", global_ranks)
    world_group = parallel_state._WORLD
    if not include_subgroups or world_group is None:
        return True
    for pg in (world_group.device_group, world_group.cpu_group):
        if pg is None or pg is world_backend:
            continue
        # Do not gate on get_peer_state: the joiner blocks in join_group (deadlock).
        deadline = time.monotonic() + 60.0
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


def try_admit_scale_ranks(global_ranks: List[int]) -> bool:
    """Admit append-only ranks into the expandable WORLD group."""
    return _activate(global_ranks, include_subgroups=False)


def try_recover_ranks(global_ranks: List[int]) -> bool:
    """Recover ranks in WORLD + sub-PGs. Scale regrow into a retired slot."""
    return _activate(global_ranks, include_subgroups=True)


def try_recover_faulted_ranks(global_ranks: List[int]) -> bool:
    """Bare-fault recovery: WORLD, its sub-PGs, and every live parallel group. Not for
    scale joiners: ``joinGroup`` admits only an isolated or inactive rank."""
    return _activate(global_ranks, include_subgroups=True, include_parallel_groups=True)


def _activate(
    global_ranks: List[int],
    include_subgroups: bool,
    *,
    include_parallel_groups: bool = False,
) -> bool:
    if not _try_recover_world(global_ranks, include_subgroups=include_subgroups):
        return False
    # Before _refresh_ep_members: its all-gather into moe_ep is sized from that group's
    # Mooncake bitmap, which only recover_ranks there updates.
    if include_parallel_groups and not _recover_parallel_groups(global_ranks):
        return False
    inst = ElasticEPStateManager.instance()
    if inst is not None:
        inst.activate_ranks(global_ranks)
    for group in _iter_live_parallel_groups():
        _flip_active_rank_mask(global_ranks, 1, group)
        # Only where a rank came back: a rebuild drops a live queue's shm and blocks.
        if _map_global_to_group_local_ranks(group.ranks, global_ranks):
            _maybe_create_message_queue(group)
    _flip_active_rank_mask(global_ranks, 1)
    _refresh_ep_members()
    return True


def _join_world_group(
    *, include_subgroups: bool = False, include_parallel_groups: bool = False
) -> None:
    from mooncake.pg import join_group

    world_backend = torch.distributed.group.WORLD
    join_group(world_backend)
    if not include_subgroups:
        return
    world_group = parallel_state._WORLD
    if world_group is not None:
        for pg in (world_group.device_group, world_group.cpu_group):
            if pg is not None and pg is not world_backend:
                join_group(pg)
    if not include_parallel_groups:
        return
    # Must match _recover_parallel_groups' readmits, or the next collective mis-sizes.
    for group in _iter_live_parallel_groups():
        if group is world_group or group.world_size <= 1:
            continue
        for pg in (group.device_group, group.cpu_group):
            if pg is not None and pg is not world_backend:
                join_group(pg)
        _maybe_create_message_queue(group)


def join_scale_process_group() -> None:
    """Join the expandable WORLD group for an append-only scale operation."""
    _join_world_group()
    _refresh_ep_members()


def join_process_groups() -> None:
    """Recover-mode grow join (includes sub-PGs)."""
    _join_world_group(include_subgroups=True)
    _refresh_ep_members()


def join_faulted_rank_process_groups() -> None:
    """Bare-fault rejoin. Mirrors ``try_recover_faulted_ranks`` on the survivor side."""
    _join_world_group(include_subgroups=True, include_parallel_groups=True)
    _refresh_ep_members()


def get_healthy_expert_location_src_rank(
    *, invoked_in_elastic_ep_rejoin_path: bool
) -> int:
    world_group = get_parallel().world_group
    # NOTE: do not key off the launch-time `ep_join_mode` here.
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
    # Not mid-resize: is_active_equal_last's sync can hang on a departing/arriving peer.
    if elastic_ep_state is None or ElasticEPStateManager.is_scale_pending():
        return False
    # Polled on the host, not inside the combine, so replayed decode graphs reach it.
    ElasticEPStateManager.poll_faults()
    if elastic_ep_state.is_active_equal_last():
        return False
    elastic_ep_state.snapshot_active_to_last()
    elastic_ep_state.sync_active_to_cpu()
    # Snapshot, then stand down if a stall cleared our own bit: the rendezvous skips us.
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() not in live_cohort_ranks():
            logger.warning(
                "[Elastic EP] this rank was marked inactive by the cohort; "
                "standing down from the fault rebalance and awaiting recovery"
            )
            return False
    logger.info("EPLB due to rank faults")
    gen = eplb_manager.rebalance()
    while True:
        try:
            next(gen)
        except StopIteration:
            break
    return True


def _flip_active_rank_mask(global_ranks: List[int], value: int, group=None) -> None:
    """``group=None`` targets the WORLD backend, which has no CPU mirror of the mask."""
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
    if world_ranks and is_mooncake_pg(world_backend):
        local = _map_global_to_group_local_ranks(world_ranks, global_ranks)
        if local:
            targets.append(("WORLD", world_backend, local))
    for group in _iter_live_parallel_groups():
        local = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not local:
            continue
        for pg in (group.device_group, group.cpu_group):
            if pg is not None and pg is not world_backend and is_mooncake_pg(pg):
                targets.append((group.unique_name, pg, local))
    return targets


# 60s: a retiree that drained up to the barrier still holds GPU work and RDMA slots.
def await_retirees_departed(global_ranks: List[int], *, budget_s: float = 60.0) -> None:
    """Hold the first post-flip device collective until Mooncake drops the retirees: our
    flip is local, so one posted first still expects them and the spin-wait kernel pegs
    the GPU. Must be get_peer_state: get_active_ranks reads back our own flip."""
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
                # Raise, not post: posting pegs the GPU; a failed scale keeps serving.
                raise RuntimeError(
                    f"[Elastic EP][retire] {label} still holds {local} after "
                    f"{budget_s:.1f}s; refusing to post a collective it still expects"
                )
            time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)


def _mooncake_deactivate_self() -> None:
    """Announce our own departure, as recover_ranks does on grow: unannounced, a shrink
    reads as a link fault, and joinGroup needs the slot inactive. Self-only, on exit."""
    from mooncake.pg import deactivate_ranks

    if not torch.distributed.is_initialized():
        return
    rank = torch.distributed.get_rank()
    for label, pg, local in _mooncake_membership_targets([rank]):
        try:
            # A Rejected proposal means a peer already carried the same deactivation.
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


# Departure is read off the mlp_sync gather, not the barrier: arrival proves a rank
# reached DRAIN, not that it acted, and every rank reads the gather at once.
_departure_announced_at: Optional[float] = None
_departure_cleared = False
# Backstop for a gather not carrying the flag, where a staggered exit beats a hang.
_DEPARTURE_ALIGN_BUDGET_S = 30.0


def departure_announce() -> None:
    """Declare this rank done with DRAIN. Idempotent: the deadline is set once."""
    global _departure_announced_at
    if _departure_announced_at is None:
        _departure_announced_at = time.monotonic()


def departure_pending() -> bool:
    """Our mlp_sync gather element: True while this rank still holds the cohort."""
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
    """Post the WORLD retire barrier. BARRIER_SKIP = nothing to sync; None = post
    failed, caller must retry. Over the store, not a collective: Mooncake issues each
    behind the previous one's completion; one pending across ticks stalls mlp_sync."""
    if not torch.distributed.is_initialized():
        return BARRIER_SKIP
    return _post_retire_barrier("world")


def retiree_local_cleanup() -> None:
    """CUDA quiesce, deactivate, then tear the backends down, all before sys.exit(0)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    # Last Mooncake call: no tick follows, so nothing broadcasts once we are dropped.
    _mooncake_deactivate_self()
    _destroy_local_process_groups()


def _destroy_local_process_groups() -> None:
    """Tear down our backends now, not at exit, where Mooncake's destructors would hit
    an unloading CUDA driver and std::terminate all survivors. Subgroups first."""
    seen = {id(torch.distributed.group.WORLD)}
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
