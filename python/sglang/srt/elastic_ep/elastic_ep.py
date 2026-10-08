from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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

# Bound on warming_up without serving. Covers the prefill-shape DeepGEMM JIT a fresh rank
# compiles on its first forward: tens of seconds, worse on a shared DG_JIT_CACHE_DIR.
_WARMUP_SETTLE_TIMEOUT_S = 60.0


class ScaleCohort(msgspec.Struct, frozen=True, kw_only=True):
    target_ep_size: int
    cuda_graph_enabled: bool


def register_scale_cohort(
    rank_offset: int, target_ep_size: int, cuda_graph_enabled: bool
) -> None:
    store = get_global_tcp_store()
    if store is None:
        raise RuntimeError("Elastic EP scale-up requires the global TCPStore.")
    payload = msgspec.json.encode(
        ScaleCohort(
            target_ep_size=target_ep_size,
            cuda_graph_enabled=cuda_graph_enabled,
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


def register_recover_cohort(
    rank_offset: int, effective_ep_size: int, tp_size: int = 1
) -> None:
    """Announce the cohort width a recover joiner sized its expert map for.

    One key per slot the joiner covers, not one at its offset.
    ``required_recover_width`` plans for a set of slots and wants a key for each, so a
    ``tp_size > 1`` joiner that announced only its offset left every slot above it
    silent: the width never resolved,
    the tokenizer retried for two minutes holding the scale lock, and the grow then
    staged through a width no joiner was waiting at.

    Its own key space, not the scale cohort's: survivors ask ``get_scale_cohort``
    whether an append grow has a cohort waiting, and a recover joiner parked at an
    offset equal to the current width would answer that for an unrelated grow.
    """
    store = get_global_tcp_store()
    if store is None:
        raise RuntimeError("Elastic EP recover-mode join requires the global TCPStore.")
    for slot in range(rank_offset, rank_offset + tp_size):
        store.set(f"{_RECOVER_COHORT_KEY_PREFIX}/{slot}", str(effective_ep_size))


def clear_recover_cohort(rank_offset: int, tp_size: int = 1) -> None:
    """Drop a stale announce so a later grow cannot be planned off a dead joiner."""
    store = get_global_tcp_store()
    if store is None:
        return
    for slot in range(rank_offset, rank_offset + tp_size):
        key = f"{_RECOVER_COHORT_KEY_PREFIX}/{slot}"
        if store.check([key]):
            store.delete_key(key)


def required_recover_width(recover_slots: List[int]) -> Optional[int]:
    """The narrowest cohort width that fits every joiner waiting on these slots.

    None until all of them have announced, and deliberately no guess in the meantime:
    going direct kills a joiner wider than the target at the cohort barrier, while
    assuming the launch width asks a narrower one to fill slots it has no ranks for.
    Reads the store once and never sleeps, because the caller is the scheduler loop.
    """
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
        # Bounds and feasibility are the caller's to reject: it can answer the client
        # with a reason, and its bound is never looser than ours.
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
        return (
            inst is not None
            and inst.pending_ep_size is not None
            and inst.pending_ep_size < inst.effective_ep_size
        )

    @classmethod
    def is_scale_pending(cls) -> bool:
        """Any resize in flight, grow or shrink. The width is moving either way."""
        inst = cls._instance
        return inst is not None and inst.pending_ep_size is not None

    @classmethod
    def commit_scale(cls) -> None:
        inst = cls._instance
        if inst is None or inst.pending_ep_size is None:
            return
        was_shrink = cls.is_shrink_pending()
        inst.effective_ep_size = inst.pending_ep_size
        inst.pending_ep_size = None
        inst.has_scaled = True
        if was_shrink:
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

        A joiner cannot warm itself: the rest is prefill-shape JIT that compiles only
        when a kernel runs, which in EP needs the a2a. One forward at the new width
        therefore proves it warm; the deadline covers an idle cohort."""
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
            # Trust the flip over the uncommitted width: barrier targets read
            # effective_ep_size, so leaving it pre-shrink waits on departed ranks.
            #
            # The contiguous prefix, not the popcount: every width here addresses ranks
            # as [0, width), so a mask with a hole would make the count name a
            # different set than the mask does, and later barriers would size
            # themselves for participants that are not the live ones.
            mask = inst.active_ranks_cpu
            active_count = int(mask.sum().item())
            prefix = 0
            while prefix < mask.shape[0] and int(mask[prefix]):
                prefix += 1
            if prefix != active_count:
                logger.error(
                    "[Elastic EP] active mask is not a contiguous prefix after a failed "
                    "scale (prefix=%d active=%d mask=%s); sizing to the prefix, so the "
                    "active ranks above it stay unreachable until a recover repairs them.",
                    prefix,
                    active_count,
                    mask.tolist(),
                )
            # Narrowing only. mask_dirty is set by activate_ranks too, so a failed grow
            # arrives here with a mask wider than the width, and adopting it would widen
            # effective_ep_size onto slots the DPC never activated -- and the tokenizer
            # sizes its control fan-out from this. A grow that failed keeps serving at
            # the width it already had; the unjoined slots are recovered, not adopted.
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
    def get_inactive_ranks(cls) -> Tuple[int, ...]:
        """Ranks inside the current width whose mask bit is clear.

        Non-empty only after a fault: a committed scale leaves the width and the mask
        agreeing. Barrier targets are sized from the width, so these are exactly the
        participants a later rendezvous would wait on and never get.
        """
        inst = cls._instance
        if inst is None or inst.active_ranks_cpu is None:
            return ()
        return tuple(
            i
            for i in range(inst.effective_ep_size)
            if not int(inst.active_ranks_cpu[i])
        )

    @classmethod
    def is_scaling(cls) -> bool:
        """Whether a scale or recovery is pending (CPU snapshot: rank polling reads it too)."""
        inst = cls._instance
        if inst is None or inst.active_ranks_cpu is None:
            return False
        if inst.pending_ep_size is not None:
            return True
        # Committed, but a grown cohort cannot serve at full speed until the joiner has
        # warmed; settle_warmup() closes this on the first forward or at the deadline.
        if inst.scale_phase == "warming_up":
            return True
        active_count = int(inst.active_ranks_cpu[: inst.effective_ep_size].sum().item())
        if active_count == inst.effective_ep_size:
            return False
        # No scale is pending, so the short mask is a post-scale rank fault. Recovery
        # is a separate path, so this does not clear on its own and a caller polling
        # this endpoint would otherwise wait on a state nothing is driving. Name the
        # ranks once rather than leave it to be inferred from a poll that never ends.
        global _fault_reported_ranks
        missing = cls.get_inactive_ranks()
        if missing != _fault_reported_ranks:
            _fault_reported_ranks = missing
            logger.warning(
                "[Elastic EP] %d of %d ranks are inactive with no scale pending: %s. "
                "This is a post-scale fault, not a scale in progress; is_scaling() "
                "stays true until those ranks are recovered.",
                inst.effective_ep_size - active_count,
                inst.effective_ep_size,
                list(missing),
            )
        return True


def elastic_expanded_world_enabled() -> bool:
    """Whether execution uses post-launch ranks: launch-time TP groups exclude them."""
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


# Shared across all three store barriers (NIXL retire, WORLD retire, scale_ready).
_BARRIER_STORE_POLL_S = 0.05
_BARRIER_EPOCH_CATCH_UP_S = 5.0
_RETIRE_BARRIER_TIMEOUT_S = 300.0
_BARRIER_RECHECK_S = 0.2

# The store stands in for dist.barrier(WORLD): Mooncake's bitmap lags our active_ranks
# flip while it digests retirees' link events.
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
    "nixl_wired": (
        "sglang_nixl_wired_arrival_counter",
        "sglang_nixl_wired_cycle_counter",
        "sglang_nixl_wired_e{}_posted",
        "[Elastic EP][nixl_wired]",
    ),
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
    """Hand the expert map to the cohort over the store, one inbox per reader. False = no
    store, caller falls back to the collective. Not a broadcast: Mooncake drives one through
    putTaskCuda, which syncs the stream from inside the collective. An inbox, not a shared
    round, since a round keyed on participant count desyncs when the width changes.

    Addressed by rank id, not by position: a reader waits on the inbox named for its own
    rank, and after a fault the live ids are no longer 0..n-1."""
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
            # get_device(), not server_args: the raw field is None until the device
            # namespace resolves it, so int() raised and no key was ever written.
            store.set(key, str(int(get_device().random_seed)))
        except Exception as exc:
            logger.warning("[Elastic EP] random_seed source write failed (%s)", exc)
        return

    try:
        store.wait([key], datetime.timedelta(seconds=30))
        set_random_seed(_store_int(store, key))
    except Exception as exc:
        logger.warning(
            "[Elastic EP] random_seed sync failed (%s); keeping boot-time value",
            exc,
        )


def _assert_shrink_runtime_supported() -> None:
    """The invariants the FSM itself needs, re-checked at shrink-request time.

    A no-op for a deployment that opted into scaling: parallel_hook asserts each of
    these at launch, but only when ``--max-ep-size > tp_size``. The point is the
    fault-tolerance deployment that never sets it, whose shrink would otherwise be
    accepted while skipping every one of them.
    """
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
            # attn_dp_enabled, not enable_dp_attention: the latter is the deprecated
            # spelling, and parallel_hook consumes it into attn_dp_size and leaves it
            # False, so reading it rejects every deployment that passed it.
            parallel.attn_dp_enabled,
            "--attn-dp-size > 1: the shrink retires EP ranks as DP slots",
        ),
    ):
        if not ok:
            raise RuntimeError(f"Elastic EP scale-down requires {requirement}.")


def assert_shrink_supported() -> None:
    """Fail a shrink request this deployment cannot serve.

    Checked here rather than at init: a fault-tolerance-only deployment never retires a
    rank, so requiring this at startup would stop an existing one from booting on a
    mooncake that is perfectly adequate for what it does. A planned shrink without it
    reads as a link fault (~10s/peer)."""
    _assert_shrink_runtime_supported()
    parallel = get_parallel()
    # The shrink reports retirees as DP slots: ElasticScaleUpdateReq.slot_offset /
    # slot_count, retired_ranks / safe_to_terminate_ranks and remove_elastic_workers
    # all index workers by EP rank. That identity holds only while one DP slot is one
    # EP rank. Once DP moves to logical-replica units (attn_tp_size > 1, #33728) a
    # shrink has to retire whole replicas and those ranges need converting, so refuse
    # here rather than deactivate the wrong workers.
    # attn_tp_size, not dp_size vs tp_size: at runtime tp_size is the per-worker
    # attention TP width, so comparing the two rejects working shapes. One rank per
    # attention DP group is what makes an EP rank a DP slot.
    if parallel.attn_tp_size != 1:
        raise RuntimeError(
            "Elastic EP scale-down requires attn_tp_size == 1 (got "
            f"attn_tp_size={parallel.attn_tp_size}): the shrink reports retired EP "
            "ranks as DP slots. Retiring whole logical replicas is not implemented yet."
        )
    # The same predicate the launch hook calls scalable, so the two halves agree.
    # Without it the hook relaxes three checks on the grounds that the deployment never
    # retires a rank: --enable-symm-mem and SGLANG_SYNC_TOKEN_IDS_ACROSS_TP tolerated
    # with a warning, and the shm broadcaster kept. Admitting a shrink anyway runs it
    # on the path this PR's own launch-time error calls bypassing active_ranks.
    if parallel.max_ep_size is None:
        raise RuntimeError(
            "Elastic EP scale-down requires --max-ep-size. Without it this deployment "
            "was launched as fault-tolerance only, and the launch checks were relaxed "
            "on the understanding that it never retires a rank."
        )
    if get_exec().moe.elastic_ep_backend != "mooncake":
        return
    try:
        from mooncake.pg import deactivate_ranks  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Elastic EP scale-down requires a mooncake build providing "
            "mooncake.pg.deactivate_ranks; upgrade mooncake."
        ) from exc


def live_cohort_ranks() -> Tuple[int, ...]:
    """The live ranks themselves, for anything that addresses a peer rather than counts
    them.

    A shrink retires the top slots, so there the live ranks are 0..n-1 and the count
    doubles as the id range. A fault clears any bit, so after one the two part ways: at
    width 6 with rank 4 down the count is 5 while the ids are 0,1,2,3,5. Addressing
    ``range(count)`` there writes to the rank that just died and skips a live one."""
    inst = ElasticEPStateManager.instance()
    if inst is None or inst.active_ranks_cpu is None or not inst.effective_ep_size:
        if torch.distributed.is_initialized():
            return tuple(range(torch.distributed.get_world_size()))
        return (0,)
    return tuple(
        i for i in range(inst.effective_ep_size) if int(inst.active_ranks_cpu[i])
    )


def live_cohort_size() -> int:
    """Live rank count, for scoping collectives that WORLD would run past retirees.

    ``effective_ep_size`` follows the commit, so this is the post-scale width once a
    shrink lands; WORLD still counts the ranks that have since called sys.exit()."""
    return len(live_cohort_ranks())


def seed_barrier_epochs() -> None:
    """Adopt the cohort's cycle ids; call on a joiner once admitted. A follower takes the
    first id above its last, and a process new to a namespace has no floor, so it would
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
    """No barrier needed. Distinct from None (post failed), which must not flip masks."""


BARRIER_SKIP = _BarrierSkip()


def _store_int(store, key: str) -> int:
    raw = store.get(key)
    return int(raw.decode() if isinstance(raw, bytes) else raw)


def _store_or_none(what: Optional[str] = None):
    """Global TCPStore or None. ``what`` = warning prefix."""
    store = get_global_tcp_store()
    if store is None and what:
        logger.warning("%s no TCPStore", what)
    return store


def _barrier_target_from_effective_ep() -> int:
    """Retire barrier target = pre-shrink effective_ep_size (chained shrinks 4->3->2)."""
    return ElasticEPStateManager.instance().effective_ep_size


def _post_retire_barrier(ns: str) -> _BarrierHandle:
    """Post a retire barrier in ``ns``. None = post failed, caller must retry."""
    rank = torch.distributed.get_rank()
    store = _store_or_none(f"{_BARRIER_NS[ns][3]} rank={rank}")
    if store is None:
        return None
    target = _barrier_target_from_effective_ep()
    return _StoreBarrier.post(store, rank, ns, target, rewind_on_fail=True)


@dataclass
class _StoreBarrier:
    """One in-flight store barrier. The epoch is elected, not computed: arrival 1 mints
    the next cycle id and later arrivals read it, so chained scales cannot share a key."""

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
        frees the leader's epoch on a failed announce (a sync driver must not reuse one)."""
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
    ``block_s`` waits in place rather than across ticks, for a caller that must not
    re-enter an event loop whose peers may have left the collectives it posts.
    ``keep_serving`` drops the catch-up window, for the one caller whose peers are all
    still serving: waiting in place there stalls decode for the whole window."""
    if isinstance(state, _BarrierSkip):
        return True
    if state is None:
        return False  # post failed: fail closed so the FSM re-posts

    store = _store_or_none()
    seen = "no store"
    if store is not None:
        now = time.monotonic()
        # Catch up on the same-tick fold, then poll without blocking. Sleeping 200ms per
        # re-tick is free when idle but costs a draining rank 7x decode.
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
    if state is None or isinstance(state, _BarrierSkip):
        return
    state.consume()


def _pre_nixl_retire(retiree_global_ranks: List[int]) -> None:
    """Survivor-side NIXL peer disconnect before the retire barrier. Survivor-only: the
    FSM routes a retiree to on_retiree_quiesce instead."""
    from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

    if NixlEPBuffer._state().buffer is None or not torch.distributed.is_initialized():
        return
    NixlEPBuffer.on_retire(retiree_global_ranks)


def nixl_retire_barrier_post() -> _BarrierHandle:
    """Post async NIXL retire barrier over the global TCPStore (elected-leader epoch).
    BARRIER_SKIP = nothing to synchronize; None = post failed, caller must retry."""
    # Keyed off the backend, not off this rank's buffer, which is lazy on first dispatch:
    # skipping is a cohort decision, and one rank opting out crosses without arriving.
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


def _is_mooncake_inactive_transient(exc: BaseException) -> bool:
    """Mooncake's "my bitmap has not caught up with active_ranks yet" rejection."""
    msg = str(exc)
    return "invalid state" in msg and "rank is not active" in msg


def mooncake_all_reduce_strict(tensor, *, op, group) -> None:
    """One all_reduce, no retry.

    A single rank must not retry a collective its peers already completed: the retry
    waits on members that moved on and they on it, deadlocking the cohort silently.
    ``mooncake_world_settle_probe`` converges the bitmap before the hot path, so a
    transient here means it did not; failing loudly is by design.
    """
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


# Bounded well under --watchdog-timeout, so a view that never settles still fails loudly
# instead of parking the cohort until the watchdog fires.
_GATHER_SETTLE_BUDGET_S = 20.0
_GATHER_SETTLE_POLL_S = 0.05


def mooncake_all_gather_settling(output, input_, *, group) -> None:
    """One mlp_sync gather, retried only while Mooncake refuses it as inactive.

    The opposite call from ``mooncake_all_reduce_strict``, for a reason that does not
    reach here. That one runs after ``mooncake_world_settle_probe`` converged the bitmap,
    so a refusal means convergence failed and waiting cannot help. A bare fault has no
    probe ahead of it: the coordinator drops the peer on its own schedule and the first
    gather after it can land while the membership change is still being digested. The
    refusal is a precondition check raised before the collective is posted, so no peer
    can have completed a gather this rank never sent, and waiting only makes it arrive
    late rather than not at all.
    """
    from sglang.srt.distributed.utils import all_gather_single

    deadline = time.monotonic() + _GATHER_SETTLE_BUDGET_S
    waited_from = None
    while True:
        try:
            all_gather_single(output, input_, group=group)
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


# Sleep before round i, in seconds; sums to ~15s. Every rank must run the same
# number of rounds, so no wall-clock bound: two ranks reading a deadline from
# either side would leave one voting alone.
_SETTLE_BACKOFF_S = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0, 2.0, 2.0, 2.0)


def mooncake_world_settle_probe(
    *, cohort_size: int, vote_timeout_s: float = 30.0
) -> None:
    """Converge Mooncake's bitmap off the hot path, in lockstep across the cohort.

    The retry decision is voted over the store (see ``_SETTLE_BACKOFF_S``), so ranks
    whose own attempt succeeded re-attempt with the rest; the probe tensor is a
    throwaway. Warn-only on give-up: aborting strands ``effective_ep_size``, and a
    cohort that truly cannot converge fails loudly on the next hot-path tick.
    """
    if not torch.distributed.is_initialized():
        return
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    last = None
    for attempt, backoff_s in enumerate(_SETTLE_BACKOFF_S):
        if backoff_s:
            time.sleep(backoff_s)
        probe = torch.zeros(1, dtype=torch.int64, device=device)
        ok = True
        try:
            torch.distributed.all_reduce(
                probe,
                op=torch.distributed.ReduceOp.SUM,
                group=None,  # WORLD
            )
        except RuntimeError as exc:
            if not _is_mooncake_inactive_transient(exc):
                raise
            ok, last = False, exc

        # A vote timeout means a peer stopped voting, which we cannot fix by
        # voting harder; treat it as non-convergence and let the hot path speak.
        try:
            if cohort_vote_via_store(
                ok,
                cohort_size,
                tag="mooncake_settle",
                timeout_s=vote_timeout_s,
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
    """Post-scale WORLD sync via TCPStore, Mooncake-free. ``target_size`` = live ranks."""
    if not torch.distributed.is_initialized():
        return

    rank = torch.distributed.get_rank()
    store = _store_or_none(f"[Elastic EP][scale_ready] rank={rank}")
    if store is None:
        return

    state = _StoreBarrier.post(store, rank, "scale_ready", target_size)
    if state is None:
        # Fail closed, as the retire barriers do: a rank that skips this barrier runs
        # on past it while its peers are still counting, which is the desync the
        # barrier exists to prevent.
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
    """Hold a rank that has just wired itself to a width until its peers have too.

    ``connect_ranks`` is not a rendezvous. A rank returns from it once its own side is
    set up, so the first one out posts a dispatch at the new width to peers still
    inside theirs, and times out once per expert per peer. At server start rank 0 came
    out of connect in 1.010s against 1.24s for ranks 1 to 3, and was the only rank to
    report: 22 timeouts against each of the three. A grow skews wider still, because
    the joiner has several GiB of transport buffers to register before it even begins
    connecting, and it lands in the post-scale graph capture, where the timeout is
    captured along with everything else.

    Warn-only, unlike the scale barriers, which raise. This runs inside a forward and
    under decode graph capture, where raising is fatal, and giving up here only leaves
    the skew that was there before. The wait is bounded well under the scale barriers'
    minute for the same reason: the leg being waited on is the joiner registering its
    buffers, measured at about 3s, so a wait that reaches the deadline is one that was
    never going to be met.
    """
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
    # Consume before returning: a timed-out leader still owes ARRIVAL, else a
    # second leader.
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
    """Publish one liveness tick for this rank.

    ``add`` creates the key at the increment when it is absent, so there is nothing to
    seed and a read never blocks on a rank that has not started.
    """
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
            # add(key, 0) reads and creates at zero, so an absent rank reads as zero
            # instead of blocking the way get does.
            ticks[rank] = int(store.add(_HEARTBEAT_KEY.format(rank), 0))
        except Exception:
            logger.debug(
                "[Elastic EP][heartbeat] could not read rank=%d", rank, exc_info=True
            )
    return ticks


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
    """Readmit ``global_ranks`` to every live parallel group's Mooncake PGs.

    Bare-fault recovery only. A departure is announced in all of these groups via
    ``_mooncake_membership_targets``, and Mooncake sizes each group's collectives from
    its own per-group bitmap, so any group left unrecovered mis-sizes the first
    collective posted over it. ``mlp_sync`` runs over ``tp_group``, which makes that the
    first one to hit.
    """
    from mooncake.pg import recover_ranks

    world_backend = torch.distributed.group.WORLD
    for group in _iter_live_parallel_groups():
        # _WORLD is in the same registry, and _try_recover_world already took its PGs.
        # The width skip matches the rejoining side, which has nothing to join in a
        # group of one. Readmitting where it does not join is the mismatch this whole
        # function exists to avoid.
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


def _activate(
    global_ranks: List[int],
    include_subgroups: bool,
    *,
    include_parallel_groups: bool = False,
) -> bool:
    if not _try_recover_world(global_ranks, include_subgroups=include_subgroups):
        return False
    # Ahead of the flip and _refresh_ep_members below: the Mooncake buffer refresh
    # all-gathers a cohort-wide list into the moe_ep group, which Mooncake still sizes
    # from its own bitmap until recover_ranks lands there.
    if include_parallel_groups and not _recover_parallel_groups(global_ranks):
        return False
    inst = ElasticEPStateManager.instance()
    if inst is not None:
        inst.activate_ranks(global_ranks)
    for group in _iter_live_parallel_groups():
        _flip_active_rank_mask(global_ranks, 1, group)
        # Only where a rank came back: rebuilding elsewhere drops a live queue's shm
        # segment and adds an untimed blocking collective to the scale path.
        if _map_global_to_group_local_ranks(group.ranks, global_ranks):
            _maybe_create_message_queue(group)
    _flip_active_rank_mask(global_ranks, 1)
    _refresh_ep_members()
    return True


def try_admit_scale_ranks(global_ranks: List[int]) -> bool:
    """Scale-up-v1 append (no sub-PG join)."""
    return _activate(global_ranks, include_subgroups=False)


def try_recover_ranks(global_ranks: List[int]) -> bool:
    """Recover ranks in WORLD + sub-PGs. Scale regrow into a retired slot."""
    return _activate(global_ranks, include_subgroups=True)


def try_recover_faulted_ranks(global_ranks: List[int]) -> bool:
    """Bare-fault recovery: WORLD, its sub-PGs, and every live parallel group.

    Wider than the scale regrow above, to match how wide the departure was. A scale
    joiner must not come through here: it is inactive rather than faulted, and
    ``joinGroup`` admits only an isolated or inactive rank.
    """
    return _activate(global_ranks, include_subgroups=True, include_parallel_groups=True)


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
            if pg is None or pg is world_backend:
                continue
            join_group(pg)
    if not include_parallel_groups:
        return
    # Exactly the groups a survivor readmits us to in _recover_parallel_groups. A rank
    # has to join the same set its peers readmit, or the next collective mis-sizes.
    for group in _iter_live_parallel_groups():
        if group is world_group or group.world_size <= 1:
            continue
        for pg in (group.device_group, group.cpu_group):
            if pg is None or pg is world_backend:
                continue
            join_group(pg)
        _maybe_create_message_queue(group)


def join_scale_process_group() -> None:
    """Scale-up-v1 append join (no sub-PG join)."""
    _join_world_group(include_subgroups=False)
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
    if elastic_ep_state is None:
        return False
    # Not while the width is moving. is_active_equal_last compares two device tensors,
    # so it forces a stream sync at the tail of every forward, and mid resize that sync
    # can land on a collective that a departing or arriving peer will never post. What
    # the cohort sees then is a scheduler watchdog timeout, not a slow tick. Until
    # commit the mask is the scale path's to maintain and a fault rebalance on top of it
    # would be fighting the finalize anyway. The comparison resumes on the first forward
    # after commit, so a real fault is deferred by one resize rather than missed.
    if ElasticEPStateManager.is_scale_pending():
        return False
    # Poll here rather than from inside the combine. This runs once per forward on the
    # host, after the model returns, so it is reached on a replayed decode graph too --
    # a counter stepped inside the combine is not, because the replay never runs the
    # python around it. It rate limits itself on wall clock, so every rank looks at the
    # same evidence at the same time however much of the routing each one carries.
    ElasticEPStateManager.poll_faults()
    if elastic_ep_state.is_active_equal_last():
        return False
    elastic_ep_state.snapshot_active_to_last()
    elastic_ep_state.sync_active_to_cpu()
    # Snapshot first, then stand down: a rank whose own bit the cohort cleared is not in
    # the cohort any more, so every rendezvous below is addressed to the ranks that are.
    # It waits to be recovered instead. Taking part would be a peer the others never
    # expect, and the expert map it blocks on is one nobody sends. Reachable because the
    # combine timeout that clears a bit is a liveness guess: a rank that merely stalled
    # for a forward is still running to see the verdict land on itself.
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


# 60s, not 15s: a retiree that drained a request up to the barrier still holds queued GPU
# work and RDMA slots, so it departs later than the idle case 15s was sized on.
def await_retirees_departed(global_ranks: List[int], *, budget_s: float = 60.0) -> None:
    """Hold the first post-flip device collective until Mooncake drops the retirees: our
    flip is local, so one posted first still expects them and the spin-wait kernel pegs
    the GPU. Must be ``get_peer_state``; ``get_active_ranks`` reads back our own flip."""
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
                # Raise rather than post anyway: posting is the pegged GPU this exists to
                # prevent, and a failed scale reconciles the width and keeps serving.
                raise RuntimeError(
                    f"[Elastic EP][retire] {label} still holds {local} after "
                    f"{budget_s:.1f}s; refusing to post a collective it still expects"
                )
            time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)


def _mooncake_deactivate_self() -> None:
    """Announce our own departure, mirroring ``recover_ranks`` on grow: the flip alone
    never reaches the coordinator, so a shrink reads as a link fault (~10s p2p timeout)
    and ``joinGroup`` needs the slot inactive. Self-only, on the way out only."""
    from mooncake.pg import deactivate_ranks

    if not torch.distributed.is_initialized():
        return

    for label, pg, local in _mooncake_membership_targets(
        [torch.distributed.get_rank()]
    ):
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


# Departure from DRAIN, not arrival: the barrier proves every rank reached it, not
# that each acted, and folding one poll early blocks on peers still in the serving
# loop's mlp_sync. Riding that mlp_sync is what makes every rank read it at once.
_departure_announced_at: Optional[float] = None
_departure_cleared = False
# Backstop, not a schedule: expiry means the gather is not carrying the flag (degenerate
# dp size, say), where a staggered exit beats a hang.
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
    """Post the WORLD retire barrier over the store. BARRIER_SKIP = nothing to
    synchronize; None = post failed, caller must retry. Not a collective: Mooncake orders
    collective issue behind the previous one's completion, so a barrier outstanding across
    ticks stalls the next mlp_sync and ranks not yet in DRAIN never arrive."""
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
    """Tear down our backends here, not at interpreter exit: Mooncake's destructors would
    run against an unloading CUDA driver, and an error escaping one takes down every
    survivor via std::terminate. Subgroups first, WORLD last, failures never block."""
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
