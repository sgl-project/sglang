from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
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


def _try_recover_world(global_ranks: List[int]) -> bool:
    from mooncake.pg import get_peer_state, recover_ranks

    world_backend = torch.distributed.group.WORLD
    if not all(get_peer_state(world_backend, global_ranks)):
        return False

    recover_ranks(world_backend, global_ranks)
    logger.debug("[Elastic EP][recover] WORLD recover_ranks(%s) done", global_ranks)
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
    """
    if not _try_recover_world(global_ranks):
        return False

    inst = ElasticEPStateManager.instance()
    if inst is not None and inst.active_ranks is not None:
        for global_rank in global_ranks:
            if 0 <= global_rank < inst.active_ranks.numel():
                inst.active_ranks[global_rank] = 1
        inst.snapshot_active_to_last()
        inst.sync_active_to_cpu()

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
    """Recover ranks in WORLD and every launch-time parallel group.

    Also defensively flips the Python-side ``active_ranks`` slice 0->1
    for the recovered ranks on every group. Mooncake C++
    ``recover_ranks`` writes back through the shared tensor storage,
    but not all launch paths observe that write from Python, so the
    explicit flip keeps EPLB / dp_attention / ``ElasticEPStateManager``
    consistent with the Mooncake view.
    """
    if not _try_recover_world(global_ranks):
        return False

    from mooncake.pg import recover_ranks

    for group in _iter_live_parallel_groups():
        local_ranks = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not local_ranks:
            continue

        _wait_for_peer_state(group.device_group, local_ranks)
        recover_ranks(group.device_group, local_ranks)
        _wait_for_peer_state(group.cpu_group, local_ranks)
        recover_ranks(group.cpu_group, local_ranks)
        _flip_active_rank_mask(group, global_ranks, value=1)
        _maybe_create_message_queue(group)

    _refresh_ep_members()
    return True


def _join_world_group() -> None:
    from mooncake.pg import join_group

    join_group(torch.distributed.group.WORLD)


def join_scale_process_group() -> None:
    """Join the expandable WORLD group for an append-only scale operation."""
    _join_world_group()
    _refresh_ep_members()


def join_process_groups() -> None:
    """Rejoin WORLD and every launch-time parallel group after recovery."""
    from mooncake.pg import join_group

    _join_world_group()
    for group in _iter_live_parallel_groups():
        if group.world_size <= 1:
            continue
        join_group(group.device_group)
        join_group(group.cpu_group)
        _maybe_create_message_queue(group)

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
    :func:`ElasticEPStateManager._finalize_scale_up`-style code paths
    already flip the mask via ``inst.active_ranks[rank] = 1`` for grow.

    Retirees stay in the WORLD PG's membership list; their slots remain
    reserved so a future :func:`try_recover_ranks` on the same rank can
    reactivate them cleanly (slot-reuse property).
    """
    if not global_ranks:
        return True

    inst = ElasticEPStateManager.instance()
    if inst is not None and inst.active_ranks is not None:
        for global_rank in global_ranks:
            if 0 <= global_rank < inst.active_ranks.numel():
                inst.active_ranks[global_rank] = 0
        inst.snapshot_active_to_last()
        inst.sync_active_to_cpu()

    for group in _iter_live_parallel_groups():
        _flip_active_rank_mask(group, global_ranks, value=0)

    _refresh_ep_members()
    logger.debug("[Elastic EP][retire] retire_ranks(%s) done", global_ranks)
    return True


def retire_barrier() -> None:
    """Cohort-wide barrier on ``torch.distributed.group.WORLD``.

    Called by every rank (survivors + retirees) BEFORE the mask flip in
    :func:`try_retire_ranks`. Two invariants are enforced by this
    barrier:

    * All in-flight collectives on WORLD have completed (barrier is
      posted after the drain phase halts new work).
    * All ranks have reached the mask-flip point simultaneously, so no
      survivor flips the mask while another rank is still expecting the
      retiree to participate in a prior collective.

    This is the last collective retirees post before ``sys.exit(0)``.
    After the barrier, retirees do only local state cleanup + exit;
    survivors flip the mask and proceed to reconfig.
    """
    torch.distributed.barrier(group=torch.distributed.group.WORLD)


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
    group()``. The launch-time process groups (both torch default WORLD
    and every sglang parallel group) stay at their launch-time
    membership; destroying them here would trigger the well-known
    "destroy blocks on live NCCL/Mooncake comms" hang (see the
    ``# Why`` note at ``parallel_state.py:2775-2784``). Process exit
    reclaims all Mooncake / NCCL / CUDA state for this process; the
    survivors' Mooncake internal peer state moves the retiree slot to
    inactive via the mask, and the reserved-but-inactive slot stays
    available for a later ``recover_ranks`` reactivation.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
