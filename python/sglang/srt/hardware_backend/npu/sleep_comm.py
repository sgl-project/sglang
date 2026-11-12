"""NPU ``comm`` memory tag: release/resume of the device HCCL communication
domains.

Every ``GroupCoordinator.device_group`` holds an HCCL comm with hundreds of MiB
of device buffers that are not torch allocations, so they are freed with
``reinit_process_group(pg, rebuild_link=True)``: the comm dies, the
ProcessGroup object stays alive (so cached ``device_group`` references remain
valid) and resume rebuilds the comm eagerly.

Ordering: release drops the captured graphs before destroying the comms, resume
rebuilds the comms before recapturing the graphs.  Every step is collective, so
all ranks walk the same groups in the same order.
"""

from __future__ import annotations

import gc
import logging
import sys
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.distributed

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.utils.sleep_comm import GPU_MEMORY_TYPE_COMM, SleepCommBackend

logger = logging.getLogger(__name__)

_GIB = 1 << 30

# Graph runners owned by a ModelRunner. Both capture collectives on the device
# group of the parallel group they run in.
_MODEL_RUNNER_GRAPH_ATTRS = ("prefill_cuda_graph_runner", "decode_cuda_graph_runner")
# Spec workers keep their own draft graph runners on the worker object.
_SPEC_WORKER_GRAPH_ATTRS = (
    "cuda_graph_runner",
    "cuda_graph_runner_for_draft_extend",
    # UNO keeps the draft decode runner it installs on the model runner.
    "_uno_draft_cuda_graph_runner",
)
# Per-shape buffers a ViT graph runner caches next to its graphs.
_VIT_GRAPH_CACHE_ATTRS = ("block_output", "block_input", "block_ws", "sin_cos_ws")
# Per-step graph runners held by an adaptive spec controller.
_ADAPTIVE_STATE_GRAPH_ATTRS = (
    "cuda_graph_runner",
    "target_graph_runner",
    "cuda_graph_runner_for_draft_extend",
)
# Sort key for the default process group, which is not a GroupCoordinator.
_DEFAULT_GROUP_NAME = "!default"

# Names of the groups whose comm this process destroyed; resume rebuilds only
# these and leaves every other group as lazy as it was before the sleep.
_DESTROYED_GROUP_NAMES: List[str] = []

# Whether this torch_npu can probe for an existing comm without creating one.
_COMM_PROBE_SUPPORTED: bool = True


def _device_group_backend(pg: torch.distributed.ProcessGroup) -> Optional[Any]:
    """The NPU backend of ``pg`` when it is HCCL, else None."""
    try:
        if "hccl" not in torch.distributed.get_backend(pg):
            return None
        return pg._get_backend(torch.device("npu"))
    except Exception:  # noqa: BLE001 - not every registered PG is usable here
        logger.debug(
            "[sleep-comm] skipping process group %s: cannot resolve its NPU backend.",
            pg,
            exc_info=True,
        )
        return None


def _comm_groups() -> List[Tuple[str, torch.distributed.ProcessGroup, Any, Any]]:
    """Every HCCL process group this process must release, sorted by name.

    Sorted because all ranks must destroy/rebuild the comms in the same order.
    """
    from sglang.srt.distributed import parallel_state

    groups: List[Tuple[str, torch.distributed.ProcessGroup, Any, Any]] = []
    seen = set()
    for group in parallel_state.get_all_groups():
        pg = getattr(group, "device_group", None)
        if pg is None or id(pg) in seen:
            continue
        backend = _device_group_backend(pg)
        if backend is None:
            continue
        seen.add(id(pg))
        groups.append((group.unique_name, pg, backend, group.cpu_group))

    # The default group is created by init_process_group and is separate from
    # the world coordinator's group, so it holds its own comm.
    default_pg = _default_group()
    if default_pg is not None and id(default_pg) not in seen:
        backend = _device_group_backend(default_pg)
        if backend is not None:
            groups.append(
                (_DEFAULT_GROUP_NAME, default_pg, backend, _world_cpu_group())
            )

    groups.sort(key=lambda entry: entry[0])
    return groups


def _default_group() -> Optional[torch.distributed.ProcessGroup]:
    from torch.distributed import distributed_c10d

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    return distributed_c10d._get_default_group()


def _world_cpu_group() -> torch.distributed.ProcessGroup:
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().world_group.cpu_group


def _signature(
    groups: Sequence[Tuple[str, torch.distributed.ProcessGroup, Any, Any]],
) -> List[Tuple[str, int]]:
    """(name, size) per group; must be identical on every rank."""
    return [(name, torch.distributed.get_world_size(pg)) for name, pg, _, _ in groups]


def _assert_ranks_agree(signature: List[Tuple[str, int]], cpu_group) -> None:
    """Abort before touching anything if ranks disagree on the group set."""
    gathered: List[Any] = [None] * torch.distributed.get_world_size(cpu_group)
    torch.distributed.all_gather_object(gathered, signature, group=cpu_group)
    if any(other != signature for other in gathered):
        raise RuntimeError(
            "[sleep-comm] ranks disagree on the communication-group set to "
            f"release: {gathered}. Nothing has been modified."
        )


def _barrier(cpu_group, what: str) -> None:
    logger.debug("[sleep-comm] barrier: %s", what)
    torch.distributed.barrier(group=cpu_group)


def _has_reinit_api() -> bool:
    """Whether this torch_npu provides the comm-reinit API."""
    try:
        from torch_npu.distributed import reinit_process_group
    except Exception:  # noqa: BLE001 - older/other torch_npu builds
        return False
    return callable(reinit_process_group)


def _has_live_comm(backend: Any, rank: int) -> bool:
    """Whether this rank already owns an HCCL comm for the group.

    ``init_comm=False`` only reads the comm cache, so no comm is created.
    """
    global _COMM_PROBE_SUPPORTED
    try:
        has_comm = bool(backend.get_hccl_comm_name(rank, init_comm=False))
    except TypeError:
        _COMM_PROBE_SUPPORTED = False
        return True
    _COMM_PROBE_SUPPORTED = True
    return has_comm


def _group_ranks(pg: torch.distributed.ProcessGroup) -> Optional[Tuple[int, ...]]:
    """Global ranks of ``pg``, or None when the API is unavailable."""
    try:
        ranks = torch.distributed.get_process_group_ranks(pg)
    except Exception:  # noqa: BLE001 - not every registered PG exposes this
        return None
    return tuple(ranks) if ranks else None


def _assert_group_members_agree(pg: torch.distributed.ProcessGroup, cpu_group) -> None:
    """Abort when the members of one group disagree about who is in it.

    A (name, size) match is not enough: the rank lists must match too.
    """
    ours = _group_ranks(pg)
    if ours is None:
        return
    gathered: List[Any] = [None] * torch.distributed.get_world_size(cpu_group)
    torch.distributed.all_gather_object(gathered, ours, group=cpu_group)
    if any(other != ours for other in gathered):
        raise RuntimeError(
            "[sleep-comm] ranks disagree about the members of a communication "
            f"group: {gathered}. Nothing has been modified."
        )


def _synchronize_quietly() -> None:
    """Sync the device without hiding a later barrier from the callers."""
    try:
        torch.npu.synchronize()
    except Exception:  # noqa: BLE001 - the failure is reported after the barrier
        logger.warning("[sleep-comm] device synchronize failed.", exc_info=True)


def _all_ranks_ok(ok: bool, cpu_group) -> bool:
    """AND-reduce ``ok`` over ``cpu_group``."""
    tensor = torch.tensor([1 if ok else 0], dtype=torch.int32)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
    )
    return bool(tensor.item())


# --------------------------------------------------------------------------- #
# Graph lifecycle
# --------------------------------------------------------------------------- #
def _cleanup_graph_runner(graph_runner: Any) -> None:
    if graph_runner is None:
        return
    cleanup = getattr(getattr(graph_runner, "backend", None), "cleanup", None)
    if not callable(cleanup):
        return
    try:
        cleanup()
    except Exception:  # noqa: BLE001 - dropping the object is enough
        # Every rank must reach the comm-destroy barrier; a locally raised
        # cleanup error would leave the other ranks waiting there.
        logger.warning(
            "[sleep-comm] graph runner cleanup failed; dropping it anyway.",
            exc_info=True,
        )


def _drop_model_runner_graphs(model_runner: Any) -> int:
    """Destroy one model runner's prefill and decode graphs; return the count."""
    from sglang.srt.model_executor.runner import EagerRunner

    if model_runner is None:
        return 0
    dropped = 0
    for attr in _MODEL_RUNNER_GRAPH_ATTRS:
        graph_runner = getattr(model_runner, attr, None)
        if graph_runner is None or isinstance(graph_runner, EagerRunner):
            continue
        _cleanup_graph_runner(graph_runner)
        setattr(model_runner, attr, None)
        dropped += 1
    # Bookkeeping read by /get_server_info and by the PD role switch.
    model_runner.decode_cuda_graph_captured = False
    model_runner.decode_cuda_graph_capture_bs = []
    return dropped


def _force_eager(model_runner: Any) -> None:
    """Leave one model runner in the eager fallback used at startup."""
    if model_runner is None:
        return
    _drop_model_runner_graphs(model_runner)
    eager_runner = getattr(model_runner, "eager_runner", None)
    model_runner.prefill_cuda_graph_runner = eager_runner


def _drop_vit_graphs(model_runners: Sequence[Any]) -> int:
    """Drop vision-encoder graphs, which also capture TP all-gathers."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        set_graph_pool_id,
    )
    from sglang.srt.hardware_backend.npu.graph_runner.vit_npu_graph_runner import (
        ViTNpuGraphRunner,
    )

    seen = set()
    for model_runner in model_runners:
        model = getattr(model_runner, "model", None)
        if model is None:
            continue
        for module in model.modules():
            graph_runner = getattr(module, "graph_runners", None)
            if graph_runner is None or id(graph_runner) in seen:
                continue
            block_graphs = getattr(graph_runner, "block_graphs", None)
            if not isinstance(block_graphs, dict):
                continue
            seen.add(id(graph_runner))
            block_graphs.clear()
            # The per-shape buffers are allocated alongside the graphs and
            # are recreated on demand by the next capture.
            for cache_attr in _VIT_GRAPH_CACHE_ATTRS:
                cache = getattr(graph_runner, cache_attr, None)
                if isinstance(cache, dict):
                    cache.clear()
    if seen:
        # The graphs are gone, so is their class-level pool and the pool id.
        ViTNpuGraphRunner._graph_memory_pool = None
        set_graph_pool_id(None)
        logger.info("[sleep-comm] dropped %d ViT graph runner(s)", len(seen))
    return len(seen)


def _spec_draft_model_runners(spec_worker: Any) -> List[Any]:
    if spec_worker is None:
        return []
    get_runners = getattr(spec_worker, "_draft_model_runners", None)
    if not callable(get_runners):
        return []
    try:
        return list(get_runners())
    except Exception:  # noqa: BLE001 - a spec worker may not have a draft model
        logger.warning(
            "[sleep-comm] cannot enumerate the draft model runners.", exc_info=True
        )
        return []


def _drop_spec_graphs(spec_worker: Any) -> int:
    """Destroy the graphs a speculative worker owns (draft + per-step)."""
    if spec_worker is None:
        return 0
    dropped = 0
    for model_runner in _spec_draft_model_runners(spec_worker):
        dropped += _drop_model_runner_graphs(model_runner)

    workers = [spec_worker]
    inner = getattr(spec_worker, "_draft_worker", None)
    if inner is not None and inner is not spec_worker:
        workers.append(inner)
    for worker in workers:
        for attr in _SPEC_WORKER_GRAPH_ATTRS:
            graph_runner = getattr(worker, attr, None)
            if graph_runner is None:
                continue
            _cleanup_graph_runner(graph_runner)
            setattr(worker, attr, None)
            dropped += 1

    # Every adaptive step owns graph runners; a later step switch would
    # otherwise reinstall graphs captured on the destroyed comms.
    controller = getattr(spec_worker, "adaptive_controller", None)
    states = getattr(controller, "_states", None)
    if states:
        for state in list(states.values()):
            for attr in _ADAPTIVE_STATE_GRAPH_ATTRS:
                _cleanup_graph_runner(getattr(state, attr, None))
        states.clear()
    return dropped


def _release_graph_pool() -> None:
    """Release the shared graph pool once every graph runner is gone."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        set_graph_pool_id,
    )
    from sglang.srt.model_executor.runner import (
        get_global_graph_memory_pool,
        set_global_graph_memory_pool,
    )

    if get_global_graph_memory_pool() is not None:
        set_global_graph_memory_pool(None)
    # The per-backend graph pool handle is gone with the backends.
    set_graph_pool_id(None)
    gc.collect()
    torch.npu.empty_cache()


def _recapture_model_runner_graphs(model_runner: Any) -> bool:
    """Recapture prefill and decode graphs on the rebuilt comms."""
    if model_runner is None:
        return True
    # init_*_cuda_graph() overwrites the runner attribute without cleanup.
    _drop_model_runner_graphs(model_runner)
    try:
        model_runner.init_prefill_cuda_graph()
        model_runner.init_decode_cuda_graph()
        return True
    except Exception:  # noqa: BLE001 - a graph is an optimization, not correctness
        logger.exception(
            "[sleep-comm] CUDA graph recapture failed; falling back to eager."
        )
        _force_eager(model_runner)
        return False


def _recapture_spec_graphs(spec_worker: Any) -> bool:
    if spec_worker is None:
        return True
    init_cuda_graphs = getattr(spec_worker, "init_cuda_graphs", None)
    if not callable(init_cuda_graphs):
        return True
    try:
        init_cuda_graphs()
        return True
    except Exception:  # noqa: BLE001
        logger.exception(
            "[sleep-comm] speculative CUDA graph recapture failed; the draft "
            "runs eager."
        )
        _drop_spec_graphs(spec_worker)
        return False


def _invalidate_cached_comm_names(model_runners: Sequence[Any]) -> None:
    """Clear the EP comm name cached by the AscendOps MoE dispatcher.

    A rebuilt comm may get a different name, so the dispatcher must re-fetch it.
    """
    for model_runner in model_runners:
        model = getattr(model_runner, "model", None)
        if model is None:
            continue
        for module in model.modules():
            dispatcher = getattr(module, "dispatcher", None)
            if getattr(dispatcher, "_ep_comm_name", None) is None:
                continue
            dispatcher._ep_comm_name = None
            logger.info(
                "[sleep-comm] cleared the cached EP comm name of a MoE dispatcher."
            )


def _invalidate_deepep_buffer() -> None:
    """Drop DeepEP's cached buffer, which is bound to the old comm."""
    module = sys.modules.get("sglang.srt.layers.moe.token_dispatcher.deepep")
    buffer_cls = getattr(module, "DeepEPBuffer", None) if module is not None else None
    if buffer_cls is None:
        return
    get_state = getattr(buffer_cls, "_state", None)
    if not callable(get_state):
        return
    state = get_state()
    if getattr(state, "buffer", None) is None:
        return
    try:
        buffer_cls.clean_buffer()
    except Exception:  # noqa: BLE001 - cleaning is best effort
        logger.warning("[sleep-comm] failed to clean the DeepEP buffer.", exc_info=True)
    state.buffer = None


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class NPUSleepCommBackend(SleepCommBackend):
    """Destroy the HCCL comms of every group on release, rebuild them on resume."""

    tag = GPU_MEMORY_TYPE_COMM
    # The backend owns the graph lifecycle, so the scheduler skips the
    # memory-saver graph tag (which cannot pause NPU graph memory anyway).
    handles_cuda_graph = True
    # Recapturing a graph runs the model, which reads weights and the KV cache.
    resume_requires = frozenset({GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE})

    @classmethod
    def unsupported_reason(cls) -> Optional[str]:
        """Topologies where destroying the comms is known to break something."""
        from sglang.srt.disaggregation.mode import DisaggregationMode
        from sglang.srt.runtime_context import (
            get_disagg,
            get_flags,
            get_model,
            get_parallel,
        )

        # Compared as a string to avoid importing model_config here.
        model_impl = getattr(get_model(), "model_impl", None)
        if str(getattr(model_impl, "value", model_impl) or "").lower() == "mindspore":
            # MindSpore pins the torch HCCL comm handle into its own group.
            return (
                "The 'comm' memory occupation tag is not supported with "
                "model_impl=mindspore."
            )

        pp_size = getattr(get_parallel(), "pp_size", 1)
        if pp_size > 1:
            # The request only reaches this process set, so a world barrier hangs.
            return (
                "The 'comm' memory occupation tag is not supported with pipeline "
                f"parallelism (pp_size={pp_size}); release the other tags without "
                "'comm' instead."
            )

        disaggregation_mode = getattr(get_disagg(), "disaggregation_mode", None)
        transfer_backend = getattr(get_disagg(), "disaggregation_transfer_backend", "")
        if transfer_backend == "ascend" and disaggregation_mode not in (
            None,
            DisaggregationMode.NULL.value,
        ):
            # The Ascend transfer engine keeps using the live HCCL comm handle.
            return (
                "The 'comm' memory occupation tag is not supported with the "
                "Ascend PD-disaggregation transfer backend."
            )

        if not _has_reinit_api():
            return (
                "The 'comm' memory occupation tag needs "
                "torch_npu.distributed.reinit_process_group, which this torch_npu "
                "does not provide."
            )

        # Compiled piecewise graphs cache HCCL collectives nothing here can drop.
        flags = get_flags()
        if getattr(getattr(flags, "capture", None), "enable_torch_compile", False):
            return (
                "The 'comm' memory occupation tag is not supported together with "
                "--enable-torch-compile on NPU: compiled piecewise graphs cache "
                "HCCL collectives outside the graph runners."
            )

        try:
            has_comm_groups = bool(_comm_groups())
        except Exception:  # noqa: BLE001 - report a clear reason instead
            logger.exception(
                "[sleep-comm] cannot enumerate the HCCL process groups; assuming "
                "there are none."
            )
            has_comm_groups = False
        if not has_comm_groups:
            # No HCCL process group in this process (e.g. zbal device backend).
            return (
                "The 'comm' memory occupation tag needs an HCCL device process "
                "group, but this process has none."
            )
        return None

    # -- release ------------------------------------------------------------
    def release(self) -> None:
        torch.npu.synchronize()
        cpu_group = _world_cpu_group()
        groups = _comm_groups()
        if not groups:
            logger.warning(
                "[sleep-comm] no HCCL process group to release; nothing to do."
            )
            return
        _assert_ranks_agree(_signature(groups), cpu_group)

        # 1. Drop captured graphs. A captured collective records the HCCL comm
        #    it ran on, which is destroyed below.
        _barrier(cpu_group, "before graph drop")
        model_runners = self._target_model_runners()
        dropped = 0
        for model_runner in model_runners:
            dropped += _drop_model_runner_graphs(model_runner)
        dropped += _drop_vit_graphs(model_runners)
        dropped += _drop_spec_graphs(self.draft_worker)
        if dropped:
            _release_graph_pool()
        # 2. DeepEP's buffer is bound to its comm.
        _invalidate_deepep_buffer()
        _synchronize_quietly()

        # 3. Destroy the comms in place. The ProcessGroup objects survive, so
        #    every cached device_group reference stays valid.
        _barrier(cpu_group, "before comm destroy")
        free_before, _ = torch.npu.mem_get_info()
        reinit_process_group = self._reinit_process_group()
        rank = torch.distributed.get_rank()
        _DESTROYED_GROUP_NAMES.clear()
        failure: Optional[BaseException] = None
        for name, pg, backend, group_cpu_group in groups:
            _assert_group_members_agree(pg, group_cpu_group)
            # Vote over the group's own ranks so they destroy the comm together.
            has_comm = _all_ranks_ok(_has_live_comm(backend, rank), group_cpu_group)
            if not has_comm:
                logger.info(
                    "[sleep-comm] group '%s' has no comm yet; leaving it lazy", name
                )
                continue
            logger.info("[sleep-comm] destroying the comm of group '%s'", name)
            # Recorded before the call so a partial failure still resumes it.
            if _COMM_PROBE_SUPPORTED:
                _DESTROYED_GROUP_NAMES.append(name)
            try:
                reinit_process_group(pg, rebuild_link=True)
            except Exception as e:  # noqa: BLE001 - reported after the barrier
                logger.exception(
                    "[sleep-comm] destroying the comm of group '%s' failed", name
                )
                failure = failure or e
        _synchronize_quietly()
        _barrier(cpu_group, "after comm destroy")
        free_after, _ = torch.npu.mem_get_info()
        logger.info(
            "[sleep-comm] destroyed %d/%d communication domain(s); free device "
            "memory %.3f GiB -> %.3f GiB",
            len(_DESTROYED_GROUP_NAMES),
            len(groups),
            free_before / _GIB,
            free_after / _GIB,
        )
        if not _all_ranks_ok(failure is None, cpu_group):
            raise RuntimeError(
                "[sleep-comm] releasing communication domains failed on at least "
                f"one rank (this rank: {failure!r})."
            )
        if not _COMM_PROBE_SUPPORTED:
            logger.warning(
                "[sleep-comm] this torch_npu cannot report whether a group has a "
                "comm; the domains are destroyed but resume leaves them lazy."
            )

    # -- resume -------------------------------------------------------------
    def resume(self) -> None:
        torch.npu.synchronize()
        cpu_group = _world_cpu_group()
        groups = _comm_groups()
        if not groups:
            logger.warning(
                "[sleep-comm] no HCCL process group to rebuild; nothing to do."
            )
            return
        _assert_ranks_agree(_signature(groups), cpu_group)

        # 1. Rebuild exactly the comms release destroyed, in the release order.
        #    Groups that had no comm stay lazy, as they were before the sleep.
        _barrier(cpu_group, "before comm rebuild")
        rank = torch.distributed.get_rank()
        to_rebuild = set(_DESTROYED_GROUP_NAMES)
        rebuilt = 0
        failure: Optional[BaseException] = None
        for name, pg, backend, group_cpu_group in groups:
            _assert_group_members_agree(pg, group_cpu_group)
            # Same vote as release: comm init is collective and would hang.
            if not _all_ranks_ok(name in to_rebuild, group_cpu_group):
                continue
            logger.info("[sleep-comm] rebuilding the comm of group '%s'", name)
            try:
                self._init_comm(backend, rank)
            except Exception as e:  # noqa: BLE001 - reported after the barrier
                logger.exception(
                    "[sleep-comm] rebuilding the comm of group '%s' failed", name
                )
                failure = failure or e
            rebuilt += 1
        _synchronize_quietly()
        _barrier(cpu_group, "after comm rebuild")
        logger.info("[sleep-comm] rebuilt %d communication domain(s)", rebuilt)
        if not _all_ranks_ok(failure is None, cpu_group):
            raise RuntimeError(
                "[sleep-comm] rebuilding communication domains failed on at least "
                f"one rank (this rank: {failure!r})."
            )
        _DESTROYED_GROUP_NAMES.clear()

        # 2. The MoE dispatchers and DeepEP must re-resolve the rebuilt comm.
        model_runners = self._target_model_runners()
        _invalidate_cached_comm_names(model_runners)
        _invalidate_deepep_buffer()

        # 3. Recapture the graphs. Every rank falls back to eager together,
        #    since eager changes its collective sequence.
        ok = True
        for model_runner in model_runners:
            ok = _recapture_model_runner_graphs(model_runner) and ok
        ok = _recapture_spec_graphs(self.draft_worker) and ok
        if not _all_ranks_ok(ok, cpu_group):
            logger.error(
                "[sleep-comm] CUDA graph recapture failed on at least one rank; "
                "all ranks fall back to eager so their collective sequences stay "
                "identical. A capture that fails inside a collective cannot "
                "unblock the peers that are waiting in it, so this vote only "
                "reaches ranks that got out; restart the engine if the wake "
                "appears wedged."
            )
            for model_runner in model_runners:
                _force_eager(model_runner)
            _drop_spec_graphs(self.draft_worker)
        torch.npu.synchronize()

    # -- agreement ----------------------------------------------------------
    def agree(self, proceed: bool) -> bool:
        """AND-reduce a local go/no-go decision over every rank.

        A rank that returns early alone would leave the others in the barrier.
        """
        return _all_ranks_ok(proceed, _world_cpu_group())

    # -- helpers ------------------------------------------------------------
    def _target_model_runners(self) -> List[Any]:
        runners = list(getattr(self.tp_worker, "model_runner_list", None) or [])
        if runners:
            return runners
        model_runner = getattr(self.tp_worker, "model_runner", None)
        return [model_runner] if model_runner is not None else []

    @staticmethod
    def _reinit_process_group():
        from torch_npu.distributed import reinit_process_group

        return reinit_process_group

    @staticmethod
    def _init_comm(backend: Any, rank: int) -> None:
        """Build the comm eagerly (equivalent to running a collective)."""
        try:
            backend.get_hccl_comm_name(rank, init_comm=True)
        except TypeError:
            # Older torch_npu always initializes.
            backend.get_hccl_comm_name(rank)
