"""NPU sleep-mode comm cleanup orchestration ("comm" tag).

Ordering contract (wired up by SchedulerWeightUpdaterManager):
  sleep : CUDA graphs are dropped  -> barrier -> HCCL comms destroyed
  wakeup: HCCL comms rebuilt       -> barrier -> CUDA graphs recaptured

Captured graphs embed HCCL comm buffer addresses, so the two must never be
alive in a mismatched state. The gloo cpu_group of every GroupCoordinator is
retained throughout and serves as the cross-rank control plane (barriers,
consistency checks); mq_broadcaster rides on cpu_group and is kept as well.

destroy_process_group / new_group are collective: every rank must operate on
the same set of groups in the same order. We therefore sort groups by
unique_name and all-gather the per-rank group list before touching anything;
a mismatch aborts the release before any resource has been destroyed.
"""

from __future__ import annotations

import logging
from typing import Any, List

import torch
import torch.distributed

from sglang.srt.distributed import parallel_state
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    drop_cuda_graphs_for_sleep,
    force_eager_graphs,
    recapture_cuda_graphs_after_sleep,
    release_shared_graph_pool,
)

logger = logging.getLogger(__name__)


def _iter_cleanup_groups() -> List[parallel_state.GroupCoordinator]:
    """Alive GroupCoordinators participating in comm cleanup, sorted by name.

    Groups excluded from cleanup (mooncake backend, elastic-EP joiners) carry
    _rebuild_spec=None and are filtered out; release_device_comm() logs them.
    """
    groups = []
    for ref in parallel_state._groups.values():
        group = ref()
        if group is None or group._rebuild_spec is None:
            continue
        groups.append(group)
    groups.sort(key=lambda g: g.unique_name)
    return groups


def _world_cpu_group():
    return parallel_state.get_world_group().cpu_group


def _barrier(cpu_group, what: str) -> None:
    logger.debug(f"[sleep-comm] barrier: {what}")
    torch.distributed.barrier(group=cpu_group)


def _group_signatures(groups) -> List[Any]:
    """(unique_name, group_ranks) pairs: names alone cannot detect the
    'same name, different rank set' divergence across ranks."""
    return [
        (g.unique_name, tuple(tuple(r) for r in g._rebuild_spec.group_ranks))
        for g in groups
    ]


def _assert_ranks_agree(signatures: List[Any], cpu_group) -> None:
    """All ranks must agree on the group set before any destroy/rebuild.

    Cheap insurance against weakref-liveness or registration-order divergence
    across ranks, which would otherwise deadlock inside new_group. Called
    before any resource is touched, so raising here is a clean abort.
    """
    gathered: List[Any] = [None] * torch.distributed.get_world_size(cpu_group)
    torch.distributed.all_gather_object(gathered, signatures, group=cpu_group)
    if any(other != signatures for other in gathered):
        raise RuntimeError(
            f"[sleep-comm] ranks disagree on the comm-cleanup group set: "
            f"{gathered}. Aborting; no resource has been modified."
        )


def _target_model_runners(tp_worker) -> List[Any]:
    runners = list(getattr(tp_worker, "model_runner_list", None) or [])
    if not runners:
        runners = [tp_worker.model_runner]
    return runners


def _cleanup_graph_runner(runner) -> None:
    """Run a graph runner's backend cleanup if it has one (None-safe)."""
    if runner is None:
        return
    cleanup = getattr(getattr(runner, "backend", None), "cleanup", None)
    if callable(cleanup):
        cleanup()


def _drop_draft_graphs(draft_worker) -> None:
    if draft_worker is None:
        return
    # 1. Draft model runners' prefill/decode graphs.
    get_runners = getattr(draft_worker, "_draft_model_runners", None)
    if callable(get_runners):
        for runner in get_runners():
            drop_cuda_graphs_for_sleep(runner)
    # 2. The spec worker's own draft-decode / draft-extend graph runners.
    #    EAGLE keeps them on the INNER draft worker (EagleDraftWorker), not on
    #    the outer wrapper the scheduler hands us, so poke both.
    workers = [draft_worker]
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None and inner is not draft_worker:
        workers.append(inner)
    for worker in workers:
        for attr in ("cuda_graph_runner", "cuda_graph_runner_for_draft_extend"):
            runner = getattr(worker, attr, None)
            if runner is None:
                continue
            _cleanup_graph_runner(runner)
            setattr(worker, attr, None)
    # 3. Adaptive spec: every candidate step's runtime state owns graph
    #    runners of its own. Cleaning only the active step would leak the
    #    others and, worse, a later step switch would reinstall graphs
    #    captured on the destroyed comms. Drop all states; the wake path
    #    (init_cuda_graphs) re-registers and rebuilds them.
    controller = getattr(draft_worker, "adaptive_controller", None)
    states = getattr(controller, "_states", None)
    if states:
        for state in states.values():
            for attr in (
                "cuda_graph_runner",
                "target_graph_runner",
                "cuda_graph_runner_for_draft_extend",
            ):
                _cleanup_graph_runner(getattr(state, attr, None))
        states.clear()


def _drop_multimodal_graphs(model_runner) -> None:
    """Drop ViT graph runners living inside the model (e.g. Qwen3-VL).

    ViT blocks run TP all-gathers that get captured into the ViT graphs, so
    they must die with the comms like every other graph. Capture is lazy:
    cleared runners recapture on the next image request after wake.
    """
    model = getattr(model_runner, "model", None)
    if model is None:
        return
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        set_graph_pool_id,
    )
    from sglang.srt.hardware_backend.npu.graph_runner.vit_npu_graph_runner import (
        ViTNpuGraphRunner,
    )

    seen = set()
    for module in model.modules():
        runner = getattr(module, "graph_runners", None)
        if runner is None or id(runner) in seen:
            continue
        block_graphs = getattr(runner, "block_graphs", None)
        if not isinstance(block_graphs, dict):
            continue
        seen.add(id(runner))
        block_graphs.clear()
    if seen:
        # The NPU ViT runner keeps a class-level pool plus a global pool id;
        # both must go so nothing references the released memory.
        ViTNpuGraphRunner._graph_memory_pool = None
        set_graph_pool_id(None)


def _recapture_draft_graphs(draft_worker) -> bool:
    if draft_worker is None:
        return True
    try:
        # Same entry the scheduler uses at startup; re-captures draft model
        # runner graphs and the EAGLE worker's own graphs, wrapped in the
        # contexts (draft_pp/draft_tp/speculative_moe_backend) they require.
        draft_worker.init_cuda_graphs()
        return True
    except Exception:
        # Draft graphs are an optimization; keep the (eager) draft path alive.
        logger.exception(
            "[sleep-comm] draft CUDA graph recapture failed; draft runs eager."
        )
        # Never leave half-captured stale runners behind.
        _drop_draft_graphs(draft_worker)
        return False


def _destroy_comms(groups, cpu_group) -> int:
    _barrier(cpu_group, "before comm destroy")
    free_before, _ = torch.npu.mem_get_info()
    released = sum(1 for g in groups if g.release_device_comm())
    _barrier(cpu_group, "after comm destroy")
    free_after, _ = torch.npu.mem_get_info()
    logger.info(
        f"[sleep-comm] destroyed {released} device group(s); "
        f"free device memory {(free_before / (1 << 30)):.3f} GiB -> "
        f"{(free_after / (1 << 30)):.3f} GiB"
    )
    return released


def _rebuild_comms(groups, cpu_group) -> int:
    _barrier(cpu_group, "before comm rebuild")
    restored = sum(1 for g in groups if g.restore_device_comm())
    _barrier(cpu_group, "after comm rebuild")
    logger.info(f"[sleep-comm] restored {restored} device group(s)")
    return restored


def release_for_sleep(tp_worker, draft_worker=None) -> None:
    """Full comm-cleanup release sequence (see module docstring for order)."""
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer

    torch.npu.synchronize()
    cpu_group = _world_cpu_group()
    groups = _iter_cleanup_groups()
    _assert_ranks_agree(_group_signatures(groups), cpu_group)
    _barrier(cpu_group, "before graph drop")

    # 1. Graphs first: they embed HCCL comm buffer addresses.
    for runner in _target_model_runners(tp_worker):
        drop_cuda_graphs_for_sleep(runner)
        _drop_multimodal_graphs(runner)
    _drop_draft_graphs(draft_worker)
    release_shared_graph_pool(tp_worker.model_runner)

    # 2. DeepEP buffer references the old comm name; drop it before destroy.
    DeepEPBuffer.invalidate_buffer()

    # 3. Comms last.
    _destroy_comms(groups, cpu_group)


def restore_after_sleep(tp_worker, draft_worker=None) -> None:
    """Full comm-cleanup resume sequence (mirror of release_for_sleep)."""
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer

    cpu_group = _world_cpu_group()
    groups = _iter_cleanup_groups()
    _assert_ranks_agree(_group_signatures(groups), cpu_group)

    # 1. Comms first: recapture runs forward passes that need collectives.
    _rebuild_comms(groups, cpu_group)

    # 2. Defensive: normally already None after release; the next dispatch
    #    lazily recreates the buffer on the rebuilt process group.
    DeepEPBuffer.invalidate_buffer()

    # 3. Recapture graphs on the rebuilt comms.
    ok = True
    for runner in _target_model_runners(tp_worker):
        ok = recapture_cuda_graphs_after_sleep(runner) and ok
    ok = _recapture_draft_graphs(draft_worker) and ok

    # Recapture may fail on only some ranks; a graph/eager mix across ranks
    # forks collective sequences (e.g. DP attention requires every rank to
    # pick the same path). Vote over the retained gloo group: if any rank
    # failed, every rank falls back to eager together.
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=cpu_group
    )
    if flag.item() == 0:
        logger.error(
            "[sleep-comm] CUDA graph recapture failed on at least one rank; "
            "ALL ranks fall back to eager to keep collectives consistent."
        )
        for runner in _target_model_runners(tp_worker):
            force_eager_graphs(runner)
        _drop_draft_graphs(draft_worker)
