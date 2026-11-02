"""NPU sleep-mode comm cleanup backend ("comm" tag).

Sleep drops graphs, then destroys HCCL comms; wake rebuilds comms, then
recaptures graphs. Captured graphs embed comm buffer addresses, so the two
must never be alive in a mismatched state. All ranks must destroy/rebuild the
same groups in the same order, since both are collective.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_COMM,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.utils.sleep_comm.backend import (
    SleepCommBackend,
    all_ranks_ok,
    assert_ranks_agree,
    barrier,
    group_signatures,
    iter_rebuildable_groups,
    world_cpu_group,
)

logger = logging.getLogger(__name__)

# Device-group-bound communicators: they hold the device group handle, so they
# are destroyed with the process group and recreated after it is rebuilt.
_DEVICE_COMM_ATTRS = ("pcie_ipc_comm",)


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
    from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
        drop_cuda_graphs_for_sleep,
    )

    get_runners = getattr(draft_worker, "_draft_model_runners", None)
    if callable(get_runners):
        for runner in get_runners():
            drop_cuda_graphs_for_sleep(runner)

    # EAGLE keeps its graph runners on the inner draft worker.
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

    # Every adaptive step owns graph runners; a later step switch would
    # otherwise reinstall graphs captured on the destroyed comms.
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
    """Drop ViT graph runners, which also capture TP all-gathers."""
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
        # Drop the class-level pool and global pool id along with the graphs.
        ViTNpuGraphRunner._graph_memory_pool = None
        set_graph_pool_id(None)


def _recapture_draft_graphs(draft_worker) -> bool:
    if draft_worker is None:
        return True
    try:
        draft_worker.init_cuda_graphs()
        return True
    except Exception:
        # Draft graphs are an optimization; keep the eager draft path alive.
        logger.exception(
            "[sleep-comm] draft CUDA graph recapture failed; draft runs eager."
        )
        _drop_draft_graphs(draft_worker)
        return False


class NPUSleepCommBackend(SleepCommBackend):
    """Destroys device-side HCCL communicators on release and rebuilds them
    on resume."""

    tag = GPU_MEMORY_TYPE_COMM
    # This backend owns the graph lifecycle, so core skips the TMS graph pause.
    handles_cuda_graph = True
    # Recapture runs forward passes, which read weights and the KV cache.
    resume_requires = frozenset({GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE})

    def __init__(self, tp_worker, draft_worker):
        super().__init__(tp_worker, draft_worker)
        self._detached_comms: List[Tuple[Any, str]] = []

    @classmethod
    def unsupported_reason(cls) -> Optional[str]:
        """MindSpore reuses the torch HCCL handles, and the PP scheduler
        dispatches control requests inline, so the world-group barrier in the
        comm path deadlocks across stages."""
        from sglang.srt.configs.model_config import ModelImpl
        from sglang.srt.runtime_context import get_model, get_parallel

        if get_model().model_impl.lower() == ModelImpl.MINDSPORE:
            return (
                "The 'comm' memory occupation tag is not supported with "
                "model_impl=mindspore (MindSpore reuses the torch HCCL handles)."
            )
        if get_parallel().pp_size > 1:
            return (
                "The 'comm' memory occupation tag is not supported with pipeline "
                f"parallelism (pp_size={get_parallel().pp_size}); release the other "
                "tags without 'comm' instead."
            )
        return None

    def release(self) -> None:
        """Drop graphs, then the DeepEP buffer, then the HCCL comms."""
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            drop_cuda_graphs_for_sleep,
            release_shared_graph_pool,
        )

        torch.npu.synchronize()
        cpu_group = world_cpu_group()
        groups = iter_rebuildable_groups()
        assert_ranks_agree(group_signatures(groups), cpu_group)
        barrier(cpu_group, "before graph drop")

        for runner in _target_model_runners(self.tp_worker):
            drop_cuda_graphs_for_sleep(runner)
            _drop_multimodal_graphs(runner)
        _drop_draft_graphs(self.draft_worker)
        release_shared_graph_pool(self.tp_worker.model_runner)

        DeepEPBuffer.invalidate_buffer()
        self._destroy_device_comms(groups, cpu_group)

    def resume(self) -> None:
        """Rebuild comms, then recapture graphs."""
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            force_eager_graphs,
            recapture_cuda_graphs_after_sleep,
        )

        cpu_group = world_cpu_group()
        groups = iter_rebuildable_groups()
        assert_ranks_agree(group_signatures(groups), cpu_group)

        self._rebuild_device_comms(groups, cpu_group)
        # Normally already None after release; recreate lazily on next dispatch.
        DeepEPBuffer.invalidate_buffer()

        ok = True
        for runner in _target_model_runners(self.tp_worker):
            ok = recapture_cuda_graphs_after_sleep(runner) and ok
        ok = _recapture_draft_graphs(self.draft_worker) and ok

        # A graph/eager mix across ranks forks collective sequences, so vote
        # over the retained gloo group and fall back everywhere together.
        if not all_ranks_ok(ok, cpu_group):
            logger.error(
                "[sleep-comm] CUDA graph recapture failed on at least one rank; "
                "ALL ranks fall back to eager to keep collectives consistent."
            )
            for runner in _target_model_runners(self.tp_worker):
                force_eager_graphs(runner)
            _drop_draft_graphs(self.draft_worker)

    def _detach_device_comms(self, group) -> None:
        """Destroy the communicators bound to ``group``'s device group."""
        for attr in _DEVICE_COMM_ATTRS:
            comm = getattr(group, attr, None)
            if comm is None:
                continue
            destroy_fn = getattr(comm, "destroy", None)
            close_fn = getattr(comm, "close", None)
            if callable(destroy_fn):
                destroy_fn()
            elif callable(close_fn):
                close_fn()
            setattr(group, attr, None)
            self._detached_comms.append((group, attr))

    def _recreate_device_comm(self, group, attr: str) -> None:
        """Recreate one device-group-bound communicator detached by release."""
        if attr == "pcie_ipc_comm":
            from sglang.srt.distributed.device_communicators.pcie_ipc_ar import (
                PcieIpcCommunicator,
            )

            group.pcie_ipc_comm = PcieIpcCommunicator(
                group=group.device_group,
                device=group.device,
                cpu_group=group.cpu_group,
            )
        else:
            raise ValueError(f"Unknown device communicator attribute: {attr}")

    def _destroy_device_comms(self, groups, cpu_group) -> int:
        barrier(cpu_group, "before comm destroy")
        free_before, _ = torch.npu.mem_get_info()
        released = 0
        for group in groups:
            if group.device_group is None:
                continue
            self._detach_device_comms(group)
            if group.destroy_device_group():
                released += 1
        barrier(cpu_group, "after comm destroy")
        free_after, _ = torch.npu.mem_get_info()
        logger.info(
            f"[sleep-comm] destroyed {released} device group(s); "
            f"free device memory {(free_before / (1 << 30)):.3f} GiB -> "
            f"{(free_after / (1 << 30)):.3f} GiB"
        )
        return released

    def _rebuild_device_comms(self, groups, cpu_group) -> int:
        barrier(cpu_group, "before comm rebuild")
        restored = 0
        rebuilt = []
        for group in groups:
            if not group.rebuild_device_group():
                continue
            restored += 1
            rebuilt.append(group)
            for detached_group, attr in self._detached_comms:
                if detached_group is group:
                    self._recreate_device_comm(group, attr)
        if rebuilt:
            rebuilt_ids = {id(group) for group in rebuilt}
            self._detached_comms = [
                (group, attr)
                for group, attr in self._detached_comms
                if id(group) not in rebuilt_ids
            ]
        barrier(cpu_group, "after comm rebuild")
        logger.info(f"[sleep-comm] restored {restored} device group(s)")
        return restored
