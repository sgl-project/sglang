from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _get_container_available_memory_bytes() -> int:
    """Return available memory in bytes, container-aware.

    In a container / K8s pod, ``psutil.virtual_memory().available`` returns the
    **host's** available memory, which can be orders of magnitude higher than
    what the cgroup actually allows.  This function checks cgroup v2 and v1
    first and falls back to psutil only when no cgroup limit is found.

    Returns ``0`` if the information cannot be determined (the caller should
    treat this as "unknown — proceed optimistically").
    """
    import pathlib

    # ---- cgroup v2 (modern Docker / K8s) ----
    mem_max = pathlib.Path("/sys/fs/cgroup/memory.max")
    mem_cur = pathlib.Path("/sys/fs/cgroup/memory.current")
    if mem_max.exists() and mem_cur.exists():
        try:
            limit_str = mem_max.read_text().strip()
            current_str = mem_cur.read_text().strip()
            # "max" means no limit
            if limit_str.lower() != "max":
                limit = int(limit_str)
                current = int(current_str)
                avail = limit - current
                return max(avail, 0)
        except (OSError, ValueError):
            pass

    # ---- cgroup v1 (older Docker) ----
    limit_path = pathlib.Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    usage_path = pathlib.Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit_path.exists() and usage_path.exists():
        try:
            limit = int(limit_path.read_text().strip())
            usage = int(usage_path.read_text().strip())
            # v1 limit is often set to a very large number when "unlimited"
            # (typically 2**63-1 or 9223372036854771712)
            if limit < (1 << 62):
                avail = limit - usage
                return max(avail, 0)
        except (OSError, ValueError):
            pass

    # ---- Fallback: host memory (no container limit detected) ----
    try:
        import psutil

        return psutil.virtual_memory().available
    except Exception:
        return 0


class SchedulerUpdateWeightsMixin:

    def update_weights_from_disk(
        self: Scheduler, recv_req: UpdateWeightFromDiskReqInput
    ):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(
        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput
    ):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(
        self: Scheduler, recv_req: DestroyWeightsUpdateGroupReqInput
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(
        self: Scheduler, recv_req: UpdateWeightsFromTensorReqInput
    ):
        """Update the online model parameter from tensors."""
        if recv_req.disable_draft_model:
            worker = self.tp_worker
        else:
            worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(
        self: Scheduler, recv_req: UpdateWeightsFromIPCReqInput
    ):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def get_weights_by_name(self: Scheduler, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        # Pre-flight check: estimate total CPU pinned memory needed for all
        # cpu-backup tags and reject the entire request if insufficient.
        # torch_memory_saver calls cudaMallocHost in C++ pause(); if that
        # fails the process is killed with exit(1), so we check proactively.
        try:
            required_backup_bytes = 0
            if (
                GPU_MEMORY_TYPE_KV_CACHE in tags
                and self.server_args.enable_kv_cache_cpu_backup
            ):
                required_backup_bytes += (
                    self.tp_worker.model_runner.token_to_kv_pool.mem_usage
                    * 1024 * 1024 * 1024
                )
            if (
                GPU_MEMORY_TYPE_WEIGHTS in tags
                and self.server_args.enable_weights_cpu_backup
            ):
                required_backup_bytes += (
                    self.tp_worker.model_runner.weight_load_mem_usage
                    * 1024 * 1024 * 1024
                )

            if required_backup_bytes > 0:
                avail_bytes = _get_container_available_memory_bytes()
                min_required_bytes = int(required_backup_bytes * 1.2)
                if avail_bytes < min_required_bytes:
                    parts = []
                    if (
                        GPU_MEMORY_TYPE_KV_CACHE in tags
                        and self.server_args.enable_kv_cache_cpu_backup
                    ):
                        kv_gb = self.tp_worker.model_runner.token_to_kv_pool.mem_usage
                        parts.append(f"KV cache ~{kv_gb:.1f} GB")
                    if (
                        GPU_MEMORY_TYPE_WEIGHTS in tags
                        and self.server_args.enable_weights_cpu_backup
                    ):
                        w_gb = self.tp_worker.model_runner.weight_load_mem_usage
                        parts.append(f"weights ~{w_gb:.1f} GB")
                    msg = (
                        f"Insufficient host memory for CPU backup ({' + '.join(parts)}): "
                        f"requires ~{required_backup_bytes / (1024**3):.1f} GB "
                        f"(with 1.2x safety margin: {min_required_bytes / (1024**3):.1f} GB), "
                        f"but only {avail_bytes / (1024**3):.1f} GB available. "
                        f"Free up host memory or disable the cpu-backup flags."
                    )
                    logger.error(msg)
                    return ReleaseMemoryOccupationReqOutput(
                        success=False, message=msg
                    )
                warn_threshold = int(required_backup_bytes * 1.5)
                if avail_bytes < warn_threshold:
                    logger.warning(
                        f"CPU backup requires ~{required_backup_bytes / (1024**3):.1f} GB "
                        f"of pinned CPU memory, only {avail_bytes / (1024**3):.1f} GB available. "
                        f"This may cause heavy swapping."
                    )
        except Exception:
            # If we can't determine memory, proceed optimistically
            pass

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            if self.server_args.enable_kv_cache_cpu_backup:
                # When enable_cpu_backup is set on the KV cache region, torch_memory_saver
                # will automatically copy GPU→CPU pinned memory on pause().
                # We additionally snapshot the radix tree topology so that prefix cache
                # hits are preserved across the sleep/wakeup cycle.
                self.stashed_kv_radix_snapshot = (
                    self.tree_cache.export_snapshot()
                    if hasattr(self.tree_cache, "export_snapshot")
                    else None
                )
                logger.info(
                    "KV cache CPU backup enabled: saving radix tree snapshot before sleep."
                )
                # pause() triggers GPU→CPU copy via torch_memory_saver cpu_backup path
                self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
                # Flush only the metadata (free-list and radix tree) without touching the
                # already-paused GPU tensors.  We do NOT call self.flush_cache() here
                # because that would clear the slot assignments we just snapshotted.
            else:
                self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
                self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            # resume() restores GPU memory (CPU→GPU copy via torch_memory_saver cpu_backup path)
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            if (
                self.server_args.enable_kv_cache_cpu_backup
                and hasattr(self, "stashed_kv_radix_snapshot")
                and self.stashed_kv_radix_snapshot is not None
                and hasattr(self.tree_cache, "import_snapshot")
            ):
                # Restore radix tree so that prefix cache hits work immediately after wakeup
                success = self.tree_cache.import_snapshot(self.stashed_kv_radix_snapshot)
                if success:
                    logger.info(
                        "KV cache CPU backup: radix tree snapshot restored after wakeup."
                    )
                else:
                    logger.error(
                        "KV cache CPU backup: radix tree snapshot restore FAILED. "
                        "Prefix cache will NOT be available. "
                        "Subsequent requests will need full prefill."
                    )
            if hasattr(self, "stashed_kv_radix_snapshot"):
                del self.stashed_kv_radix_snapshot

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            self.tp_worker.model_runner.check_weights(action=recv_req.action)
            return CheckWeightsReqOutput(success=True, message="Success.")
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
