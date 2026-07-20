from __future__ import annotations

import hashlib
import logging
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    BeginRemoteInstanceWeightTransferReqInput,
    BeginRemoteInstanceWeightTransferReqOutput,
    ChecksumInfo,
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
    ReleaseRemoteInstanceWeightTransferReqInput,
    ReleaseRemoteInstanceWeightTransferReqOutput,
    RenewRemoteInstanceWeightTransferReqInput,
    RenewRemoteInstanceWeightTransferReqOutput,
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

logger = logging.getLogger(__name__)


def _get_draft_model_runner(draft_worker):
    # DFlash / FrozenKVMTP workers expose draft_model_runner directly
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    # EAGLEWorkerV2: _draft_worker.draft_runner
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None:
        runner = getattr(inner, "draft_runner", None)
        if runner is not None:
            return runner
    return None


def _merge_checksum_payloads(target: Dict, draft: Dict) -> Dict:
    merged_checksums = dict(target["checksums"])
    for name, chk in draft["checksums"].items():
        merged_checksums[f"draft.{name}"] = chk
    h = hashlib.sha256()
    for name in sorted(merged_checksums):
        h.update(name.encode())
        h.update(merged_checksums[name].encode())
    target["checksums"] = merged_checksums
    target["per_gpu_checksum"] = h.hexdigest()
    return target


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    world_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    remote_weight_transfer_cpu_group: Any = None
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    remote_weight_transfer_leases: Dict[str, str] = field(default_factory=dict)
    remote_weight_transfer_executor: Optional[ThreadPoolExecutor] = field(
        default=None, init=False, repr=False
    )
    remote_weight_transfer_pending: List[Tuple[Future, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    @staticmethod
    def _commit_weight_runtime_revision(worker) -> None:
        runner = getattr(worker, "model_runner", None)
        if getattr(runner, "weight_snapshot_coordinator", None) is None:
            return
        runner.commit_weight_runtime_revision()

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            success, message = self.tp_worker.update_weights_from_disk(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_disk(recv_req)
            if tp_success:
                self._commit_weight_runtime_revision(self.tp_worker)
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
            )

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success=success, message=message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success=success, message=message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        with self._observe_weight_load("distributed"):
            success, message = self.tp_worker.update_weights_from_distributed(recv_req)
            if success:
                self._commit_weight_runtime_revision(self.tp_worker)
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        with self._observe_weight_load("tensor"):
            if recv_req.disable_draft_model:
                worker = self.tp_worker
            else:
                worker = self.draft_worker or self.tp_worker
            success, message = worker.update_weights_from_tensor(recv_req)
            if success:
                self._commit_weight_runtime_revision(worker)
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self._commit_weight_runtime_revision(self.tp_worker)
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def _defer_remote_instance_weight_transfer(self, operation, recv_req) -> None:
        if self.remote_weight_transfer_executor is None:
            self.remote_weight_transfer_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sglang-weight-transfer",
            )
        future = self.remote_weight_transfer_executor.submit(operation, recv_req)
        self.remote_weight_transfer_pending.append((future, recv_req))

    def defer_begin_remote_instance_weight_transfer(
        self, recv_req: BeginRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.begin_remote_instance_weight_transfer, recv_req
        )

    def defer_release_remote_instance_weight_transfer(
        self, recv_req: ReleaseRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.release_remote_instance_weight_transfer, recv_req
        )

    def defer_renew_remote_instance_weight_transfer(
        self, recv_req: RenewRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.renew_remote_instance_weight_transfer, recv_req
        )

    @staticmethod
    def _remote_instance_weight_transfer_failure(recv_req, error: Exception):
        kwargs = {
            "transfer_id": recv_req.transfer_id,
            "success": False,
            "message": str(error),
        }
        if isinstance(recv_req, BeginRemoteInstanceWeightTransferReqInput):
            return BeginRemoteInstanceWeightTransferReqOutput(**kwargs)
        if isinstance(recv_req, ReleaseRemoteInstanceWeightTransferReqInput):
            return ReleaseRemoteInstanceWeightTransferReqOutput(**kwargs)
        return RenewRemoteInstanceWeightTransferReqOutput(**kwargs)

    def check_pending_remote_instance_weight_transfers(self):
        completed = []
        remaining = []
        for future, recv_req in self.remote_weight_transfer_pending:
            if not future.done():
                remaining.append((future, recv_req))
                continue
            try:
                output = future.result()
            except Exception as error:
                logger.exception("Remote instance weight transfer control failed")
                output = self._remote_instance_weight_transfer_failure(recv_req, error)
            completed.append((output, recv_req))
        self.remote_weight_transfer_pending = remaining
        return completed

    def close_remote_instance_weight_transfer_executor(self) -> None:
        if self.remote_weight_transfer_executor is None:
            return
        self.remote_weight_transfer_executor.shutdown(wait=True)
        self.remote_weight_transfer_executor = None

    def begin_remote_instance_weight_transfer(
        self, recv_req: BeginRemoteInstanceWeightTransferReqInput
    ) -> BeginRemoteInstanceWeightTransferReqOutput:
        """Acquire one address-stable snapshot on every model rank."""
        collective_group = self.remote_weight_transfer_cpu_group or self.world_cpu_group
        local_manifest = None
        try:
            if recv_req.transfer_id in self.remote_weight_transfer_leases:
                raise RuntimeError(
                    f"remote weight transfer already exists: {recv_req.transfer_id}"
                )
            local_manifest = (
                self.tp_worker.model_runner.get_remote_instance_weight_runtime_manifest(
                    model_id=recv_req.model_id,
                    revision=recv_req.revision,
                    transfer_id=recv_req.transfer_id,
                    lease_timeout_sec=recv_req.lease_timeout_sec,
                )
            )
            local_payload = (
                local_manifest
                if isinstance(local_manifest, dict)
                else msgspec.to_builtins(local_manifest)
            )
            local_result = {
                "success": True,
                "message": "Success.",
                "manifest": local_payload,
            }
        except Exception as error:
            local_result = {
                "success": False,
                "message": str(error),
                "manifest": None,
            }

        try:
            world_size = torch.distributed.get_world_size(group=collective_group)
            gathered = [None] * world_size
            torch.distributed.all_gather_object(
                gathered, local_result, group=collective_group
            )
        except Exception as error:
            if local_manifest is not None:
                self.tp_worker.model_runner.release_weight_runtime_manifest(
                    local_manifest["lease_id"]
                    if isinstance(local_manifest, dict)
                    else local_manifest.lease_id
                )
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=f"Failed to gather source runtime manifests: {error}",
            )

        failures = [item["message"] for item in gathered if not item["success"]]
        manifests = [item["manifest"] for item in gathered if item["success"]]
        if not failures:
            try:
                self._validate_remote_transfer_manifests(manifests, world_size)
            except Exception as error:
                failures.append(str(error))

        if failures:
            if local_manifest is not None:
                self.tp_worker.model_runner.release_weight_runtime_manifest(
                    local_manifest["lease_id"]
                    if isinstance(local_manifest, dict)
                    else local_manifest.lease_id
                )
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=" | ".join(failures),
            )

        local_lease_id = (
            local_manifest["lease_id"]
            if isinstance(local_manifest, dict)
            else local_manifest.lease_id
        )
        self.remote_weight_transfer_leases[recv_req.transfer_id] = local_lease_id
        return BeginRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=True,
            message="Success.",
            manifests=manifests,
        )

    def _gather_remote_weight_transfer_status(
        self, *, success: bool, message: str, operation: str
    ) -> Tuple[bool, str]:
        local_result = {"success": success, "message": message}
        collective_group = self.remote_weight_transfer_cpu_group or self.world_cpu_group
        try:
            world_size = torch.distributed.get_world_size(group=collective_group)
            gathered = [None] * world_size
            torch.distributed.all_gather_object(
                gathered, local_result, group=collective_group
            )
        except Exception as error:
            return False, f"Failed to gather source {operation} results: {error}"

        failures = [item["message"] for item in gathered if not item["success"]]
        if failures:
            return False, " | ".join(failures)
        return True, "Success."

    def renew_remote_instance_weight_transfer(
        self, recv_req: RenewRemoteInstanceWeightTransferReqInput
    ) -> RenewRemoteInstanceWeightTransferReqOutput:
        lease_id = self.remote_weight_transfer_leases.get(recv_req.transfer_id)
        if lease_id is None:
            local_success = False
            local_message = "Remote weight transfer does not exist or has expired."
        else:
            try:
                self.tp_worker.model_runner.renew_weight_runtime_manifest(
                    lease_id,
                    lease_timeout_sec=recv_req.lease_timeout_sec,
                )
                local_success = True
                local_message = "Success."
            except Exception as error:
                local_success = False
                local_message = str(error)
                is_active = getattr(
                    self.tp_worker.model_runner,
                    "has_weight_runtime_manifest_lease",
                    None,
                )
                if callable(is_active) and not is_active(lease_id):
                    self.remote_weight_transfer_leases.pop(recv_req.transfer_id, None)

        success, message = self._gather_remote_weight_transfer_status(
            success=local_success,
            message=local_message,
            operation="renewal",
        )
        return RenewRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=success,
            message=message,
        )

    @staticmethod
    def _validate_remote_transfer_manifests(manifests, world_size: int) -> None:
        if len(manifests) != world_size:
            raise RuntimeError(
                f"expected {world_size} source manifests, got {len(manifests)}"
            )
        if any(not manifest.get("tensors") for manifest in manifests):
            raise RuntimeError("every source rank must publish at least one tensor")

        worker_ids = {
            tensor["worker_id"]
            for manifest in manifests
            for tensor in manifest["tensors"][:1]
        }
        if len(worker_ids) != world_size:
            raise RuntimeError("source runtime manifest worker IDs are not unique")

        identities = {
            (
                manifest.get("model_id"),
                manifest.get("revision"),
                manifest.get("generation"),
            )
            for manifest in manifests
        }
        if len(identities) != 1:
            raise RuntimeError(
                "source runtime manifests do not describe one model generation"
            )

    def release_remote_instance_weight_transfer(
        self, recv_req: ReleaseRemoteInstanceWeightTransferReqInput
    ) -> ReleaseRemoteInstanceWeightTransferReqOutput:
        lease_id = self.remote_weight_transfer_leases.get(recv_req.transfer_id)
        if lease_id is None:
            local_success = True
            local_message = "Remote weight transfer was already released."
        else:
            is_active = getattr(
                self.tp_worker.model_runner,
                "has_weight_runtime_manifest_lease",
                None,
            )
            if callable(is_active) and not is_active(lease_id):
                self.remote_weight_transfer_leases.pop(recv_req.transfer_id, None)
                local_success = True
                local_message = "Remote weight transfer lease already expired."
            else:
                try:
                    self.tp_worker.model_runner.release_weight_runtime_manifest(
                        lease_id
                    )
                    self.remote_weight_transfer_leases.pop(recv_req.transfer_id, None)
                    local_success = True
                    local_message = "Success."
                except Exception as error:
                    if callable(is_active) and not is_active(lease_id):
                        self.remote_weight_transfer_leases.pop(
                            recv_req.transfer_id, None
                        )
                        local_success = True
                        local_message = "Remote weight transfer lease already expired."
                    else:
                        local_success = False
                        local_message = str(error)

        success, message = self._gather_remote_weight_transfer_status(
            success=local_success,
            message=local_message,
            operation="release",
        )
        return ReleaseRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=success,
            message=message,
        )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.release_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.release_memory_occupation()
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

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
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
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.resume_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(
                action=recv_req.action, allow_quant_error=recv_req.allow_quant_error
            )

            if self.draft_worker is not None:
                draft_runner = _get_draft_model_runner(self.draft_worker)
                if draft_runner is not None:
                    draft_payload = draft_runner.check_weights(
                        action=recv_req.action,
                        allow_quant_error=recv_req.allow_quant_error,
                    )
                    if payload is not None and draft_payload is not None:
                        payload = _merge_checksum_payloads(payload, draft_payload)

            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            if payload is not None:
                # Normalize to one ChecksumInfo per rank so the wire shape is a
                # uniform List[ChecksumInfo] (tp==1 becomes a single-element list).
                per_rank = payload if isinstance(payload, list) else [payload]
                payload = [msgspec.convert(p, ChecksumInfo) for p in per_rank]
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.weight_exporter.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.weight_exporter.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        self.tp_worker.model_runner.weight_exporter.save_sharded_model(
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
