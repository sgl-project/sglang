from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

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
    BeginWeightUpdateReqInput,
    BeginWeightUpdateReqOutput,
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    EndWeightUpdateReqInput,
    EndWeightUpdateReqOutput,
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
    UpdateWeightVersionReqInput,
    UpdateWeightVersionReqOutput,
)
from sglang.srt.runtime_context import get_model, get_parallel
from sglang.srt.utils.weight_checker import overall_checksum

logger = logging.getLogger(__name__)


def _merge_checksum_payloads(role_payloads: List[Tuple[str, Dict]]) -> Dict:
    merged: Dict[str, str] = {}
    parallelism_infos = []
    for role, p in role_payloads:
        for name, chk in p["checksums"].items():
            # only draft roles are prefixed, so target keys stay stable
            key = name if role == "target" else f"{role}.{name}"
            if key in merged:
                raise ValueError(f"checksum key collision: {key}")
            merged[key] = chk
        parallelism_infos.append(p["parallelism_info"])
    return {
        "checksums": merged,
        "per_gpu_checksum": overall_checksum(merged),
        "parallelism_info": parallelism_infos,
    }


def _parse_runner_selector(selector: str) -> Set[str]:
    if selector == "all":
        return {"target", "draft"}
    if selector in ("target", "draft"):
        return {selector}
    raise ValueError(
        f"invalid selector {selector!r}; expected 'target', 'draft', or 'all'"
    )


class _WeightUpdateSession(msgspec.Struct, frozen=True):
    # recorded at begin so end finalizes the same runners
    selector: str
    loaded_weights: bool = False


def _native_weight_transition(response_type, phase):
    """Keep the disabled path and existing runner/update implementation intact."""

    def decorate(function):
        @wraps(function)
        def wrapped(self, request):
            return self._native_weight_call(
                request, response_type, lambda: function(self, request), phase
            )

        return wrapped

    return decorate


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    # replicated on every TP rank, so a rejected call returns on all ranks before any barrier
    _session: Optional[_WeightUpdateSession] = None
    tp_cpu_group: Any = field(init=False)

    def __post_init__(self) -> None:
        self.tp_cpu_group = get_parallel().tp_group.cpu_group

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

    def flush_cache_after_weight_update(
        self, recv_req, success: bool, message: str
    ) -> Tuple[bool, str]:
        """Flush the cache the old weights produced. A flush refused because
        requests are still running or queued fails the update call."""
        service = getattr(getattr(self, "scheduler", None), "kv_control_service", None)
        if (
            getattr(getattr(service, "control", None), "enabled", False) is True
            and service.weights.active
        ):
            service.weights.request_flush(
                recv_req.flush_cache, recv_req.torch_empty_cache
            )
            return success, message  # Flush only after parameter work synchronizes.
        if recv_req.flush_cache and not self.flush_cache(
            empty_cache=recv_req.torch_empty_cache
        ):
            return False, (
                "Weights were updated, but the cache was not flushed because "
                "requests are still running or queued. Retract or finish them, "
                "then call /flush_cache."
            )
        return success, message

    def record_weight_version_after_update(self, weight_version: Optional[str]) -> None:
        self.scheduler.record_weight_version_change(new_version=weight_version)

    def _native_kv_weight_guard(self, response_type):
        service = getattr(getattr(self, "scheduler", None), "kv_control_service", None)
        if (
            getattr(getattr(service, "control", None), "enabled", False) is True
            and not service.namespace_keys
        ):
            return response_type(
                success=False,
                message="Weight updates require KV weight-namespace isolation, which is not integrated with this control profile",
            )
        return None

    def _native_finish_weights(self, service, result, response_type):
        service.weights.request_finish(result.success)
        try:
            device = service.cache.token_to_kv_pool_allocator.device
            module = torch.get_device_module(device)
            if str(device).split(":")[0] == "cuda":
                module.synchronize(device)
            else:
                module.synchronize()
            service.weights.mark_quiescent()
            fence = service.weights.fence
            if fence.flush_requested and not self.flush_cache(
                empty_cache=fence.empty_cache_requested
            ):
                return response_type(
                    success=False,
                    message="Parameter work finished, but cache flush could not complete; weight generation remains fenced",
                )
            committed = service.weights.finish(result.success, synchronized=True)
        except Exception as error:
            service.weights.fail()
            logger.exception("Native weight generation could not be finalized")
            return response_type(
                success=False,
                message=f"Weight generation remains fenced: {type(error).__name__}: {str(error)[:256]}",
            )
        if result.success and not committed:
            return msgspec.structs.replace(
                result,
                success=False,
                message="Weight session closed after a failed update; its cache namespace remains isolated",
            )
        return result

    def _native_weight_call(self, request, response_type, call, phase):
        service = getattr(getattr(self, "scheduler", None), "kv_control_service", None)
        if getattr(getattr(service, "control", None), "enabled", False) is not True:
            return call()
        try:
            service.weights._owner(require_keys=False)
        except ValueError as error:
            return response_type(success=False, message=str(error))
        if phase == "version":
            version = request.new_version
            if not service.weights.active and version == service.weights.version:
                return call()
            if service.weights.active:
                try:
                    service.weights.note_version(version)
                except ValueError as error:
                    return response_type(success=False, message=str(error))
                return call()
            phase = "single"
        if phase == "checker":
            if request.action != "reset_tensors":
                if (
                    service.weights.active
                    or GPU_MEMORY_TYPE_WEIGHTS in self.offload_tags
                ):
                    return response_type(
                        success=False,
                        message="Finish the weight update before inspecting parameters",
                    )
                return call()
            phase = "single"
        if (rejected := self._native_kv_weight_guard(response_type)) is not None:
            return rejected
        if str(service.cache.token_to_kv_pool_allocator.device).split(":")[0] not in (
            "cpu",
            "cuda",
        ):
            return response_type(
                success=False,
                message="Native weight-update validation currently supports CPU/CUDA owners",
            )
        session = self._session
        if self.offload_tags:
            return response_type(
                success=False,
                message="Resume offloaded model/cache resources before updating weights",
            )
        if phase == "begin" and session is not None:
            return call()  # Preserve the existing typed already-open refusal.
        if phase == "part" and session is None:
            return call()  # No mutation or namespace changes for missing begin.
        if phase == "end" and session is None:
            if service.weights.active and service.weights.fence.ready_to_finish:
                return self._native_finish_weights(
                    service,
                    response_type(success=True, message="Finalize parameter work"),
                    response_type,
                )
            return call()
        if phase == "single" and session is not None:
            return response_type(
                success=False,
                message="Close the multipart weight session before a standalone weight update",
            )
        try:
            selector = getattr(request, "selector", None)
            if selector is not None:
                _parse_runner_selector(selector)
                if not self._select_runners(selector):
                    return response_type(
                        success=False,
                        message="No selected weight-update runner is configured",
                    )
            if phase in ("single", "begin"):
                service.weights.begin(
                    getattr(
                        request, "weight_version", getattr(request, "new_version", None)
                    )
                )
            elif not service.weights.active:
                return response_type(
                    success=False, message="The native weight update was not prepared"
                )
            if phase == "part":
                service.weights.note_version(getattr(request, "weight_version", None))
            service.weights.request_flush(
                getattr(request, "flush_cache", False),
                getattr(request, "torch_empty_cache", False),
            )
        except (ValueError, TypeError) as error:
            return response_type(success=False, message=str(error)[:512])
        try:
            result = call()
        except Exception as error:
            service.weights.fail()
            logger.exception("Native parameter work failed; inference remains fenced")
            return response_type(
                success=False,
                message=f"Parameter work remains fenced: {type(error).__name__}: {str(error)[:256]}",
            )
        if not result.success:
            service.weights.fail()
        if phase in ("single", "end"):
            return self._native_finish_weights(service, result, response_type)
        return result

    @_native_weight_transition(UpdateWeightVersionReqOutput, "version")
    def update_weight_version(self, request: UpdateWeightVersionReqInput):
        self.record_weight_version_after_update(request.new_version)
        return UpdateWeightVersionReqOutput()

    @_native_weight_transition(UpdateWeightFromDiskReqOutput, "single")
    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        if (
            rejected := self._native_kv_weight_guard(UpdateWeightFromDiskReqOutput)
        ) is not None:
            return rejected
        with self._observe_weight_load("disk"):
            success, message = True, "Succeeded to update model weights."
            target_updated = False
            for role, runner in self._select_runners():
                success, message = runner.weight_updater.update_weights_from_disk(
                    recv_req.model_path,
                    recv_req.load_format,
                    recapture_cuda_graph=recv_req.recapture_cuda_graph,
                )
                if not success:
                    break
                target_updated |= role == "target"
            if success:
                self.record_weight_version_after_update(recv_req.weight_version)
            # the served weights changed even if a draft runner failed afterwards
            if target_updated:
                success, message = self.flush_cache_after_weight_update(
                    recv_req, success, message
                )
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

    def _select_runners(self, selector: str = "all") -> List[Tuple[str, Any]]:
        roles = _parse_runner_selector(selector)
        runners: List[Tuple[str, Any]] = []
        if "target" in roles:
            runners += self.tp_worker.weight_update_runners()
        if "draft" in roles and self.draft_worker is not None:
            runners += self.draft_worker.weight_update_runners()
        return runners

    @_native_weight_transition(UpdateWeightsFromDistributedReqOutput, "part")
    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter, fanning out to the selected runners."""
        if (
            rejected := self._native_kv_weight_guard(
                UpdateWeightsFromDistributedReqOutput
            )
        ) is not None:
            return rejected
        if self._session is None:
            return UpdateWeightsFromDistributedReqOutput(
                success=False,
                message="update_weights_from_distributed must run between "
                "begin_weight_update() and end_weight_update()",
            )
        with self._observe_weight_load("distributed"):
            # only the target runner joined the update group; drafts load its receive
            target = self.tp_worker.model_runner.weight_updater
            try:
                weights = target.receive_weights_from_distributed(
                    names=recv_req.names,
                    dtypes=recv_req.dtypes,
                    shapes=recv_req.shapes,
                    group_name=recv_req.group_name,
                    load_format=recv_req.load_format,
                )
            except Exception as e:
                success, message = False, f"Failed to receive weights: {e}"
                logger.error(message)
            else:
                success, message = True, "Succeeded to update parameter online."
                for _, runner in self._select_runners(recv_req.selector):
                    success, message = (
                        runner.weight_updater.load_weights_from_distributed(weights)
                    )
                    if not success:
                        break
            if success:
                self._session = msgspec.structs.replace(
                    self._session, loaded_weights=True
                )
                self.record_weight_version_after_update(recv_req.weight_version)
                success, message = self.flush_cache_after_weight_update(
                    recv_req, success, message
                )
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    @_native_weight_transition(UpdateWeightsFromTensorReqOutput, "part")
    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors on the selected runners."""
        if (
            rejected := self._native_kv_weight_guard(UpdateWeightsFromTensorReqOutput)
        ) is not None:
            return rejected
        if self._session is None:
            return UpdateWeightsFromTensorReqOutput(
                success=False,
                message="update_weights_from_tensor must run between "
                "begin_weight_update() and end_weight_update()",
            )
        with self._observe_weight_load("tensor"):
            named_tensors = self.tp_worker.deserialize_own_rank(
                recv_req.serialized_named_tensors
            )
            success, message = True, "Success"
            for _, runner in self._select_runners(recv_req.selector):
                success, message = runner.weight_updater.update_weights_from_tensor(
                    named_tensors=named_tensors,
                    load_format=recv_req.load_format,
                )
                if not success:
                    break
            if success:
                self._session = msgspec.structs.replace(
                    self._session, loaded_weights=True
                )
                self.record_weight_version_after_update(recv_req.weight_version)
                success, message = self.flush_cache_after_weight_update(
                    recv_req, success, message
                )
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    @_native_weight_transition(UpdateWeightsFromIPCReqOutput, "single")
    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        if (
            rejected := self._native_kv_weight_guard(UpdateWeightsFromIPCReqOutput)
        ) is not None:
            return rejected
        with self._observe_weight_load("ipc"):
            success, message = True, "Succeeded to update model weights."
            target_updated = False
            for role, runner in self._select_runners():
                success, message = runner.weight_updater.update_weights_from_ipc(
                    recv_req
                )
                if not success:
                    break
                target_updated |= role == "target"
            if success:
                self.record_weight_version_after_update(recv_req.weight_version)
            # the served weights changed even if a draft runner failed afterwards
            if target_updated:
                success, message = self.flush_cache_after_weight_update(
                    recv_req, success, message
                )
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def _assert_weight_cache_inactive(self, op: str) -> None:
        """Reject freeing/restoring model weights while the CUDA IPC weight
        cache is active: the weights are shared with the daemon via CUDA IPC, so
        freeing them would leave the daemon and every peer pointing at released
        memory.
        """
        mode = get_model().weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} of model weights is not supported while the "
                f"weight cache is active (--weight-cache-mode {mode}): the weights "
                f"are shared with the daemon via CUDA IPC, so freeing them would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )

    @_native_weight_transition(BeginWeightUpdateReqOutput, "begin")
    def begin_weight_update(self, recv_req: BeginWeightUpdateReqInput):
        """Open the session: restore in-place-packed weights on the selected runners."""
        if (
            rejected := self._native_kv_weight_guard(BeginWeightUpdateReqOutput)
        ) is not None:
            return rejected
        if self._session is not None:
            return BeginWeightUpdateReqOutput(
                success=False,
                message="a weight-update session is already open; "
                "call end_weight_update() first",
            )
        for _, runner in self._select_runners(recv_req.selector):
            runner.weight_updater.begin_weight_update()
        self._session = _WeightUpdateSession(selector=recv_req.selector)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return BeginWeightUpdateReqOutput(success=True, message="Success")

    @_native_weight_transition(EndWeightUpdateReqOutput, "end")
    def end_weight_update(self, recv_req: EndWeightUpdateReqInput):
        """Finalize the runners begin opened; post_load_weights only if no load ran (P2P/RDMA)."""
        if self._session is None:
            return EndWeightUpdateReqOutput(
                success=False,
                message="no weight-update session is open; call begin_weight_update() first",
            )
        run_post_load = not self._session.loaded_weights
        for _, runner in self._select_runners(self._session.selector):
            runner.weight_updater.end_weight_update(run_post_load=run_post_load)
        self._session = None
        torch.distributed.barrier(group=self.tp_cpu_group)
        return EndWeightUpdateReqOutput(success=True, message="Success")

    def _native_memory_guard(self, response_type):
        service = getattr(getattr(self, "scheduler", None), "kv_control_service", None)
        if getattr(getattr(service, "control", None), "enabled", False) is not True:
            return None
        if service.weights.active:
            return response_type(
                success=False,
                message="Finish the weight update before changing memory ownership",
            )
        try:
            service.require_quiescent_cache()
        except ValueError as error:
            return response_type(success=False, message=str(error))
        return None

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        if (
            rejected := self._native_memory_guard(ReleaseMemoryOccupationReqOutput)
        ) is not None:
            return rejected
        scheduler = self.scheduler
        assert self.is_fully_idle(
            ignore_waiting=scheduler is not None and scheduler._engine_paused
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
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
            self._assert_weight_cache_inactive("release_memory_occupation")
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
        if (
            rejected := self._native_memory_guard(ResumeMemoryOccupationReqOutput)
        ) is not None:
            return rejected
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self._assert_weight_cache_inactive("resume_memory_occupation")
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

    @_native_weight_transition(CheckWeightsReqOutput, "checker")
    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            role_payloads = []
            for role, runner in self._select_runners(recv_req.selector):
                p = runner.check_weights(
                    action=recv_req.action,
                    allow_quant_error=recv_req.allow_quant_error,
                    skip_tensor_list=recv_req.skip_tensor_list,
                    role=role,
                )
                if p is not None:
                    role_payloads.append((role, p))
            payload = _merge_checksum_payloads(role_payloads) if role_payloads else None

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
            assert draft_url is not None, (
                "draft_url must be provided when draft model is enabled"
            )
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
