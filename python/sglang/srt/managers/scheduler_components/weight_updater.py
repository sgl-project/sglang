from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

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
    PullWeightsReqInput,
    PullWeightsReqOutput,
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
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils.weight_checker import overall_checksum

logger = logging.getLogger(__name__)


def _merge_checksum_payloads(role_payloads: List[Tuple[str, Dict]]) -> Dict:
    merged: Dict[str, str] = {}
    parallelism_infos = []
    for role, p in role_payloads:
        for name, chk in p["checksums"].items():
            # Add prefix for non-target roles
            key = name if role == "" else f"{role}.{name}"
            if key in merged:
                raise ValueError(f"checksum key collision: {key}")
            merged[key] = chk
        parallelism_infos.append({"role": role or "target", **p["parallelism_info"]})
    return {
        "checksums": merged,
        "per_gpu_checksum": overall_checksum(merged),
        "parallelism_info": parallelism_infos,
    }


def _parse_runner_selector(selector: str) -> Set[str]:
    """Map a {target, draft, all} weight-op selector to the set of roles it covers."""
    if selector == "all":
        return {"target", "draft"}
    if selector in ("target", "draft"):
        return {selector}
    raise ValueError(
        f"invalid selector {selector!r}; expected 'target', 'draft', or 'all'"
    )


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    _weight_update_in_progress: bool = False
    _weight_update_loaded: bool = False
    # Runner selector for the open session, recorded at begin_weight_update and
    # reused by end_weight_update so the same set is restored and finalized.
    _weight_update_selector: str = "all"

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

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            success, message = self.tp_worker.update_weights_from_disk(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_disk(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
            )

    def pull_weights(self, recv_req: PullWeightsReqInput):
        """Sync this host's local checkpoint up to recv_req.target_version.

        Every rank runs the pull; a per-host file lock collapses co-located
        ranks to one pull. Success is gathered across the TP group (all nodes),
        so the reply only reports success once every host holds a verified
        checkpoint.
        """
        from sglang.srt.weight_sync import local_checkpoint

        server_args = self.tp_worker.model_runner.server_args
        try:
            local_checkpoint.pull(
                local_checkpoint_dir=recv_req.local_checkpoint_dir,
                base_dir=server_args.model_path,
                source_dir=recv_req.source_dir,
                target_version=recv_req.target_version,
                pre_read_hook=server_args.custom_pull_weights_pre_read_hook,
            )
            success, message = True, "Success."
        except Exception:
            success, message = False, traceback.format_exc()
            logger.error(message)

        tp_size = (
            torch.distributed.get_world_size(group=self.tp_cpu_group)
            if torch.distributed.is_initialized()
            else 1
        )
        if tp_size > 1:
            results = [None] * tp_size
            torch.distributed.all_gather_object(
                results, (success, message), group=self.tp_cpu_group
            )
            success = all(ok for ok, _ in results)
            message = "; ".join(msg for ok, msg in results if not ok) or message
        return PullWeightsReqOutput(success=success, message=message)

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

    def iter_weight_update_workers(
        self, selector: str = "all"
    ) -> List[Tuple[str, Any]]:
        """Resolve a {target, draft, all} selector to (role, worker) pairs, target
        first: the target worker and, when present, the draft worker. This is the
        worker-level inclusion decision; each worker then contributes its own runners
        via iter_runners()."""
        parsed = _parse_runner_selector(selector)
        workers: List[Tuple[str, Any]] = []
        if "target" in parsed:
            workers.append(("target", self.tp_worker))
        if "draft" in parsed and self.draft_worker is not None:
            workers.append(("draft", self.draft_worker))
        return workers

    def get_model_runners(self, selector: str = "all") -> List[Tuple[str, Any]]:
        """Resolve a {target, draft, all} selector to (role, ModelRunner) pairs, target
        first. Derived from iter_weight_update_workers: each selected worker yields its
        own runners via iter_runners() — role "" for the target runner, draft roles
        from the draft worker."""
        runners: List[Tuple[str, Any]] = []
        for _, worker in self.iter_weight_update_workers(selector):
            runners += worker.iter_runners()
        return runners

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter, fanning out to the selected runners."""
        assert (
            self._weight_update_in_progress
        ), "update_weights_from_distributed requires an open begin_weight_update session"
        with self._observe_weight_load("distributed"):
            # The target (main) model owns this process's connection to the training
            # engine, so it receives the broadcast once; the received weights are then
            # loaded into each selected runner locally.
            try:
                weights = self.tp_worker.model_runner.receive_weights_from_distributed(
                    recv_req.names,
                    recv_req.dtypes,
                    recv_req.shapes,
                    recv_req.group_name,
                    recv_req.load_format,
                )
                for _, runner in self.get_model_runners(recv_req.selector):
                    runner.load_weights(weights)
                success, message = True, "Succeeded to update parameter online."
            except Exception as e:
                success = False
                message = (
                    f"Failed to update parameter online: {e}. The full weights of the "
                    "ModelRunner are partially updated. Please discard the whole weights."
                )
                logger.error(message)
            if success:
                self._weight_update_loaded = True
                self.flush_cache_after_weight_update(recv_req)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors, fanning out to the
        selected runners."""
        assert (
            self._weight_update_in_progress
        ), "update_weights_from_tensor requires an open begin_weight_update session"
        with self._observe_weight_load("tensor"):
            monkey_patch_torch_reductions()
            named_tensors = MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_worker.tp_rank]
            )
            success, message = True, "Success"
            for _, runner in self.get_model_runners(recv_req.selector):
                success, message = runner.update_weights_from_tensor(
                    named_tensors=named_tensors,
                    load_format=recv_req.load_format,
                )
                if not success:
                    break
            if success:
                self._weight_update_loaded = True
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
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def begin_weight_update(self, recv_req: BeginWeightUpdateReqInput):
        """Begin a weight-update session: restore in-place-packed weights to a
        loadable state on the selected runners (target and/or draft), so the draft
        model is prepared identically to the target. The selector is recorded and
        reused by end_weight_update so the same set is finalized."""
        assert (
            not self._weight_update_in_progress
        ), "begin_weight_update called while a weight-update session is already open"
        self._weight_update_selector = recv_req.selector
        for _, runner in self.get_model_runners(recv_req.selector):
            runner.begin_weight_update()
        self._weight_update_in_progress = True
        self._weight_update_loaded = False
        torch.distributed.barrier(group=self.tp_cpu_group)
        return BeginWeightUpdateReqOutput(success=True, message="Success")

    def end_weight_update(self, recv_req: EndWeightUpdateReqInput):
        """End the weight-update session on the runners begin_weight_update opened
        (its recorded selector): quant finalize on each, plus model.post_load_weights
        only when load_weights was bypassed this session (e.g. P2P/RDMA).
        """
        assert (
            self._weight_update_in_progress
        ), "end_weight_update called without begin_weight_update"
        run_post_load = not self._weight_update_loaded
        for _, runner in self.get_model_runners(self._weight_update_selector):
            runner.end_weight_update(run_post_load=run_post_load)
        self._weight_update_in_progress = False
        torch.distributed.barrier(group=self.tp_cpu_group)
        return EndWeightUpdateReqOutput(success=True, message="Success")

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        scheduler = self.scheduler
        assert self.is_fully_idle(
            ignore_retracted=scheduler is not None and scheduler._engine_paused
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
            runners = self.get_model_runners(recv_req.selector)

            def _check(role, runner):
                try:
                    return runner.check_weights(
                        action=recv_req.action,
                        allow_quant_error=recv_req.allow_quant_error,
                        skip_tensor_list=recv_req.skip_tensor_list,
                    )
                except Exception as e:
                    raise RuntimeError(f"[{role or 'target'}] {e}") from e

            if recv_req.action == "checksum":
                payload = _merge_checksum_payloads(
                    [(role, _check(role, runner)) for role, runner in runners]
                )
            else:
                for role, runner in runners:
                    _check(role, runner)
                payload = None

            if payload is not None and torch.distributed.is_initialized():
                tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
                if tp_size > 1:
                    all_payloads = [None] * tp_size
                    torch.distributed.all_gather_object(
                        all_payloads, payload, group=self.tp_cpu_group
                    )
                    payload = all_payloads
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self, params):
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
