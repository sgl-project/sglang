from __future__ import annotations

import hashlib
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
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
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
)
from sglang.srt.runtime_context import get_model
from sglang.srt.utils.weight_checker import overall_checksum
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

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


def _split_lora_named_tensors(named_tensors):
    """Partition a weight-update payload on the ``{lora_name}:`` prefix; a lora_A/lora_B
    tensor without one is a sender bug, not base data."""
    base_tensors, lora_tensors = [], []
    for name, tensor in named_tensors:
        if ":" in name:
            lora_tensors.append((name, tensor))
        else:
            assert ".lora_A." not in name and ".lora_B." not in name, (
                f"LoRA tensor {name!r} arrived without a '{{lora_name}}:' prefix"
            )
            base_tensors.append((name, tensor))
    return base_tensors, lora_tensors


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


class _WeightUpdateSession(msgspec.Struct, frozen=True):
    # recorded at begin so end finalizes the same runners
    selector: str
    # False: adapter-only session; base weights are neither unpacked nor accepted
    sync_base: bool = True
    loaded_weights: bool = False
    # version carried by the session's buckets; recorded only when end commits
    pending_version: Optional[str] = None


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
    # replicated on every TP rank, so a rejected call returns on all ranks before any barrier
    _session: Optional[_WeightUpdateSession] = None
    # streamed adapter tensors of the open session: {lora_name: {hf_key: tensor}}
    _lora_stash: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    # tensor-name set of each adapter's last applied stream; a change means a partial stream
    _lora_applied_names: Dict[str, frozenset] = field(default_factory=dict)

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

    def record_weight_version_after_update(self, weight_version: Optional[str]) -> None:
        if self._session is not None:
            if weight_version is not None:
                self._session = msgspec.structs.replace(
                    self._session, pending_version=weight_version
                )
            return
        self.scheduler.record_weight_version_change(new_version=weight_version)

    def _reject_base_tensors(self) -> Optional[str]:
        if self._session.sync_base:
            return None
        return "base tensors arrived in a sync_base=False weight-update session"

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
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
            # the served weights changed even if a draft runner failed afterwards
            if target_updated:
                self.flush_cache_after_weight_update(recv_req)
            if success:
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
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

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter, fanning out to the selected runners."""
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
                base_tensors, lora_tensors = _split_lora_named_tensors(weights)
                self._stash_lora_tensors(lora_tensors)
                success, message = True, "Succeeded to update parameter online."
                if base_tensors and (error := self._reject_base_tensors()):
                    success, message = False, error
                if base_tensors and success:
                    for _, runner in self._select_runners(recv_req.selector):
                        success, message = (
                            runner.weight_updater.load_weights_from_distributed(
                                base_tensors
                            )
                        )
                        if not success:
                            break
            if success:
                if base_tensors:
                    self._session = msgspec.structs.replace(
                        self._session, loaded_weights=True
                    )
                self.flush_cache_after_weight_update(recv_req)
                self.record_weight_version_after_update(recv_req.weight_version)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors on the selected runners."""
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
            load_format = recv_req.load_format
            if load_format == "flattened_bucket":
                # names live in the bucket metadata; a pure-base bucket keeps its format
                reconstructed = FlattenedTensorBucket(
                    flattened_tensor=named_tensors["flattened_tensor"],
                    metadata=named_tensors["metadata"],
                ).reconstruct_tensors()
                base_tensors, lora_tensors = _split_lora_named_tensors(reconstructed)
                if lora_tensors:
                    named_tensors, load_format = base_tensors, None
            else:
                base_tensors, lora_tensors = _split_lora_named_tensors(named_tensors)
                named_tensors = base_tensors
            # the stash outlives this RPC, but an IPC sender may reuse the bucket once it replies
            self._stash_lora_tensors(lora_tensors, copy_tensors=True)
            success, message = True, "Success"
            if base_tensors and (error := self._reject_base_tensors()):
                success, message = False, error
            if base_tensors and success:
                for _, runner in self._select_runners(recv_req.selector):
                    success, message = runner.weight_updater.update_weights_from_tensor(
                        named_tensors=named_tensors,
                        load_format=load_format,
                    )
                    if not success:
                        break
            if success:
                if base_tensors:
                    self._session = msgspec.structs.replace(
                        self._session, loaded_weights=True
                    )
                self.flush_cache_after_weight_update(recv_req)
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
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
            # the served weights changed even if a draft runner failed afterwards
            if target_updated:
                self.flush_cache_after_weight_update(recv_req)
            if success:
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
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

    def begin_weight_update(self, recv_req: BeginWeightUpdateReqInput):
        """Open the session: restore in-place-packed weights on the selected runners."""
        if self._session is not None:
            return BeginWeightUpdateReqOutput(
                success=False,
                message="a weight-update session is already open; "
                "call end_weight_update() first",
            )
        if recv_req.sync_base:
            for _, runner in self._select_runners(recv_req.selector):
                runner.weight_updater.begin_weight_update()
        self._session = _WeightUpdateSession(
            selector=recv_req.selector, sync_base=recv_req.sync_base
        )
        self._lora_stash = {}
        torch.distributed.barrier(group=self.tp_cpu_group)
        return BeginWeightUpdateReqOutput(success=True, message="Success")

    def end_weight_update(self, recv_req: EndWeightUpdateReqInput):
        """Finalize the runners begin opened; post_load_weights only if no load ran (P2P/RDMA)."""
        if self._session is None:
            return EndWeightUpdateReqOutput(
                success=False,
                message="no weight-update session is open; call begin_weight_update() first",
            )
        session, self._session = self._session, None
        if session.sync_base:
            run_post_load = not session.loaded_weights
            for _, runner in self._select_runners(session.selector):
                runner.weight_updater.end_weight_update(run_post_load=run_post_load)
        if recv_req.abort:
            self._lora_stash = {}
            success, message = True, "Aborted: streamed adapters discarded"
        else:
            success, message = self._apply_lora_stash(recv_req.expected_lora_checksums)
            if success:
                self.record_weight_version_after_update(session.pending_version)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return EndWeightUpdateReqOutput(success=success, message=message)

    def forget_lora_adapter(self, lora_name: str) -> None:
        """A re-registered or unloaded name is a new adapter identity and may stream a
        different tensor set."""
        self._lora_applied_names.pop(lora_name, None)

    def _stash_lora_tensors(self, lora_tensors, *, copy_tensors: bool = False) -> None:
        copied_devices = set()
        for prefixed_name, tensor in lora_tensors:
            lora_name, hf_key = prefixed_name.split(":", 1)
            if isinstance(tensor, LocalSerializedTensor):
                tensor = tensor.get(self.tp_worker.model_runner.tp_rank)
            assert isinstance(tensor, torch.Tensor), (
                f"streamed LoRA tensor {prefixed_name!r} must arrive as a plain tensor"
            )
            if copy_tensors:
                tensor = tensor.clone()
                if tensor.is_cuda:
                    copied_devices.add(tensor.device)
            self._lora_stash.setdefault(lora_name, {})[hf_key] = tensor
        for device in copied_devices:
            torch.cuda.current_stream(device).synchronize()

    def _apply_lora_stash(
        self, expected_checksums: Optional[Dict[str, Dict[str, str]]]
    ) -> Tuple[bool, str]:
        """Hand each streamed adapter to the LoRA manager whole (config from registration,
        upsert in place), then clear the stash."""
        if expected_checksums is not None and set(expected_checksums) != set(
            self._lora_stash
        ):
            return False, (
                f"[LORA-CHECK] streamed adapters {sorted(self._lora_stash)} do not "
                f"match the expected manifest {sorted(expected_checksums)}"
            )
        if not self._lora_stash:
            return True, "Success"
        lora_manager = self.tp_worker.model_runner.lora_manager
        if lora_manager is None:
            return False, "streamed LoRA tensors require --enable-lora"
        for lora_name in sorted(self._lora_stash):
            tensors = self._lora_stash[lora_name]
            names = frozenset(tensors)
            if expected_checksums is not None:
                expected = expected_checksums[lora_name]
                if set(expected) != set(tensors):
                    return False, (
                        f"[LORA-CHECK] adapter {lora_name!r}: streamed tensor names "
                        f"do not match the expected manifest"
                    )
                for name in sorted(tensors):
                    if _sha256_tensor(tensors[name]) != expected[name]:
                        return False, (
                            f"[LORA-CHECK] adapter {lora_name!r}: checksum mismatch for {name!r}"
                        )
            prev = self._lora_applied_names.get(lora_name)
            if prev is not None and prev != names:
                return False, (
                    f"streamed adapter {lora_name!r} arrived with a different tensor "
                    f"set than its previous sync (partial stream?)"
                )
            result = lora_manager.apply_streamed_adapter(lora_name, tensors)
            if not result.success:
                return False, result.error_message
            self._lora_applied_names[lora_name] = names
        self._lora_stash = {}
        return True, "Success"

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
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
