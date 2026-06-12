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
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    PostProcessWeightsReqInput,
    PostProcessWeightsReqOutput,
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


class SchedulerUpdateWeightsMixin:
    def flush_cache_after_weight_update(self: Scheduler, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    def _quiesce_for_weight_update(self: Scheduler):
        """Drain in-flight forward work before any NCCL weight mutation.
        Synchronize forward_stream and schedule_stream to ensure all ranks are quiescent.
        """
        if self.enable_overlap:
            self.forward_stream.synchronize()
        self.schedule_stream.synchronize()
        if self.tp_cpu_group is not None:
            torch.distributed.barrier(group=self.tp_cpu_group)

    def update_weights_from_disk(
        self: Scheduler, recv_req: UpdateWeightFromDiskReqInput
    ):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        tp_success = success
        if success and self.draft_worker is not None:
            success, message = self.draft_worker.update_weights_from_disk(recv_req)
        if tp_success:
            self.flush_cache_after_weight_update(recv_req)
        if not success:
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
        self._quiesce_for_weight_update()
        if recv_req.disable_draft_model:
            worker = self.tp_worker
        else:
            worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_distributed(recv_req)
        if success:
            self.flush_cache_after_weight_update(recv_req)
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(
        self: Scheduler, recv_req: UpdateWeightsFromTensorReqInput
    ):
        """Update the online model parameter from tensors."""
        self._quiesce_for_weight_update()
        if recv_req.disable_draft_model:
            worker = self.tp_worker
        else:
            worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        if success:
            self.flush_cache_after_weight_update(recv_req)
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(
        self: Scheduler, recv_req: UpdateWeightsFromIPCReqInput
    ):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        self._quiesce_for_weight_update()
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        tp_success = success
        if success and self.draft_worker is not None:
            success, message = self.draft_worker.update_weights_from_ipc(recv_req)
        if tp_success:
            self.flush_cache_after_weight_update(recv_req)
        if not success:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def post_process_weights(self, recv_req: PostProcessWeightsReqInput):
        """Optional post-processing for updated weights (e.g., Marlin conversion)."""
        self._quiesce_for_weight_update()
        success, message = self.tp_worker.post_process_weights(recv_req)
        if self.tp_cpu_group is not None:
            torch.distributed.barrier(group=self.tp_cpu_group)
        return PostProcessWeightsReqOutput(success, message)

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

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if hasattr(self, "disagg_decode_prealloc_queue"):
                    self.disagg_decode_prealloc_queue.release_memory_occupation()
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                if hasattr(self, "disagg_prefill_bootstrap_queue"):
                    self.disagg_prefill_bootstrap_queue.release_memory_occupation()

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
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if hasattr(self, "disagg_decode_prealloc_queue"):
                    self.disagg_decode_prealloc_queue.resume_memory_occupation()
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                if hasattr(self, "disagg_prefill_bootstrap_queue"):
                    self.disagg_prefill_bootstrap_queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            selector = recv_req.selector if recv_req.selector is not None else "target"
            # Validate before mutating any runner so a bad selector cannot leave
            # weights partially reset (an empty string is rejected here, not
            # silently coerced to "target").
            if selector not in ("target", "draft", "all"):
                raise ValueError(
                    f"invalid selector {selector!r}; expected one of target/draft/all"
                )

            # Validate the action up front: the empty-draft-runner fast path below
            # returns without dispatching to WeightChecker.handle, so an action that
            # handle would reject (unsupported or since-removed) must be caught here
            # too, or it would slip through as a success on draft-only selections.
            if recv_req.action not in (
                "snapshot",
                "reset_tensors",
                "compare",
                "checksum",
            ):
                raise Exception(f"Unsupported action={recv_req.action!r}")

            # Byte-identical fast path: the default /weights_checker payload keeps
            # its exact top-level shape ({"checksums","parallelism_info"}, no
            # "runners", target keys unprefixed).
            if selector == "target":
                payload = self.tp_worker.model_runner.check_weights(
                    action=recv_req.action
                )
                return CheckWeightsReqOutput(
                    success=True, message="Success.", payload=payload
                )

            # Target first so its parallelism_info is the checksum base.
            runners = []
            if selector == "all":
                runners.append(("", self.tp_worker.model_runner))
            if selector in ("draft", "all") and self.draft_worker is not None:
                runners += self.draft_worker.iter_draft_runners()

            if not runners:
                payload = (
                    {"checksums": {}, "parallelism_info": None, "runners": []}
                    if recv_req.action == "checksum"
                    else None
                )
                return CheckWeightsReqOutput(
                    success=True,
                    message="no separate draft weights to check",
                    payload=payload,
                )

            def _check(role, r, **kwargs):
                try:
                    return r.check_weights(**kwargs)
                except Exception as e:
                    raise RuntimeError(f"[{role or 'target'}] {e}") from e

            if recv_req.action == "reset_tensors":
                # Selecting a runner resets its complete coverage, including any
                # storage it shares with another runner (e.g. embed/head tied to
                # the target via set_embed_and_head). The reset sentinel is
                # idempotent, so a shared storage written by several selected
                # runners ends at the same value regardless of order.
                for role, r in runners:
                    _check(role, r, action="reset_tensors")
                payload = None
            elif recv_req.action == "checksum":
                merged = {}
                runner_infos = []
                base_parallelism = None
                for role, r in runners:
                    p = _check(role, r, action="checksum")
                    for k, v in p["checksums"].items():
                        key = k if role == "" else f"{role}.{k}"
                        if key in merged:
                            raise ValueError(f"checksum key collision: {key}")
                        merged[key] = v
                    if base_parallelism is None:
                        base_parallelism = p["parallelism_info"]
                    runner_infos.append(
                        {"role": role or "target", **p["parallelism_info"]}
                    )
                payload = {
                    "checksums": merged,
                    "parallelism_info": base_parallelism,
                    "runners": runner_infos,
                }
            else:
                for role, r in runners:
                    _check(role, r, action=recv_req.action)
                payload = None

            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
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
