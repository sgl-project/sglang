# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import dataclasses
import pickle
import time
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from typing import Any, Iterator, List

import zmq

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    GetWeightsChecksumReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromTensorCheckerReqInput,
    UpdateWeightFromTensorReqInput,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    GetDisaggStatsReq,
    ListLorasReq,
    MergeLoraWeightsReq,
    ReleaseRealtimeSessionReq,
    SetLoraReq,
    ShutdownReq,
    UnmergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.ipc_array import (
    is_local_endpoint,
    spill_large_arrays_to_file_refs,
)
from sglang.multimodal_gen.runtime.managers.continuous_batching import (
    ContinuousBatchingError,
    ContinuousDenoisingCoordinator,
    ContinuousResponseGroup,
    DenoisingRequestState,
    validate_continuous_batching_request,
)
from sglang.multimodal_gen.runtime.managers.continuous_stage_worker import (
    AsyncContinuousStageWorker,
    async_stages_supported,
)
from sglang.multimodal_gen.runtime.managers.cpu_worker import CPUWorker
from sglang.multimodal_gen.runtime.managers.dynamic_batch_admission import (
    BatchAdmissionController,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    BatchMetricsWindow,
    OutputBatch,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_post_training_mixin import (
    SchedulerPostTrainingMixin,
)
from sglang.multimodal_gen.runtime.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.server_warmup import (
    SchedulerWarmupMixin,
    get_first_generation_req,
    is_warmup_req,
    should_return_warmup_result,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.trace_wrapper import DiffStage, trace_slice

logger = init_logger(__name__)

_MAX_RECV_REQS_PER_POLL = 1024
_BATCH_METRICS_LOG_INTERVAL = 5


@dataclasses.dataclass(slots=True)
class _PendingAdmission:
    """A request whose encode stages are running on the stage worker."""

    identity: bytes | None
    req: "Req"
    raw_req_snapshot: "Req | None"
    enqueue_time: float
    group_id: str | None = None
    group_index: int = 0
    group_size: int = 1


class Scheduler(SchedulerWarmupMixin, SchedulerPostTrainingMixin, SchedulerDisaggMixin):
    """
    Runs the main event loop for the rank 0 worker.
    It listens for external requests via ZMQ and coordinates with other workers.
    This class does NOT manage worker processes.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        port_args: PortArgs,
        task_pipes_to_slaves: list = None,
        result_pipes_from_slaves: list = None,
        local_rank: int | None = None,
    ):
        self.server_args = server_args
        self.port_args = port_args

        # local_rank is the physical GPU index for torch.cuda.set_device.
        # In non-disagg mode, it equals gpu_id. In disagg mode, it may differ
        # (e.g., denoiser rank 0 on physical GPU 1).
        if local_rank is None:
            local_rank = gpu_id

        set_global_server_args(server_args=server_args)

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)
        endpoint = server_args.scheduler_endpoint
        if gpu_id == 0:
            # router allocates identify (envelope) for each connection
            self.receiver, actual_endpoint = get_zmq_socket(
                self.context, zmq.ROUTER, endpoint, True
            )
            logger.info(f"Scheduler bind at endpoint: {actual_endpoint}")
        else:
            self.receiver = None
        from sglang.multimodal_gen.runtime.platforms import current_platform

        Exec_worker = CPUWorker if current_platform.is_cpu() else GPUWorker
        worker = Exec_worker(
            local_rank=local_rank,
            master_port=port_args.master_port,
            rank=gpu_id,
            server_args=server_args,
        )
        self.worker = worker
        self.task_pipes_to_slaves = task_pipes_to_slaves
        self.result_pipes_from_slaves = result_pipes_from_slaves
        self.gpu_id = gpu_id
        self._show_warmup_progress = gpu_id == 0
        self._running = True

        self.request_handlers = {
            SetLoraReq: self._handle_set_lora,
            MergeLoraWeightsReq: self._handle_merge_lora,
            UnmergeLoraWeightsReq: self._handle_unmerge_lora,
            Req: self._handle_generation,
            ListLorasReq: self._handle_list_loras,
            ShutdownReq: self._handle_shutdown,
            ReleaseRealtimeSessionReq: self._handle_release_realtime_session,
            GetDisaggStatsReq: self._handle_get_disagg_stats,
            UpdateWeightFromDiskReqInput: self._handle_update_weights_from_disk,
            UpdateWeightFromTensorReqInput: self._handle_update_weights_from_tensor,
            UpdateWeightFromTensorCheckerReqInput: (
                self._handle_update_weights_from_tensor_checker
            ),
            GetWeightsChecksumReqInput: self._handle_get_weights_checksum,
            ReleaseMemoryOccupationReqInput: self._handle_release_memory_occupation,
            ResumeMemoryOccupationReqInput: self._handle_resume_memory_occupation,
        }

        # FIFO queue entries: (identity, request, enqueue_ts_s)
        self.waiting_queue: deque[tuple[bytes | None, Any, float]] = deque()
        self._batching_max_size = server_args.batching_max_size
        self._batching_delay_s = server_args.batching_delay_ms / 1000.0
        self._batch_metrics_enabled = server_args.enable_batching_metrics
        self._batch_metrics_window = BatchMetricsWindow()
        self._batch_admission = BatchAdmissionController(server_args, gpu_id=local_rank)
        self._response_group_counter = 0
        self._next_admission_ticket = 0
        self._poller = zmq.Poller()
        if self.receiver is not None:
            self._poller.register(self.receiver, zmq.POLLIN)

        self.req_based_warmup_scheduled = False
        # warmup progress tracking
        self._warmup_total = 0
        self._warmup_processed = 0
        self._warmup_progress_bar: Any | None = None
        self._logged_server_ready_after_warmup = False

        # Maximum consecutive errors before terminating the event loop
        self._max_consecutive_errors = 3
        self._consecutive_error_count = 0

        self._init_disagg_state(server_args, local_rank)

        if self._batch_metrics_enabled:
            logger.info(
                "Batch metrics enabled; logging summary every %d dispatches.",
                _BATCH_METRICS_LOG_INTERVAL,
            )

    def get_disagg_metrics(self) -> dict | None:
        """Return disagg role metrics snapshot, or None if not in disagg mode."""
        if self._disagg_metrics is None:
            return None
        return self._disagg_metrics.snapshot().to_dict()

    def _handle_get_disagg_stats(self, _reqs: List[Any]) -> OutputBatch:
        """Handle stats request — return disagg metrics via OutputBatch.output."""
        stats = self.get_disagg_metrics()
        return OutputBatch(
            output=stats or {"role": "monolithic", "message": "not in disagg mode"}
        )

    def _handle_set_lora(self, reqs: List[Any]) -> OutputBatch:
        # TODO: return set status
        # TODO: return with SetLoRAResponse or something more appropriate
        req = reqs[0]
        return self.worker.set_lora(
            req.lora_nickname,
            req.lora_path,
            req.target,
            req.strength,
            req.merge_mode,
        )

    def _handle_merge_lora(self, reqs: List[Any]):
        req = reqs[0]
        return self.worker.merge_lora_weights(req.target, req.strength)

    def _handle_unmerge_lora(self, reqs: List[Any]) -> OutputBatch:
        req = reqs[0]
        return self.worker.unmerge_lora_weights(req.target)

    def _handle_list_loras(self, _reqs: List[Any]) -> OutputBatch:
        return self.worker.list_loras()

    def _handle_shutdown(self, _reqs: List[Any]) -> OutputBatch:
        self._running = False
        return OutputBatch()

    def _handle_release_realtime_session(self, reqs: List[Any]) -> OutputBatch:
        req = reqs[0]
        return self.worker.release_realtime_session(req.session_id)

    def _handle_update_weights_from_disk(self, reqs: List[Any]) -> OutputBatch:
        """Handle update_weights_from_disk request for RL workflows."""
        if self.worker.is_sleeping():
            raise RuntimeError(
                "Cannot update weights while the server is sleeping. "
                "Call resume_memory_occupation first."
            )
        return super()._handle_update_weights_from_disk(reqs)

    @staticmethod
    def _normalize_generation_reqs(reqs: list[Any]) -> list[Req]:
        if len(reqs) == 1 and isinstance(reqs[0], list):
            return reqs[0]
        return reqs

    def _dispatch_single_request(self, req_or_group: Any) -> OutputBatch:
        if isinstance(req_or_group, list):
            if not all(isinstance(req, Req) for req in req_or_group):
                return OutputBatch(
                    error=f"Unknown request group type: {type(req_or_group)}"
                )
            return self._handle_generation(req_or_group, allow_dynamic_batching=False)

        handler = self.request_handlers.get(type(req_or_group))
        if handler is None:
            return OutputBatch(error=f"Unknown request type: {type(req_or_group)}")
        return handler([req_or_group])

    def _dispatch_items(
        self, items: list[tuple[bytes | None, Any]]
    ) -> OutputBatch | list[OutputBatch]:
        """Dispatch ready queue items; several plain `Req`s form one dynamic batch."""
        reqs = [item[1] for item in items]
        if len(reqs) > 1 and all(isinstance(req, Req) for req in reqs):
            return self._handle_generation(reqs, allow_dynamic_batching=True)
        if len(reqs) > 1:
            return [self._dispatch_single_request(req) for req in reqs]
        return self._dispatch_single_request(reqs[0])

    def _handle_generation(
        self, reqs: list[Any], *, allow_dynamic_batching: bool = True
    ):
        """Dispatch generation requests, merging compatible requests when allowed."""
        reqs = self._normalize_generation_reqs(reqs)
        if self.worker.is_sleeping():
            raise RuntimeError(
                "Server is sleeping. Call resume_memory_occupation first."
            )
        warmup_reqs = [req for req in reqs if req.is_warmup]
        if warmup_reqs:
            self._ensure_warmup_progress_bar(warmup_reqs[0])

        # Use the head request trace context for scheduler-side dispatch work.
        req = reqs[0]
        req.trace_ctx.rebuild_thread_context()
        with trace_slice(
            req.trace_ctx,
            DiffStage.SCHEDULER_DISPATCH,
            thread_finish_flag=True,
        ):
            if len(reqs) == 1 or not allow_dynamic_batching:
                return self.worker.execute_forward(reqs)

            if self.server_args.pipeline_config.supports_native_grouped_requests():
                return self._execute_generation_grouped(reqs)

            merged_req = self._try_merge_generation_reqs(reqs)
            if merged_req is None:
                return self._execute_generation_sequential(reqs)

            batch_size = len(reqs)
            try:
                output_batch = self.worker.execute_forward([merged_req])
                if output_batch.error:
                    logger.error(
                        "Dynamic batch execution returned error. Skipping sequential fallback and returning errors: %s",
                        output_batch.error,
                    )
                    return self._build_dynamic_batch_error_outputs(
                        reqs=reqs,
                        error_msg=output_batch.error,
                    )

                split_outputs = self._split_batched_output(output_batch, reqs)
                if split_outputs is None:
                    logger.error(
                        "Failed to split dynamic batched output cleanly. Skipping sequential fallback and returning errors."
                    )
                    return self._build_dynamic_batch_error_outputs(
                        reqs=reqs,
                        error_msg="Dynamic batching failed: could not split merged output.",
                    )

                logger.info(
                    "Processed dynamic batch of %d/%d request(s) with max_delay=%.2fms",
                    batch_size,
                    self._batching_max_size,
                    self._batching_delay_s * 1000.0,
                )
                return split_outputs
            except Exception as e:
                logger.error(
                    "Dynamic batching failed (%s). Skipping sequential fallback and returning errors.",
                    e,
                    exc_info=True,
                )
                return self._build_dynamic_batch_error_outputs(
                    reqs=reqs,
                    error_msg=f"Dynamic batching failed: {e}",
                )

    def _execute_generation_grouped(self, reqs: List[Req]) -> List[OutputBatch]:
        batch_size = len(reqs)
        try:
            output_batch = self.worker.execute_forward(reqs)
            if output_batch.error:
                logger.error(
                    "Native grouped execution returned error. Returning per-request errors: %s",
                    output_batch.error,
                )
                return self._build_dynamic_batch_error_outputs(
                    reqs=reqs,
                    error_msg=output_batch.error,
                )

            split_outputs = self._split_batched_output(output_batch, reqs)
            if split_outputs is None:
                logger.error(
                    "Failed to split native grouped output cleanly. Returning per-request errors."
                )
                return self._build_dynamic_batch_error_outputs(
                    reqs=reqs,
                    error_msg="Native grouped execution failed: could not split output.",
                )

            logger.info(
                "Processed native grouped batch of %d/%d request(s) with max_delay=%.2fms",
                batch_size,
                self._batching_max_size,
                self._batching_delay_s * 1000.0,
            )
            return split_outputs
        except Exception as e:
            logger.error(
                "Native grouped execution failed (%s). Returning per-request errors.",
                e,
                exc_info=True,
            )
            return self._build_dynamic_batch_error_outputs(
                reqs=reqs,
                error_msg=f"Native grouped execution failed: {e}",
            )

    def _execute_generation_sequential(self, reqs: List[Req]) -> List[OutputBatch]:
        return [self.worker.execute_forward([req]) for req in reqs]

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))),
        )
        return ordered[index]

    def _make_batch_signature_value(self, value: Any):
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return tuple(
                (str(k), self._make_batch_signature_value(v))
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(self._make_batch_signature_value(v) for v in value)
        return repr(value)

    def _sampling_param_signature_items(self, req: Req) -> list[tuple[str, Any]] | None:
        """Return per-field sampling-param signature items, skipping batch_sig_exclude fields."""
        sp = req.sampling_params
        if sp is None:
            return None

        try:
            sp_fields = dataclasses.fields(sp)
        except Exception:
            return None

        return [
            (
                f.name,
                self._make_batch_signature_value(getattr(sp, f.name, None)),
            )
            for f in sp_fields
            if not f.metadata.get("batch_sig_exclude", False)
        ]

    def _diffusers_kwargs_signature_value(self, req: Req) -> Any:
        return self._make_batch_signature_value(
            (req.extra or {}).get("diffusers_kwargs")
        )

    def _build_dynamic_batch_signature(self, req: Req) -> tuple[Any, ...] | None:
        """Build the request compatibility signature for dynamic batching.

        The signature is built from `SamplingParams` fields, excluding fields
        marked with `batch_sig_exclude`, plus generation-affecting
        `extra.diffusers_kwargs`.
        """
        signature_items = self._sampling_param_signature_items(req)
        if signature_items is None:
            return None

        if req.extra:
            diffusers_kwargs = req.extra.get("diffusers_kwargs")
            if diffusers_kwargs:
                signature_items.append(
                    (
                        "diffusers_kwargs",
                        self._make_batch_signature_value(diffusers_kwargs),
                    )
                )

        return tuple(signature_items)

    def _get_cached_signature(self, req: Req) -> tuple[Any, ...] | None:
        cached = getattr(req, "_dynamic_batch_sig", None)
        if cached is not None:
            return cached
        sig = self._build_dynamic_batch_signature(req)
        req._dynamic_batch_sig = sig  # type: ignore[attr-defined]
        return sig

    def _find_sampling_param_mismatch_field(
        self, base_req: Req, candidate_req: Req
    ) -> str | None:
        base_items = self._sampling_param_signature_items(base_req)
        candidate_items = self._sampling_param_signature_items(candidate_req)
        if base_items is None or candidate_items is None:
            return None

        if len(base_items) != len(candidate_items):
            return "sampling_params"

        for (name, base_value), (candidate_name, candidate_value) in zip(
            base_items, candidate_items
        ):
            if name != candidate_name:
                return "sampling_params"
            if base_value != candidate_value:
                return f"sampling_params.{name}"

        base_diffusers_kwargs = self._diffusers_kwargs_signature_value(base_req)
        candidate_diffusers_kwargs = self._diffusers_kwargs_signature_value(
            candidate_req
        )
        if base_diffusers_kwargs != candidate_diffusers_kwargs:
            return "extra.diffusers_kwargs"

        return None

    def _get_dynamic_batch_reject_reason(
        self, base_req: Req, candidate_req: Req
    ) -> str | None:
        """Return the first reason `candidate_req` cannot batch with `base_req`, or None."""
        if self._can_dynamic_batch(base_req, candidate_req):
            return None

        if base_req.is_warmup or candidate_req.is_warmup:
            return "warmup"
        if self._has_realtime_session(base_req) or self._has_realtime_session(
            candidate_req
        ):
            return "realtime_session"
        if not isinstance(base_req.prompt, str) or not isinstance(
            candidate_req.prompt, str
        ):
            return "prompt_type"
        if (
            getattr(base_req, "image_path", None) is not None
            or getattr(candidate_req, "image_path", None) is not None
        ):
            return "image_conditioning"
        if base_req.return_file_paths_only != candidate_req.return_file_paths_only:
            return "return_file_paths_only"

        base_sig = self._get_cached_signature(base_req)
        candidate_sig = self._get_cached_signature(candidate_req)
        if base_sig is None or candidate_sig is None:
            return "signature_unavailable"

        return (
            self._find_sampling_param_mismatch_field(base_req, candidate_req)
            or "signature_mismatch"
        )

    @staticmethod
    def _has_realtime_session(req: Req) -> bool:
        return bool(req.realtime_session_id) or req.session is not None

    def _can_dynamic_batch(self, base_req: Req, candidate_req: Req) -> bool:
        """Return whether `candidate_req` can be merged into a batch with `base_req`."""
        if base_req.is_warmup or candidate_req.is_warmup:
            return False

        if self._has_realtime_session(base_req) or self._has_realtime_session(
            candidate_req
        ):
            return False

        if not isinstance(base_req.prompt, str) or not isinstance(
            candidate_req.prompt, str
        ):
            return False

        if (
            getattr(base_req, "image_path", None) is not None
            or getattr(candidate_req, "image_path", None) is not None
        ):
            return False
        if base_req.return_file_paths_only != candidate_req.return_file_paths_only:
            return False

        base_sig = self._get_cached_signature(base_req)
        cand_sig = self._get_cached_signature(candidate_req)
        return base_sig is not None and base_sig == cand_sig

    def _record_batch_dispatch_metrics(
        self,
        batch_size: int,
        queue_wait_ms: float,
        effective_max_batch_size: int,
        reject_reasons: list[str] | None = None,
        stop_reason: str | None = None,
        active_requests: int | None = None,
        queue_depth: int | None = None,
    ) -> None:
        if not self._batch_metrics_enabled:
            return

        effective_max_batch_size = max(1, effective_max_batch_size)
        logger.info(
            "Batch dispatch: size=%d/%d, user_max=%d, queue_wait=%.2fms, active=%s, queue_depth=%s, stop_reason=%s",
            batch_size,
            effective_max_batch_size,
            self._batching_max_size,
            max(queue_wait_ms, 0.0),
            active_requests if active_requests is not None else "-",
            queue_depth if queue_depth is not None else "-",
            stop_reason or "unspecified",
        )

        window = self._batch_metrics_window
        window.dispatches += 1
        window.total_requests += batch_size
        window.total_capacity += effective_max_batch_size
        window.batch_size_counts.update([batch_size])
        if batch_size > 1:
            window.merged_dispatches += 1
        if batch_size >= effective_max_batch_size:
            window.full_dispatches += 1
        window.wait_times_ms.append(max(queue_wait_ms, 0.0))
        if active_requests is not None:
            active_requests = max(0, int(active_requests))
            window.active_request_samples.append(active_requests)
            window.max_active_requests = max(
                window.max_active_requests, active_requests
            )
        if queue_depth is not None:
            window.queue_depth_samples.append(max(0, int(queue_depth)))
        if reject_reasons:
            window.reject_reasons.update(reject_reasons)

        if window.dispatches >= _BATCH_METRICS_LOG_INTERVAL:
            self._log_batch_metrics_summary()

    def _record_batch_reject_metrics(self, reason: str) -> None:
        if not self._batch_metrics_enabled:
            return
        logger.info("Batch admission rejected: %s", reason)
        self._batch_metrics_window.reject_reasons.update([reason])

    def _log_batch_metrics_summary(self) -> None:
        if not self._batch_metrics_enabled:
            return

        window = self._batch_metrics_window
        if window.dispatches == 0:
            return

        avg_size = window.total_requests / window.dispatches
        utilization = window.total_requests / max(1, window.total_capacity)
        avg_wait_ms = sum(window.wait_times_ms) / len(window.wait_times_ms)
        p95_wait_ms = self._percentile(window.wait_times_ms, 95.0)
        avg_active = (
            sum(window.active_request_samples) / len(window.active_request_samples)
            if window.active_request_samples
            else 0.0
        )
        avg_queue_depth = (
            sum(window.queue_depth_samples) / len(window.queue_depth_samples)
            if window.queue_depth_samples
            else 0.0
        )
        merged_rate = window.merged_dispatches / window.dispatches
        full_rate = window.full_dispatches / window.dispatches
        batch_sizes = ", ".join(
            f"{size}={count}"
            for size, count in sorted(window.batch_size_counts.items())
        )
        if not batch_sizes:
            batch_sizes = "none"
        top_rejects = ", ".join(
            f"{reason}={count}"
            for reason, count in window.reject_reasons.most_common(5)
        )
        if not top_rejects:
            top_rejects = "none"

        logger.info(
            "Batch stats (last %d dispatches): avg_size=%.2f, merged_rate=%.1f%%, full_rate=%.1f%%, utilization=%.1f%%, wait_avg=%.2fms, wait_p95=%.2fms, active_avg=%.2f, active_max=%d, queue_avg=%.2f, batch_sizes=%s, top_rejects=%s",
            window.dispatches,
            avg_size,
            merged_rate * 100.0,
            full_rate * 100.0,
            utilization * 100.0,
            avg_wait_ms,
            p95_wait_ms,
            avg_active,
            window.max_active_requests,
            avg_queue_depth,
            batch_sizes,
            top_rejects,
        )
        self._batch_metrics_window = BatchMetricsWindow()

    def _build_dynamic_batch_error_outputs(
        self,
        reqs: List[Req],
        error_msg: str,
    ) -> List[OutputBatch]:
        return [OutputBatch(error=error_msg) for _ in reqs]

    def _should_return_lightweight_warmup_result(self, processed_req: Any) -> bool:
        req = get_first_generation_req(processed_req)
        return (req is not None and bool(req.extra.get("server_internal_prewarm"))) or (
            is_warmup_req(processed_req) and should_return_warmup_result(processed_req)
        )

    def return_result(
        self,
        output_batch: OutputBatch,
        identity: bytes | None = None,
        should_not_return: bool = False,
    ):
        """
        replies to client, only on rank 0
        """
        if not should_not_return and self.receiver is not None and identity is not None:
            # if the server is local, use temp file to spill the frame array instead of
            # leaving it in OutputBatch to be pickled later
            if is_local_endpoint(self.server_args.scheduler_endpoint):
                with self._record_return_stage(
                    output_batch, "Scheduler.return_result.spill_arrays"
                ):
                    output_batch.output = spill_large_arrays_to_file_refs(
                        output_batch.output
                    )

            with self._record_return_stage(
                output_batch, "Scheduler.return_result.pickle"
            ):
                payload = pickle.dumps(output_batch)

            with self._record_return_stage(
                output_batch, "Scheduler.return_result.send"
            ):
                self.receiver.send_multipart([identity, b"", payload])

    @contextmanager
    def _record_return_stage(
        self, output_batch: OutputBatch, stage_name: str
    ) -> Iterator[None]:
        """helper function to record a stage metric"""
        start_time = time.perf_counter()
        yield
        if output_batch.metrics is not None:
            output_batch.metrics.record_stage(
                stage_name, time.perf_counter() - start_time
            )

    def _try_merge_generation_reqs(self, reqs: List[Req]) -> Req | None:
        """Create a batched generation request from compatible requests.

        Per-request seeds and output paths are stored in `extra` so downstream
        stages can preserve request ordering.
        """
        if len(reqs) <= 1:
            return reqs[0] if reqs else None

        base_req = reqs[0]
        for req in reqs[1:]:
            if not self._can_dynamic_batch(base_req, req):
                return None

        merged_req = deepcopy(base_req)
        merged_req.prompt = [req.prompt for req in reqs]

        merged_req.extra = deepcopy(merged_req.extra)
        merged_req.extra["dynamic_batch_seeds"] = [req.seed for req in reqs]
        merged_req.return_file_paths_only = base_req.return_file_paths_only
        if merged_req.return_file_paths_only:
            dynamic_output_paths: list[str] = []
            for req in reqs:
                for output_idx in range(req.num_outputs_per_prompt):
                    dynamic_output_paths.append(
                        req.output_file_path(req.num_outputs_per_prompt, output_idx)
                    )
            merged_req.extra["dynamic_batch_output_paths"] = dynamic_output_paths
        merged_req.request_id = f"dynamic_batch::{merged_req.request_id}"

        return merged_req

    @staticmethod
    def _count_first_dim(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return len(value)

        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                if len(shape) > 0:
                    return int(shape[0])
            except Exception:
                return None
        return None

    def _slice_batched_value(
        self, value: Any, start: int, end: int, total_items: int
    ) -> Any:
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            if len(value) == total_items:
                sliced = value[start:end]
                return list(sliced) if isinstance(value, list) else tuple(sliced)
            return deepcopy(value)

        value_items = self._count_first_dim(value)
        if value_items == total_items:
            try:
                return value[start:end]
            except Exception:
                pass

        # Scalar / non-batched metadata
        return deepcopy(value)

    def _split_batched_output(
        self, output_batch: OutputBatch, reqs: List[Req]
    ) -> List[OutputBatch] | None:
        """Split a merged result only when outputs map one-to-one to requests."""
        per_req_counts = [req.num_outputs_per_prompt for req in reqs]
        total_items = sum(per_req_counts)
        output_items = self._count_first_dim(output_batch.output)
        output_path_items = self._count_first_dim(output_batch.output_file_paths)

        if output_items is None and output_path_items is None:
            logger.warning(
                "Batched output has neither tensor outputs nor output_file_paths; cannot split safely."
            )
            return None

        if output_items is not None and output_items != total_items:
            logger.warning(
                "Unexpected batched output size: got %s items, expected %s",
                output_items,
                total_items,
            )
            return None
        if output_path_items is not None and output_path_items != total_items:
            logger.warning(
                "Unexpected batched output_file_paths size: got %s items, expected %s",
                output_path_items,
                total_items,
            )
            return None

        outputs: list[OutputBatch] = []
        start = 0
        for req_index, (req, req_count) in enumerate(zip(reqs, per_req_counts)):
            end = start + req_count
            metrics = (
                deepcopy(output_batch.metrics_list[req_index])
                if output_batch.metrics_list is not None
                and req_index < len(output_batch.metrics_list)
                else deepcopy(output_batch.metrics)
            )
            split = OutputBatch(
                output=self._slice_batched_value(
                    output_batch.output, start, end, total_items
                ),
                audio=self._slice_batched_value(
                    output_batch.audio, start, end, total_items
                ),
                audio_sample_rate=output_batch.audio_sample_rate,
                action_pred=self._slice_batched_value(
                    output_batch.action_pred, start, end, total_items
                ),
                trajectory_timesteps=self._slice_batched_value(
                    output_batch.trajectory_timesteps, start, end, total_items
                ),
                trajectory_latents=self._slice_batched_value(
                    output_batch.trajectory_latents, start, end, total_items
                ),
                trajectory_decoded=self._slice_batched_value(
                    output_batch.trajectory_decoded, start, end, total_items
                ),
                error=output_batch.error,
                output_file_paths=self._slice_batched_value(
                    output_batch.output_file_paths, start, end, total_items
                ),
                metrics=metrics,
                noise_pred=self._slice_batched_value(
                    output_batch.noise_pred, start, end, total_items
                ),
                peak_memory_mb=output_batch.peak_memory_mb,
            )
            if split.metrics is not None:
                split.metrics.request_id = req.request_id
            outputs.append(split)
            start = end

        return outputs

    def _dynamic_batching_enabled(self) -> bool:
        """Return whether this server and pipeline can use dynamic batching.

        This is the coarse gate; request-level checks decide which requests can
        actually be merged.
        """
        pipeline_config = self.server_args.pipeline_config
        supports_dynamic_batching = getattr(
            pipeline_config, "supports_dynamic_batching", None
        )
        if callable(supports_dynamic_batching):
            return self._batch_admission.enabled and supports_dynamic_batching()
        return self._batch_admission.enabled

    def _continuous_batching_enabled_for_pipeline(self) -> bool:
        if self.server_args.batching_mode != "continuous":
            return False
        supports = getattr(
            self.server_args.pipeline_config, "supports_continuous_batching", None
        )
        return callable(supports) and supports()

    def get_next_batch_to_run(self) -> list[tuple[bytes | None, Any]] | None:
        """Return the next dispatchable queue item or dynamic batch.

        Returns None when the head request is waiting for more compatible
        requests within the configured batching delay.
        """
        if not self.waiting_queue:
            return None

        if not self._dynamic_batching_enabled():
            identity, req, enqueue_time = self.waiting_queue.popleft()
            if isinstance(req, Req):
                self._record_batch_dispatch_metrics(
                    batch_size=1,
                    queue_wait_ms=(time.monotonic() - enqueue_time) * 1000.0,
                    effective_max_batch_size=1,
                    stop_reason="dynamic_disabled",
                )
            return [(identity, req)]

        identity, req, enqueue_time = self.waiting_queue[0]
        if not isinstance(req, Req):
            identity, req, _ = self.waiting_queue.popleft()
            return [(identity, req)]

        # If the head request itself is not eligible for dynamic batching
        # (e.g., image-conditioned i2i request), dispatch it immediately.
        if not self._can_dynamic_batch(req, req):
            identity, req, head_enqueue_time = self.waiting_queue.popleft()
            reject_reasons: list[str] = []
            if self._batch_metrics_enabled:
                reason = self._get_dynamic_batch_reject_reason(req, req)
                if reason is not None:
                    reject_reasons.append(f"head:{reason}")
            self._record_batch_dispatch_metrics(
                batch_size=1,
                queue_wait_ms=(time.monotonic() - head_enqueue_time) * 1000.0,
                effective_max_batch_size=1,
                reject_reasons=reject_reasons,
                stop_reason=reject_reasons[0] if reject_reasons else "head_ineligible",
            )
            return [(identity, req)]

        compatible_indices: list[int] = [0]
        compatible_reqs: list[Req] = [req]
        reject_reasons: list[str] = []
        for idx in range(1, len(self.waiting_queue)):
            if len(
                compatible_indices
            ) >= self._batching_max_size or self._batch_admission.batch_is_full(
                compatible_reqs
            ):
                break
            _identity, candidate_req, _enqueue_time = self.waiting_queue[idx]
            if isinstance(candidate_req, Req) and self._can_dynamic_batch(
                req, candidate_req
            ):
                admission_reject = self._batch_admission.reject_reason_for_candidate(
                    compatible_reqs, candidate_req
                )
                if admission_reject is None:
                    compatible_indices.append(idx)
                    compatible_reqs.append(candidate_req)
                elif self._batch_metrics_enabled:
                    reject_reasons.append(admission_reject)
            elif self._batch_metrics_enabled and isinstance(candidate_req, Req):
                reason = self._get_dynamic_batch_reject_reason(req, candidate_req)
                if reason is not None:
                    reject_reasons.append(reason)

        batch_len = len(compatible_indices)

        oldest_wait_s = time.monotonic() - enqueue_time

        should_wait_for_more = (
            batch_len < self._batching_max_size
            and not self._batch_admission.batch_is_full(compatible_reqs)
            and oldest_wait_s < self._batching_delay_s
        )
        if should_wait_for_more:
            return None

        batch_items: list[tuple[bytes | None, Any]] = [None] * batch_len
        for pos, idx in enumerate(reversed(compatible_indices)):
            item_identity, item_req, _ = self.waiting_queue[idx]
            batch_items[batch_len - 1 - pos] = (item_identity, item_req)
            del self.waiting_queue[idx]
        stop_reason = self._batch_admission.limit_reason_for_batch(compatible_reqs)
        if stop_reason is None:
            if batch_len >= self._batching_max_size:
                stop_reason = "max_size"
            elif reject_reasons:
                stop_reason = reject_reasons[0]
            elif oldest_wait_s >= self._batching_delay_s:
                stop_reason = "delay"
            else:
                stop_reason = "ready"
        self._record_batch_dispatch_metrics(
            batch_size=batch_len,
            queue_wait_ms=oldest_wait_s * 1000.0,
            effective_max_batch_size=self._batch_admission.max_admissible_batch_size(
                compatible_reqs[0]
            ),
            reject_reasons=reject_reasons,
            stop_reason=stop_reason,
        )
        return batch_items

    @staticmethod
    def _normalize_received_payload(
        identity: bytes, reqs: Any
    ) -> list[tuple[bytes, Any]]:
        """Normalize client payloads into queue entries.

        A single-item `[Req]` is one request; a multi-item `list[Req]` remains
        grouped as one logical request.
        """
        if not isinstance(reqs, list):
            return [(identity, reqs)]
        if not reqs:
            return []
        if all(isinstance(req, Req) for req in reqs):
            # AsyncSchedulerClient sends ordinary single requests as [Req].
            # Only multi-item list[Req] payloads represent a grouped multi-output request.
            if len(reqs) == 1:
                return [(identity, reqs[0])]
            return [(identity, reqs)]
        return [(identity, req) for req in reqs]

    def recv_reqs(self) -> List[tuple[bytes, Any]]:
        """
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj
        """
        if self.receiver is not None:
            try:
                recv_reqs: list[tuple[bytes, Any]] = []
                while len(recv_reqs) < _MAX_RECV_REQS_PER_POLL:
                    try:
                        # Accept valid REQ envelopes only, ignore malformed/probe frames.
                        parts = self.receiver.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break

                    try:
                        identity, payload = parts[0], parts[-1]
                        reqs = pickle.loads(payload) if len(parts) > 2 else []
                    except (pickle.UnpicklingError, IndexError, EOFError):
                        continue

                    recv_reqs.extend(self._normalize_received_payload(identity, reqs))
            except zmq.ZMQError:
                # re-raise or handle appropriately to let the outer loop continue
                raise
        else:
            recv_reqs = None

        # TODO: fix this condition
        if self.server_args.sp_degree != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.sp_group.rank,
                self.worker.sp_cpu_group,
                src=self.worker.sp_group.ranks[0],
            )

        if self.server_args.enable_cfg_parallel:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.cfg_group.rank,
                self.worker.cfg_cpu_group,
                src=self.worker.cfg_group.ranks[0],
            )

        if self.server_args.tp_size > 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.tp_group.rank,
                self.worker.tp_cpu_group,
                src=self.worker.tp_group.ranks[0],
            )

        assert recv_reqs is not None

        return recv_reqs

    def _get_continuous_admission_reject_reason(
        self,
        active_states: list[DenoisingRequestState],
        candidate_reqs: list[Req],
        pending_reqs: list[Req] | None = None,
    ) -> str | None:
        proposed = (
            [state.req for state in active_states]
            + list(pending_reqs or [])
            + candidate_reqs
        )
        if not proposed:
            return None

        limits = [self._batch_admission.limit_for(req) for req in proposed]
        max_batch_size = min(
            self._batching_max_size,
            *(limit.max_batch_size for limit in limits),
        )
        if len(proposed) > max_batch_size:
            return f"config_cap:{max_batch_size}"

        max_costs = [limit.max_cost for limit in limits if limit.max_cost is not None]
        if max_costs:
            max_cost = min(max_costs)
            batch_cost = self._batch_admission.estimate_batch_cost(proposed)
            if batch_cost > max_cost:
                return f"cost_budget:{batch_cost:.0f}>{max_cost:.0f}"
        return None

    def _get_continuous_effective_batch_cap(
        self,
        states: list[DenoisingRequestState],
    ) -> int:
        if not states:
            return self._batching_max_size
        limits = [self._batch_admission.limit_for(state.req) for state in states]
        return min(self._batching_max_size, *(limit.max_batch_size for limit in limits))

    def _return_continuous_result(
        self,
        output_batch: OutputBatch,
        identity: bytes | None,
        *,
        req_or_group: Req | list[Req] | None = None,
        is_warmup: bool = False,
    ) -> bool:
        """Return a continuous-batching result, respecting warmup policy."""
        should_not_return = is_warmup
        if (
            is_warmup
            and req_or_group is not None
            and should_return_warmup_result(req_or_group)
        ):
            output_batch.drop_payload_for_warmup()
            should_not_return = False
        try:
            self.return_result(
                output_batch,
                identity,
                should_not_return=should_not_return,
            )
            return True
        except zmq.ZMQError as e:
            logger.error("ZMQ error sending continuous batching reply: %s", e)
            return False

    @staticmethod
    def _is_continuous_request_group(item: Any) -> bool:
        return (
            isinstance(item, list)
            and item
            and all(isinstance(req, Req) for req in item)
        )

    @staticmethod
    def _continuous_item_is_warmup(item: Any) -> bool:
        if isinstance(item, Req):
            return item.is_warmup
        if Scheduler._is_continuous_request_group(item):
            return any(req.is_warmup for req in item)
        return is_warmup_req(item)

    def _validate_continuous_group_reqs(self, group_reqs: list[Req]) -> None:
        has_warmup = any(req.is_warmup for req in group_reqs)
        if has_warmup and not all(req.is_warmup for req in group_reqs):
            raise ContinuousBatchingError(
                "continuous batching does not support mixed warmup request groups"
            )

        validate_group_reqs = getattr(self.worker, "_validate_group_forward_reqs", None)
        if validate_group_reqs is not None:
            validate_group_reqs(group_reqs)

    @staticmethod
    def _move_recently_run_states_to_back(
        active_states: list[DenoisingRequestState],
        denoising_batch: list[DenoisingRequestState],
    ) -> list[DenoisingRequestState]:
        """Move just-run states to the back of the active list."""
        just_ran = {id(state) for state in denoising_batch}
        return [state for state in active_states if id(state) not in just_ran] + [
            state for state in active_states if id(state) in just_ran
        ]

    def _admit_single_continuous_request(
        self,
        continuous_coordinator: ContinuousDenoisingCoordinator,
        active_states: list[DenoisingRequestState],
        stage_worker: Any | None,
        pending_admissions: dict[int, "_PendingAdmission"],
        *,
        identity: bytes | None,
        req: Req,
        enqueue_time: float,
        group_id: str | None = None,
        group_index: int = 0,
        group_size: int = 1,
        pending_groups: dict[str, ContinuousResponseGroup] | None = None,
    ) -> None:
        """Admit one request synchronously or via the async stage worker."""

        def on_error(e: Exception, log: bool = False) -> None:
            if log:
                logger.error(
                    "Error admitting request for continuous batching: %s",
                    e,
                    exc_info=True,
                )
                message = f"continuous batching admission failed: {e}"
            else:
                message = str(e)
            if group_id is not None and pending_groups is not None:
                group = pending_groups.get(group_id)
                if group is not None:
                    group.set_member_output(group_index, OutputBatch(error=message))
                    self._return_response_group_if_complete(group_id, pending_groups)
            else:
                self._return_continuous_result(
                    OutputBatch(error=message),
                    identity,
                    req_or_group=req,
                    is_warmup=req.is_warmup,
                )

        if stage_worker is None or req.is_warmup:
            try:
                state = continuous_coordinator.prepare_request_state(
                    identity=identity,
                    req=req,
                    response_group_id=group_id,
                    response_index=group_index,
                    response_group_size=group_size,
                )
                state.queue_wait_ms = (time.monotonic() - enqueue_time) * 1000.0
                active_states.append(state)
            except ContinuousBatchingError as e:
                on_error(e)
            except Exception as e:
                on_error(e, log=True)
            return

        # Validate on the scheduler thread, then run encode on the side stream.
        try:
            validate_continuous_batching_request(req, self.server_args)
        except ContinuousBatchingError as e:
            on_error(e)
            return
        raw_req_snapshot = continuous_coordinator._snapshot_raw_request(req)
        ticket = self._next_admission_ticket
        self._next_admission_ticket += 1
        pending_admissions[ticket] = _PendingAdmission(
            identity=identity,
            req=req,
            raw_req_snapshot=raw_req_snapshot,
            enqueue_time=enqueue_time,
            group_id=group_id,
            group_index=group_index,
            group_size=group_size,
        )
        pipeline = continuous_coordinator.pipeline
        server_args = self.server_args
        try:
            stage_worker.submit(
                "encode",
                ticket,
                lambda: pipeline.run_stages_before_denoising(req, server_args),
            )
        except Exception as e:
            # Admission pre-checks queue capacity, so a submit failure means
            # the worker died or the invariant broke; fail the request loudly.
            pending_admissions.pop(ticket, None)
            on_error(e, log=True)

    def _admit_waiting_requests_to_continuous_loop(
        self,
        continuous_coordinator: ContinuousDenoisingCoordinator,
        active_states: list[DenoisingRequestState],
        pending_groups: dict[str, ContinuousResponseGroup],
        stage_worker: Any | None = None,
        pending_admissions: dict[int, "_PendingAdmission"] | None = None,
    ) -> None:
        """Admit queued requests into the active continuous denoising set."""
        if pending_admissions is None:
            pending_admissions = {}
        while self.waiting_queue:
            identity, req_or_group, enqueue_time = self.waiting_queue[0]
            has_inflight = bool(active_states) or bool(pending_admissions)

            if any(state.req.is_warmup for state in active_states) or (
                has_inflight and self._continuous_item_is_warmup(req_or_group)
            ):
                break

            if (
                stage_worker is not None
                and not self._continuous_item_is_warmup(req_or_group)
                and (
                    isinstance(req_or_group, Req)
                    or self._is_continuous_request_group(req_or_group)
                )
            ):
                needed = (
                    len(req_or_group)
                    if self._is_continuous_request_group(req_or_group)
                    else 1
                )
                if not stage_worker.can_submit("encode", needed):
                    # Bounded encode queue: apply backpressure by leaving the
                    # request in the waiting queue until capacity frees up.
                    break

            pending_reqs = [item.req for item in pending_admissions.values()]

            if self._is_continuous_request_group(req_or_group):
                group_reqs = req_or_group
                try:
                    self._validate_continuous_group_reqs(group_reqs)
                except Exception as e:
                    self.waiting_queue.popleft()
                    self._return_continuous_result(
                        OutputBatch(error=str(e)),
                        identity,
                        req_or_group=group_reqs,
                        is_warmup=self._continuous_item_is_warmup(group_reqs),
                    )
                    continue

                reject_reason = self._get_continuous_admission_reject_reason(
                    active_states,
                    group_reqs,
                    pending_reqs,
                )
                if reject_reason is not None:
                    if has_inflight:
                        break
                    self._record_batch_reject_metrics(reject_reason)
                    self.waiting_queue.popleft()
                    self._return_continuous_result(
                        OutputBatch(
                            error=(
                                "continuous batching admission rejected request group: "
                                f"{reject_reason}"
                            )
                        ),
                        identity,
                        req_or_group=group_reqs,
                        is_warmup=self._continuous_item_is_warmup(group_reqs),
                    )
                    continue

                self.waiting_queue.popleft()
                group_id = f"response_group::{self._response_group_counter}"
                self._response_group_counter += 1
                group = ContinuousResponseGroup(identity=identity, reqs=group_reqs)
                pending_groups[group_id] = group
                for index, req in enumerate(group_reqs):
                    self._admit_single_continuous_request(
                        continuous_coordinator,
                        active_states,
                        stage_worker,
                        pending_admissions,
                        identity=identity,
                        req=req,
                        enqueue_time=enqueue_time,
                        group_id=group_id,
                        group_index=index,
                        group_size=len(group_reqs),
                        pending_groups=pending_groups,
                    )
                self._return_response_group_if_complete(group_id, pending_groups)
                if self._continuous_item_is_warmup(group_reqs):
                    break
                continue

            if not isinstance(req_or_group, Req):
                if (active_states or pending_admissions) and not isinstance(
                    req_or_group, ShutdownReq
                ):
                    break
                self.waiting_queue.popleft()
                try:
                    result = self._dispatch_single_request(req_or_group)
                except Exception as e:
                    logger.error(
                        "Error executing control request in continuous scheduler loop: %s",
                        e,
                        exc_info=True,
                    )
                    result = OutputBatch(error=str(e))
                self._return_continuous_result(result, identity)
                if not self._running:
                    break
                continue

            reject_reason = self._get_continuous_admission_reject_reason(
                active_states,
                [req_or_group],
                pending_reqs,
            )
            if reject_reason is not None:
                if has_inflight:
                    break
                self._record_batch_reject_metrics(reject_reason)
                self.waiting_queue.popleft()
                self._return_continuous_result(
                    OutputBatch(
                        error=(
                            "continuous batching admission rejected request: "
                            f"{reject_reason}"
                        )
                    ),
                    identity,
                    req_or_group=req_or_group,
                    is_warmup=req_or_group.is_warmup,
                )
                continue

            self.waiting_queue.popleft()
            self._admit_single_continuous_request(
                continuous_coordinator,
                active_states,
                stage_worker,
                pending_admissions,
                identity=identity,
                req=req_or_group,
                enqueue_time=enqueue_time,
            )
            if req_or_group.is_warmup:
                break

    def _return_response_group_if_complete(
        self, group_id: str, pending_groups: dict[str, ContinuousResponseGroup]
    ) -> bool:
        group = pending_groups.get(group_id)
        if group is None:
            return False
        if not group.has_all_outputs:
            return False

        merged = group.merge_outputs(self.worker)
        self._log_warmup_result(merged, group.reqs, group.is_warmup)
        self._return_continuous_result(
            merged,
            group.identity,
            req_or_group=group.reqs,
            is_warmup=group.is_warmup,
        )
        del pending_groups[group_id]
        return True

    def _return_completed_request_state(
        self,
        state: DenoisingRequestState,
        pending_groups: dict[str, ContinuousResponseGroup],
    ) -> None:
        """Return a completed request or merge it into its response group."""
        output_batch = state.output_batch or OutputBatch(error=state.error)
        if state.response_group_id is None or state.response_group_size <= 1:
            is_warmup = state.req.is_warmup
            self._log_warmup_result(output_batch, state.req, is_warmup)
            self._return_continuous_result(
                output_batch,
                state.identity,
                req_or_group=state.req,
                is_warmup=is_warmup,
            )
            return

        group = pending_groups.get(state.response_group_id)
        if group is None:
            logger.error(
                "Missing continuous response group %s for request %s",
                state.response_group_id,
                state.request_id,
            )
            return
        group.set_member_output(state.response_index, output_batch)
        self._return_response_group_if_complete(
            state.response_group_id,
            pending_groups,
        )

    def _should_wait_for_more_initial_requests(
        self,
        active_states: list[DenoisingRequestState],
        pending_admissions: dict[int, "_PendingAdmission"] | None = None,
    ) -> bool:
        if (
            active_states
            or pending_admissions
            or not self.waiting_queue
            or self._batching_delay_s <= 0
        ):
            return False
        _identity, req_or_group, enqueue_time = self.waiting_queue[0]
        if isinstance(req_or_group, Req):
            queued_count = sum(
                1 for _, item, _ in self.waiting_queue if isinstance(item, Req)
            )
        elif self._is_continuous_request_group(req_or_group):
            queued_count = len(req_or_group)
        else:
            return False
        if queued_count >= self._batching_max_size:
            return False
        return (time.monotonic() - enqueue_time) < self._batching_delay_s

    def _process_stage_worker_results(
        self,
        continuous_coordinator: ContinuousDenoisingCoordinator,
        stage_worker: Any,
        active_states: list[DenoisingRequestState],
        pending_groups: dict[str, ContinuousResponseGroup],
        pending_admissions: dict[int, _PendingAdmission],
        pending_finalizes: dict[int, DenoisingRequestState],
        *,
        block_one: bool = False,
    ) -> None:
        """Apply finished async encode/finalize jobs to the active loop."""
        for result in stage_worker.poll_results(block_one=block_one):
            if result.kind == "encode":
                pending = pending_admissions.pop(result.ticket, None)
                if pending is None:
                    continue
                if result.error is not None:
                    self._fail_pending_admission(pending, pending_groups, result.error)
                    continue
                try:
                    state = continuous_coordinator.prepare_request_state_from_prepared(
                        identity=pending.identity,
                        prepared_req=result.value,
                        raw_req_snapshot=pending.raw_req_snapshot,
                        response_group_id=pending.group_id,
                        response_index=pending.group_index,
                        response_group_size=pending.group_size,
                    )
                    state.queue_wait_ms = (
                        time.monotonic() - pending.enqueue_time
                    ) * 1000.0
                    active_states.append(state)
                except Exception as e:
                    self._fail_pending_admission(pending, pending_groups, e)
            elif result.kind == "finalize":
                state = pending_finalizes.pop(result.ticket, None)
                if state is None:
                    continue
                if result.error is not None and state.output_batch is None:
                    state.set_error_output(
                        f"continuous batching completion failed: {result.error}"
                    )
                self._safe_return_completed_request_state(state, pending_groups)

    def _fail_pending_admission(
        self,
        pending: _PendingAdmission,
        pending_groups: dict[str, ContinuousResponseGroup],
        error: BaseException,
    ) -> None:
        message = (
            str(error)
            if isinstance(error, ContinuousBatchingError)
            else f"continuous batching admission failed: {error}"
        )
        if pending.group_id is not None:
            group = pending_groups.get(pending.group_id)
            if group is not None:
                group.set_member_output(pending.group_index, OutputBatch(error=message))
                self._return_response_group_if_complete(
                    pending.group_id, pending_groups
                )
            return
        self._return_continuous_result(
            OutputBatch(error=message),
            pending.identity,
            req_or_group=pending.req,
            is_warmup=pending.req.is_warmup,
        )

    def _safe_return_completed_request_state(
        self,
        state: DenoisingRequestState,
        pending_groups: dict[str, ContinuousResponseGroup],
    ) -> None:
        try:
            self._return_completed_request_state(state, pending_groups)
        except Exception as e:
            logger.error(
                "Error returning continuous batching result for request %s: %s",
                state.request_id,
                e,
                exc_info=True,
            )
            self._return_continuous_result(
                OutputBatch(error=f"continuous batching response failed: {e}"),
                state.identity,
                req_or_group=state.req,
                is_warmup=state.req.is_warmup,
            )

    def _drain_finalize_backlog(
        self,
        continuous_coordinator: ContinuousDenoisingCoordinator,
        stage_worker: Any,
        finalize_backlog: list[DenoisingRequestState],
        pending_finalizes: dict[int, DenoisingRequestState],
        pending_groups: dict[str, ContinuousResponseGroup],
    ) -> None:
        """Submit backlogged finalize jobs as bounded-queue capacity frees up."""
        from sglang.multimodal_gen.runtime.managers.continuous_stage_worker import (
            WorkerState,
        )

        if stage_worker.worker_state("finalize") is not WorkerState.RUNNING:
            # The finalize worker died; fail the backlog loudly instead of
            # silently decoding inline.
            while finalize_backlog:
                state = finalize_backlog.pop(0)
                state.set_error_output(
                    "continuous batching finalize worker is not running"
                )
                self._safe_return_completed_request_state(state, pending_groups)
            return
        while finalize_backlog and stage_worker.can_submit("finalize"):
            state = finalize_backlog.pop(0)
            ticket = self._next_admission_ticket
            self._next_admission_ticket += 1
            pending_finalizes[ticket] = state
            submitted = stage_worker.try_submit(
                "finalize",
                ticket,
                lambda state=state: (
                    continuous_coordinator.finalize_completed_request(state)
                ),
            )
            if not submitted:
                pending_finalizes.pop(ticket, None)
                finalize_backlog.insert(0, state)
                break

    def _run_continuous_batching_loop(self) -> None:
        logger.info(
            "Continuous batching scheduler enabled "
            "(max_active=%d, max_delay=%.2fms, policy=%s, async_stages=%s)",
            self._batching_max_size,
            self._batching_delay_s * 1000.0,
            getattr(self.server_args, "cb_schedule_policy", "largest"),
            async_stages_supported(self.server_args),
        )
        continuous_coordinator = ContinuousDenoisingCoordinator(
            worker=self.worker,
            server_args=self.server_args,
            batching_max_size=self._batching_max_size,
        )
        active_states: list[DenoisingRequestState] = []
        pending_groups: dict[str, ContinuousResponseGroup] = {}
        pending_admissions: dict[int, _PendingAdmission] = {}
        pending_finalizes: dict[int, DenoisingRequestState] = {}
        finalize_backlog: list[DenoisingRequestState] = []
        stage_worker = (
            AsyncContinuousStageWorker(
                queue_depth=int(
                    getattr(self.server_args, "cb_stage_queue_depth", 8) or 8
                )
            )
            if async_stages_supported(self.server_args)
            else None
        )
        rotate_policy = continuous_coordinator.schedule_policy == "rotate"

        drain_dir = getattr(self.server_args, "cb_drain_export_dir", None)
        if drain_dir:
            resumed = continuous_coordinator.import_states_from_dir(drain_dir)
            if resumed:
                logger.info(
                    "Resumed %d drained continuous batching request(s) from %s",
                    len(resumed),
                    drain_dir,
                )
                active_states.extend(resumed)

        while self._running:
            try:
                new_reqs = self.recv_reqs()
                new_reqs = self.process_received_reqs_with_req_based_warmup(new_reqs)
                now = time.monotonic()
                self.waiting_queue.extend(
                    [(identity, req, now) for identity, req in new_reqs]
                )
                self._consecutive_error_count = 0
            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    "Error receiving requests in continuous scheduler loop "
                    "(attempt %d/%d): %s",
                    self._consecutive_error_count,
                    self._max_consecutive_errors,
                    e,
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    raise RuntimeError(
                        "Continuous scheduler terminated after "
                        f"{self._max_consecutive_errors} consecutive errors. "
                        f"Last error: {e}"
                    ) from e

            if stage_worker is not None:
                self._process_stage_worker_results(
                    continuous_coordinator,
                    stage_worker,
                    active_states,
                    pending_groups,
                    pending_admissions,
                    pending_finalizes,
                )
                self._drain_finalize_backlog(
                    continuous_coordinator,
                    stage_worker,
                    finalize_backlog,
                    pending_finalizes,
                    pending_groups,
                )

            if self._should_wait_for_more_initial_requests(
                active_states, pending_admissions
            ):
                oldest_ts = self.waiting_queue[0][2]
                elapsed_ms = (time.monotonic() - oldest_ts) * 1000.0
                remaining_ms = max(0, self._batching_delay_s * 1000.0 - elapsed_ms)
                if remaining_ms > 0 and self.receiver is not None:
                    self._poller.poll(timeout=remaining_ms)
                elif remaining_ms > 0:
                    time.sleep(remaining_ms / 1000.0)
                continue

            self._admit_waiting_requests_to_continuous_loop(
                continuous_coordinator,
                active_states,
                pending_groups,
                stage_worker,
                pending_admissions,
            )

            denoising_batch = continuous_coordinator.select_next_step_batch(
                active_states
            )
            if not denoising_batch:
                if stage_worker is not None and (
                    stage_worker.pending > 0 or finalize_backlog
                ):
                    # No denoising work yet; block on the next async result.
                    self._process_stage_worker_results(
                        continuous_coordinator,
                        stage_worker,
                        active_states,
                        pending_groups,
                        pending_admissions,
                        pending_finalizes,
                        block_one=True,
                    )
                    self._drain_finalize_backlog(
                        continuous_coordinator,
                        stage_worker,
                        finalize_backlog,
                        pending_finalizes,
                        pending_groups,
                    )
                elif self.waiting_queue and self.receiver is not None:
                    self._poller.poll(
                        timeout=max(1, int(self._batching_delay_s * 1000))
                    )
                else:
                    time.sleep(0.001)
                continue

            try:
                completed = (
                    continuous_coordinator.run_selected_steps_and_advance_requests(
                        denoising_batch
                    )
                )
                self._record_batch_dispatch_metrics(
                    batch_size=len(denoising_batch),
                    queue_wait_ms=sum(state.queue_wait_ms for state in denoising_batch)
                    / max(1, len(denoising_batch)),
                    effective_max_batch_size=self._get_continuous_effective_batch_cap(
                        denoising_batch
                    ),
                    stop_reason="denoising_batch_run",
                    active_requests=len(active_states),
                    queue_depth=len(self.waiting_queue),
                )
                if self._batch_metrics_enabled:
                    logger.info(
                        "Continuous denoising step batch: size=%d/%d",
                        len(denoising_batch),
                        self._batching_max_size,
                    )
            except Exception as e:
                logger.error(
                    "Error executing continuous denoising step batch: %s",
                    e,
                    exc_info=True,
                )
                completed = []
                for state in denoising_batch:
                    continuous_coordinator.fail_request(
                        state,
                        f"Continuous denoising failed: {e}",
                    )
                    completed.append(state)

            for state in completed:
                if state.is_complete:
                    # Already failed; return the error immediately.
                    self._safe_return_completed_request_state(state, pending_groups)
                elif stage_worker is not None and not state.req.is_warmup:
                    # Offload VAE decode/postprocessing to keep denoising busy;
                    # the bounded finalize queue is fed through the backlog.
                    finalize_backlog.append(state)
                else:
                    continuous_coordinator.finalize_completed_request(state)
                    self._safe_return_completed_request_state(state, pending_groups)
            if stage_worker is not None:
                self._drain_finalize_backlog(
                    continuous_coordinator,
                    stage_worker,
                    finalize_backlog,
                    pending_finalizes,
                    pending_groups,
                )

            active_states = [state for state in active_states if not state.is_complete]
            if rotate_policy and len(active_states) > 1:
                active_states = self._move_recently_run_states_to_back(
                    active_states,
                    denoising_batch,
                )
            self._admit_waiting_requests_to_continuous_loop(
                continuous_coordinator,
                active_states,
                pending_groups,
                stage_worker,
                pending_admissions,
            )

        # Flush async work and drain-export remaining in-flight requests.
        if stage_worker is not None:
            while stage_worker.pending > 0 or finalize_backlog:
                self._drain_finalize_backlog(
                    continuous_coordinator,
                    stage_worker,
                    finalize_backlog,
                    pending_finalizes,
                    pending_groups,
                )
                if stage_worker.pending > 0:
                    self._process_stage_worker_results(
                        continuous_coordinator,
                        stage_worker,
                        active_states,
                        pending_groups,
                        pending_admissions,
                        pending_finalizes,
                        block_one=True,
                    )
            stage_worker.shutdown()

        in_flight = [state for state in active_states if not state.is_complete]
        if in_flight and drain_dir:
            exported = continuous_coordinator.export_states_to_dir(in_flight, drain_dir)
            logger.info(
                "Drain-exported %d/%d in-flight continuous batching request(s) to %s",
                len(exported),
                len(in_flight),
                drain_dir,
            )

        self._log_batch_metrics_summary()

        if self.receiver is not None:
            self.receiver.close()
        self._cleanup_disagg()
        self.context.destroy(linger=0)

    def event_loop(self) -> None:
        """
        The main event loop that listens for ZMQ requests.
        Handles abortion
        """
        # Pool mode: all roles use the pool event loop
        if self._disagg_role != RoleType.MONOLITHIC:
            self._disagg_event_loop()
            return

        if self.server_args.batching_mode == "continuous":
            if not self._continuous_batching_enabled_for_pipeline():
                raise RuntimeError(
                    "continuous batching is not enabled for this pipeline"
                )
            self._run_continuous_batching_loop()
            return

        logger.debug(
            f"Rank 0 scheduler listening on tcp://*:{self.server_args.scheduler_port}"
        )

        while self._running:
            # Update queue depth for metrics
            if self._disagg_metrics:
                self._disagg_metrics.update_queue_depth(len(self.waiting_queue))

            # 1: receive requests
            try:
                new_reqs = self.recv_reqs()
                new_reqs = self.process_received_reqs_with_req_based_warmup(new_reqs)
                now = time.monotonic()
                self.waiting_queue.extend(
                    [(identity, req, now) for identity, req in new_reqs]
                )
                # Reset error count on success
                self._consecutive_error_count = 0
            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    f"Error receiving requests in scheduler event loop "
                    f"(attempt {self._consecutive_error_count}/{self._max_consecutive_errors}): {e}",
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    logger.error(
                        f"Maximum consecutive errors ({self._max_consecutive_errors}) reached. "
                        "Terminating scheduler event loop."
                    )
                    raise RuntimeError(
                        f"Scheduler terminated after {self._max_consecutive_errors} "
                        f"consecutive errors. Last error: {e}"
                    ) from e
                continue

            # 2: execute, make sure a reply is always sent
            items = self.get_next_batch_to_run()
            if not items:
                if self.waiting_queue and self._dynamic_batching_enabled():
                    oldest_ts = self.waiting_queue[0][2]
                    elapsed_ms = (time.monotonic() - oldest_ts) * 1000.0
                    remaining_ms = max(0, self._batching_delay_s * 1000.0 - elapsed_ms)
                    if remaining_ms > 0 and self.receiver is not None:
                        self._poller.poll(timeout=remaining_ms)
                    elif remaining_ms > 0:
                        time.sleep(remaining_ms / 1000.0)
                continue

            try:
                handler_result = self._dispatch_items(items)
            except Exception as e:
                logger.error(
                    f"Error executing request in scheduler event loop: {e}",
                    exc_info=True,
                )
                handler_result = OutputBatch(error=str(e))

            if isinstance(handler_result, list):
                output_batches = handler_result
            else:
                output_batches = [handler_result]

            if len(output_batches) != len(items):
                logger.error(
                    "Handler returned %d output(s) for %d request(s). Returning error for unmatched requests.",
                    len(output_batches),
                    len(items),
                )
                output_batches = [
                    OutputBatch(
                        error=(
                            f"Internal scheduler error: expected {len(items)} outputs, "
                            f"got {len(output_batches)}."
                        )
                    )
                    for _ in items
                ]

            # 3. return results
            try:
                for (identity, processed_req), output_batch in zip(
                    items, output_batches, strict=True
                ):
                    is_warmup = is_warmup_req(processed_req)
                    self._log_warmup_result(output_batch, processed_req, is_warmup)

                    should_return_lightweight_warmup_result = (
                        self._should_return_lightweight_warmup_result(processed_req)
                    )
                    if should_return_lightweight_warmup_result:
                        # internal prewarm is a real-path request; reply but drop payloads
                        output_batch.drop_payload_for_warmup()
                        self.return_result(
                            output_batch, identity, should_not_return=False
                        )
                    else:
                        self.return_result(
                            output_batch, identity, should_not_return=is_warmup
                        )
            except zmq.ZMQError as e:
                # Reply failed; log and keep loop alive to accept future requests
                logger.error(f"ZMQ error sending reply: {e}")
                continue

        self._log_batch_metrics_summary()

        if self.receiver is not None:
            self.receiver.close()
        self._cleanup_disagg()
        self.context.destroy(linger=0)

    def _broadcast_task(self, payload: dict[str, Any]) -> None:
        """Broadcast a task to all slave worker processes."""
        method = payload["method"]
        kwargs = {k: v for k, v in payload.items() if k != "method"}
        task = {"method": method, "kwargs": kwargs}
        for pipe in self.task_pipes_to_slaves:
            pipe.send(task)

    def _collect_slave_results(self) -> List[dict[str, Any]]:
        """Collect results from all slave worker processes."""
        results = []
        for pipe in self.result_pipes_from_slaves:
            results.append(pipe.recv())
        return results

    def _handle_release_memory_occupation(self, _reqs: List[Any]) -> OutputBatch:
        logger.info(f"[SLEEP] handle_release_memory_occupation on rank={self.gpu_id}")
        return OutputBatch(output=self.worker.release_memory_occupation())

    def _handle_resume_memory_occupation(self, _reqs: List[Any]) -> OutputBatch:
        logger.info(f"[WAKE] handle_resume_memory_occupation on rank={self.gpu_id}")
        return OutputBatch(output=self.worker.resume_memory_occupation())
