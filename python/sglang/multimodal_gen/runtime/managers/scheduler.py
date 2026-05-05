# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import asyncio
import dataclasses
import os
import pickle
import tempfile
import time
from collections import deque
from copy import deepcopy
from enum import Enum
from typing import Any, List

import zmq

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
)
from sglang.multimodal_gen.runtime.distributed import get_world_group
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    GetWeightsChecksumReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    GetDisaggStatsReq,
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    ShutdownReq,
    UnmergeLoraWeightsReq,
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
from sglang.multimodal_gen.runtime.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import GREEN, RESET, init_logger
from sglang.multimodal_gen.runtime.utils.trace_wrapper import DiffStage, trace_slice

logger = init_logger(__name__)

MINIMUM_PICTURE_BASE64_FOR_WARMUP = "data:image/jpg;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Placeholder negative_prompt used in synthesized warmup Reqs when
# --enable-cfg-parallel is on. A non-empty, real word (vs "" or " ") so
# every tokenizer backend emits a predictable, non-degenerate token
# sequence — rank 1's uncond branch then produces a valid tensor for
# _combine_cfg_parallel's all-reduce.
DEFAULT_PLACEHOLDER_PROMPT = "warmup"

_MAX_RECV_REQS_PER_POLL = 1024
_BATCH_METRICS_LOG_INTERVAL = 5


class Scheduler(SchedulerDisaggMixin):
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
        self._running = True

        self.request_handlers = {
            SetLoraReq: self._handle_set_lora,
            MergeLoraWeightsReq: self._handle_merge_lora,
            UnmergeLoraWeightsReq: self._handle_unmerge_lora,
            Req: self._handle_generation,
            ListLorasReq: self._handle_list_loras,
            ShutdownReq: self._handle_shutdown,
            GetDisaggStatsReq: self._handle_get_disagg_stats,
            UpdateWeightFromDiskReqInput: self._handle_update_weights_from_disk,
            GetWeightsChecksumReqInput: self._handle_get_weights_checksum,
        }

        # FIFO queue entries: (identity, request, enqueue_ts_s)
        self.waiting_queue: deque[tuple[bytes | None, Any, float]] = deque()
        self._batching_max_size = server_args.batching_max_size
        self._batching_delay_s = server_args.batching_delay_ms / 1000.0
        self._batch_metrics_enabled = server_args.enable_batching_metrics
        self._batch_metrics_window = BatchMetricsWindow()
        self._batch_admission = BatchAdmissionController(server_args, gpu_id=local_rank)
        self._poller = zmq.Poller()
        if self.receiver is not None:
            self._poller.register(self.receiver, zmq.POLLIN)

        # whether we've send the necessary warmup reqs
        self.warmed_up = False
        # warmup progress tracking
        self._warmup_total = 0
        self._warmup_processed = 0

        self.prepare_server_warmup_reqs()

        # Maximum consecutive errors before terminating the event loop
        self._max_consecutive_errors = 3
        self._consecutive_error_count = 0

        self._init_disagg_state(server_args, local_rank)

        if self._batch_metrics_enabled:
            logger.info(
                "Dynamic batch metrics enabled; logging summary every %d dispatches.",
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
            req.lora_nickname, req.lora_path, req.target, req.strength
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

    def _handle_update_weights_from_disk(self, reqs: List[Any]) -> OutputBatch:
        """Handle update_weights_from_disk request for RL workflows."""
        req = reqs[0]
        success, message = self.worker.update_weights_from_disk(
            model_path=req.model_path,
            flush_cache=req.flush_cache,
            target_modules=req.target_modules,
        )
        return OutputBatch(
            output={"success": success, "message": message},
            error=None if success else message,
        )

    def _handle_get_weights_checksum(self, reqs: List[Any]) -> OutputBatch:
        """Handle get_weights_checksum request."""
        req = reqs[0]
        checksums = self.worker.get_weights_checksum(module_names=req.module_names)
        return OutputBatch(output=checksums)

    @staticmethod
    def _normalize_generation_reqs(reqs: list[Any]) -> list[Req]:
        if len(reqs) == 1 and isinstance(reqs[0], list):
            return reqs[0]
        return reqs

    @staticmethod
    def _first_generation_req(req_or_group: Any) -> Req | None:
        """Extract the first req"""
        if isinstance(req_or_group, Req):
            return req_or_group
        if isinstance(req_or_group, list) and req_or_group:
            first_req = req_or_group[0]
            if isinstance(first_req, Req):
                return first_req
        return None

    @classmethod
    def _is_warmup_item(cls, req_or_group: Any) -> bool:
        req = cls._first_generation_req(req_or_group)
        return req.is_warmup if req is not None else False

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

    def _log_warmup_result(self, output_batch: OutputBatch, is_warmup: bool) -> None:
        if not is_warmup:
            return

        if output_batch.error is None:
            total_duration_s = (
                output_batch.metrics.total_duration_s
                if output_batch.metrics is not None
                else 0.0
            )
            if self._warmup_total > 0:
                logger.info(
                    f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processed in {GREEN}%.2f{RESET} seconds",
                    total_duration_s,
                )
            else:
                logger.info(
                    f"Warmup req processed in {GREEN}%.2f{RESET} seconds",
                    total_duration_s,
                )
        else:
            if self._warmup_total > 0:
                logger.info(
                    f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processing failed"
                )
            else:
                logger.info("Warmup req processing failed")

    def _handle_generation(
        self, reqs: list[Any], *, allow_dynamic_batching: bool = True
    ):
        """Dispatch generation requests, merging compatible requests when allowed."""
        reqs = self._normalize_generation_reqs(reqs)
        warmup_reqs = [req for req in reqs if req.is_warmup]
        if warmup_reqs:
            self._warmup_processed += len(warmup_reqs)
            if self._warmup_total > 0:
                logger.info(
                    f"Processing warmup req... ({self._warmup_processed}/{self._warmup_total})"
                )
            else:
                logger.info("Processing warmup req...")

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

    def _freeze_signature_value(self, value: Any):
        """Convert a value into a hashable, order-stable form for signature comparison."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {
                str(k): self._freeze_signature_value(v)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_signature_value(v) for v in value)
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
            (f.name, self._freeze_signature_value(getattr(sp, f.name, None)))
            for f in sp_fields
            if not f.metadata.get("batch_sig_exclude", False)
        ]

    def _diffusers_kwargs_signature_value(self, req: Req) -> Any:
        return self._freeze_signature_value((req.extra or {}).get("diffusers_kwargs"))

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
                        self._freeze_signature_value(diffusers_kwargs),
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
        if not isinstance(base_req.prompt, str) or not isinstance(
            candidate_req.prompt, str
        ):
            return "prompt_type"
        if base_req.image_path is not None or candidate_req.image_path is not None:
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

    def _can_dynamic_batch(self, base_req: Req, candidate_req: Req) -> bool:
        """Return whether `candidate_req` can be merged into a batch with `base_req`."""
        if base_req.is_warmup or candidate_req.is_warmup:
            return False

        if not isinstance(base_req.prompt, str) or not isinstance(
            candidate_req.prompt, str
        ):
            return False

        if base_req.image_path is not None or candidate_req.image_path is not None:
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
    ) -> None:
        if not self._batch_metrics_enabled:
            return

        effective_max_batch_size = max(1, effective_max_batch_size)
        logger.info(
            "Dynamic batch dispatch: size=%d/%d, user_max=%d, queue_wait=%.2fms, stop_reason=%s",
            batch_size,
            effective_max_batch_size,
            self._batching_max_size,
            max(queue_wait_ms, 0.0),
            stop_reason or "unspecified",
        )

        window = self._batch_metrics_window
        window.dispatches += 1
        window.total_requests += batch_size
        window.total_capacity += effective_max_batch_size
        if batch_size > 1:
            window.merged_dispatches += 1
        if self._dynamic_batching_enabled() and batch_size >= effective_max_batch_size:
            window.full_dispatches += 1
        window.wait_times_ms.append(max(queue_wait_ms, 0.0))
        if reject_reasons:
            window.reject_reasons.update(reject_reasons)

        if window.dispatches >= _BATCH_METRICS_LOG_INTERVAL:
            self._log_batch_metrics_summary()

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
        merged_rate = window.merged_dispatches / window.dispatches
        full_rate = window.full_dispatches / window.dispatches
        top_rejects = ", ".join(
            f"{reason}={count}"
            for reason, count in window.reject_reasons.most_common(5)
        )
        if not top_rejects:
            top_rejects = "none"

        logger.info(
            "Dynamic batch stats (last %d dispatches): avg_size=%.2f, merged_rate=%.1f%%, full_rate=%.1f%%, utilization=%.1f%%, wait_avg=%.2fms, wait_p95=%.2fms, top_rejects=%s",
            window.dispatches,
            avg_size,
            merged_rate * 100.0,
            full_rate * 100.0,
            utilization * 100.0,
            avg_wait_ms,
            p95_wait_ms,
            top_rejects,
        )
        self._batch_metrics_window = BatchMetricsWindow()

    def _build_dynamic_batch_error_outputs(
        self,
        reqs: List[Req],
        error_msg: str,
    ) -> List[OutputBatch]:
        return [OutputBatch(error=error_msg) for _ in reqs]

    def return_result(
        self,
        output_batch: OutputBatch,
        identity: bytes | None = None,
        is_warmup: bool = False,
    ):
        """
        replies to client, only on rank 0
        """
        if not is_warmup and self.receiver is not None and identity is not None:
            self.receiver.send_multipart([identity, b"", pickle.dumps(output_batch)])

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
        for req, req_count in zip(reqs, per_req_counts):
            end = start + req_count
            split = OutputBatch(
                output=self._slice_batched_value(
                    output_batch.output, start, end, total_items
                ),
                audio=self._slice_batched_value(
                    output_batch.audio, start, end, total_items
                ),
                audio_sample_rate=output_batch.audio_sample_rate,
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
                metrics=deepcopy(output_batch.metrics),
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

    def prepare_server_warmup_reqs(self):
        if (
            self.server_args.warmup
            and not self.warmed_up
            and self.server_args.warmup_resolutions is not None
        ):
            # insert warmup reqs constructed with each warmup-resolution
            self._warmup_total = len(self.server_args.warmup_resolutions)
            self._warmup_processed = 0
            task_type = self.server_args.pipeline_config.task_type

            requires_warmup_image = task_type.accepts_image_input()
            warmup_input_path = None
            if requires_warmup_image:
                warmup_input_path = self._prepare_shared_warmup_image_path()

            for resolution in self.server_args.warmup_resolutions:
                width, height = _parse_size(resolution)

                # CFG-parallel splits cond/uncond across ranks, so rank 1
                # needs a real uncond pass. Force do_classifier_free_guidance
                # + non-empty negative_prompt when cfg-parallel is on, so the
                # synthesized warmup Req exercises both ranks' denoising paths.
                # When cfg-parallel is off, the Req construction is
                # byte-identical to the pre-fix behavior.
                req_kwargs = dict(
                    data_type=task_type.data_type(),
                    width=width,
                    height=height,
                    prompt="",
                )
                if requires_warmup_image:
                    req_kwargs["negative_prompt"] = ""
                    req_kwargs["image_path"] = [warmup_input_path]
                if self.server_args.enable_cfg_parallel:
                    req_kwargs["negative_prompt"] = DEFAULT_PLACEHOLDER_PROMPT
                    req_kwargs["do_classifier_free_guidance"] = True
                req = Req(**req_kwargs)
                req.set_as_warmup(self.server_args.warmup_steps)
                self.waiting_queue.append((None, req, time.monotonic()))
            # if server is warmed-up, set this flag to avoid req-based warmup
            self.warmed_up = True

    def _prepare_shared_warmup_image_path(self) -> str:
        world_group = get_world_group()
        src_rank = world_group.ranks[0]

        warmup_sync: dict[str, str | None]
        if world_group.rank == src_rank:
            try:
                if self.server_args.input_save_path is not None:
                    uploads_dir = self.server_args.input_save_path
                    os.makedirs(uploads_dir, exist_ok=True)
                else:
                    uploads_dir = tempfile.mkdtemp(prefix="sglang_input_")
                warmup_image_base = os.path.join(uploads_dir, "warmup_image")
                input_path = asyncio.run(
                    save_image_to_path(
                        MINIMUM_PICTURE_BASE64_FOR_WARMUP,
                        warmup_image_base,
                    )
                )
                warmup_sync = {"input_path": input_path, "error": None}
            except Exception as e:
                warmup_sync = {"input_path": None, "error": str(e)}
        else:
            warmup_sync = {}

        # Sync rank 0's warmup-image write result (path or error) to all ranks.
        warmup_sync = broadcast_pyobj(
            warmup_sync,
            world_group.rank,
            world_group.cpu_group,
            src=src_rank,
        )
        if not isinstance(warmup_sync, dict):
            raise RuntimeError("Invalid warmup sync payload received across ranks")

        error = warmup_sync.get("error")
        if error is not None:
            raise RuntimeError(
                f"Warmup image preparation failed on rank {src_rank}: {error}"
            )

        input_path = warmup_sync.get("input_path")
        if not isinstance(input_path, str) or not input_path:
            raise RuntimeError("Warmup image preparation returned empty input path")

        return input_path

    def process_received_reqs_with_req_based_warmup(
        self, recv_reqs: List[tuple[bytes, Any]]
    ) -> List[tuple[bytes, Any]]:
        if (
            self.warmed_up
            or not self.server_args.warmup
            or not recv_reqs
            or self.server_args.warmup_resolutions is not None
        ):
            return recv_reqs

        # handle server req-based warmup by inserting an identical req to the beginning of the waiting queue
        # only the very first req through server's lifetime will be warmed up
        identity, req_or_group = recv_reqs[0]
        req = self._first_generation_req(req_or_group)
        if req is not None:
            warmup_req = req.copy_as_warmup(self.server_args.warmup_steps)
            recv_reqs.insert(0, (identity, warmup_req))
            self._warmup_total = 1
            self._warmup_processed = 0
            self.warmed_up = True
        return recv_reqs

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

    def event_loop(self) -> None:
        """
        The main event loop that listens for ZMQ requests.
        Handles abortion
        """
        # Pool mode: all roles use the pool event loop
        if self._disagg_role != RoleType.MONOLITHIC:
            self._disagg_event_loop()
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
                    is_warmup = self._is_warmup_item(processed_req)
                    self._log_warmup_result(output_batch, is_warmup)

                    self.return_result(output_batch, identity, is_warmup=is_warmup)
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
