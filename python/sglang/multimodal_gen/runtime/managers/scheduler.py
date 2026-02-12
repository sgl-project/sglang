# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import asyncio
import dataclasses
import json
import os
import pickle
import time
from collections import deque
from copy import deepcopy
from enum import Enum
from typing import Any, List

import zmq

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    ShutdownReq,
    UnmergeLoraWeightsReq,
    _parse_size,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import GREEN, RESET, init_logger

logger = init_logger(__name__)

MINIMUM_PICTURE_BASE64_FOR_WARMUP = "data:image/jpg;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="


_DYNAMIC_BATCH_SIGNATURE_EXCLUDED_FIELDS = {
    "prompt",
    "request_id",
    "output_file_name",
    "seed",
    "perf_dump_path",
    "suppress_logs",
}


class Scheduler:
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
    ):
        self.server_args = server_args
        self.port_args = port_args

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

        worker = GPUWorker(
            local_rank=gpu_id,
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
            List[Req]: self._handle_generation,
            ListLorasReq: self._handle_list_loras,
            ShutdownReq: self._handle_shutdown,
        }

        # FIFO queue entries: (identity, request, enqueue_ts_s)
        self.waiting_queue: deque[tuple[bytes | None, Any, float]] = deque()
        self._dynamic_batch_max_size = max(1, server_args.dynamic_batch_max_size)
        self._dynamic_batch_delay_s = max(
            0.0, server_args.dynamic_batch_delay_ms / 1000.0
        )

        # whether we've send the necessary warmup reqs
        self.warmed_up = False
        # warmup progress tracking
        self._warmup_total = 0
        self._warmup_processed = 0

        self.prepare_server_warmup_reqs()

        # Maximum consecutive errors before terminating the event loop
        self._max_consecutive_errors = 3
        self._consecutive_error_count = 0

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

    def _handle_generation(self, reqs: List[Req]):
        warmup_reqs = [req for req in reqs if req.is_warmup]
        if warmup_reqs:
            self._warmup_processed += len(warmup_reqs)
            if self._warmup_total > 0:
                logger.info(
                    f"Processing warmup req... ({self._warmup_processed}/{self._warmup_total})"
                )
            else:
                logger.info("Processing warmup req...")

        if len(reqs) == 1:
            return self.worker.execute_forward(reqs)

        merged_req = self._try_merge_generation_reqs(reqs)
        if merged_req is None:
            return self._execute_generation_sequential(reqs)

        batch_size = len(reqs)
        try:
            output_batch = self.worker.execute_forward([merged_req])
            if output_batch.error:
                logger.warning(
                    "Dynamic batch execution returned error. Falling back to sequential execution: %s",
                    output_batch.error,
                )
                return self._execute_generation_sequential(reqs)

            split_outputs = self._split_batched_output(output_batch, reqs)
            if split_outputs is None:
                logger.warning(
                    "Failed to split dynamic batched output cleanly. Falling back to sequential execution."
                )
                return self._execute_generation_sequential(reqs)

            logger.info(
                "Processed dynamic batch of %d request(s) with max_delay=%.2fms",
                batch_size,
                self._dynamic_batch_delay_s * 1000.0,
            )
            return split_outputs
        except Exception as e:
            logger.warning(
                "Dynamic batching failed (%s). Falling back to sequential execution.",
                e,
                exc_info=True,
            )
            return self._execute_generation_sequential(reqs)

    def _execute_generation_sequential(self, reqs: List[Req]) -> List[OutputBatch]:
        return [self.worker.execute_forward([req]) for req in reqs]

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

    def _normalize_for_signature(self, value: Any):
        if isinstance(value, Enum):
            return value.value
        if dataclasses.is_dataclass(value):
            return self._normalize_for_signature(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {
                str(k): self._normalize_for_signature(v)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize_for_signature(v) for v in value]
        return value

    def _build_dynamic_batch_signature(self, req: Req) -> str | None:
        sp = req.sampling_params
        if sp is None:
            return None

        try:
            sp_dict = dataclasses.asdict(sp)
        except Exception:
            return None

        for key in _DYNAMIC_BATCH_SIGNATURE_EXCLUDED_FIELDS:
            sp_dict.pop(key, None)

        if req.extra:
            # Only include user-facing knobs that can alter generation behavior.
            diffusers_kwargs = req.extra.get("diffusers_kwargs")
            if diffusers_kwargs:
                sp_dict["diffusers_kwargs"] = diffusers_kwargs

        normalized = self._normalize_for_signature(sp_dict)
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    def _can_dynamic_batch(self, base_req: Req, candidate_req: Req) -> bool:
        if base_req.is_warmup or candidate_req.is_warmup:
            return False

        if not isinstance(base_req.prompt, str) or not isinstance(
            candidate_req.prompt, str
        ):
            return False

        # Keep image-conditioned requests out of dynamic batching to avoid
        # ambiguity with list-valued image inputs in existing pipelines.
        if base_req.image_path is not None or candidate_req.image_path is not None:
            return False

        base_sig = self._build_dynamic_batch_signature(base_req)
        cand_sig = self._build_dynamic_batch_signature(candidate_req)
        return base_sig is not None and base_sig == cand_sig

    def _try_merge_generation_reqs(self, reqs: List[Req]) -> Req | None:
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
        merged_req.extra["dynamic_batch_request_ids"] = [req.request_id for req in reqs]

        # We split outputs back per original request in scheduler, so avoid
        # saving output files inside a merged worker call.
        merged_req.return_file_paths_only = False
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
            # Nested list/tuple (e.g., trajectory per denoising step). Slice recursively.
            sliced_nested = [
                self._slice_batched_value(v, start, end, total_items) for v in value
            ]
            return sliced_nested if isinstance(value, list) else tuple(sliced_nested)

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
        per_req_counts = [req.num_outputs_per_prompt for req in reqs]
        total_items = sum(per_req_counts)
        output_items = self._count_first_dim(output_batch.output)

        if output_items is not None and output_items != total_items:
            logger.warning(
                "Unexpected batched output size: got %s items, expected %s",
                output_items,
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
                output_file_paths=None,
                timings=deepcopy(output_batch.timings),
                noise_pred=self._slice_batched_value(
                    output_batch.noise_pred, start, end, total_items
                ),
                peak_memory_mb=output_batch.peak_memory_mb,
            )
            if split.timings is not None:
                split.timings.request_id = req.request_id
            outputs.append(split)
            start = end

        return outputs

    def _dynamic_batching_enabled(self) -> bool:
        return self._dynamic_batch_max_size > 1

    def get_next_batch_to_run(self) -> list[tuple[bytes | None, Any]] | None:
        """Pull one request or one dynamic batch from waiting_queue."""
        if not self.waiting_queue:
            return None

        if not self._dynamic_batching_enabled():
            identity, req, _ = self.waiting_queue.popleft()
            return [(identity, req)]

        identity, req, enqueue_time = self.waiting_queue[0]
        if not isinstance(req, Req):
            identity, req, _ = self.waiting_queue.popleft()
            return [(identity, req)]

        batch_len = 0
        hit_incompatible_front = False
        for _identity, candidate_req, _enqueue_time in self.waiting_queue:
            if batch_len >= self._dynamic_batch_max_size:
                break
            if not isinstance(candidate_req, Req) or not self._can_dynamic_batch(
                req, candidate_req
            ):
                hit_incompatible_front = True
                break
            batch_len += 1

        if batch_len <= 0:
            return None

        oldest_wait_s = time.monotonic() - enqueue_time
        should_wait_for_more = (
            batch_len < self._dynamic_batch_max_size
            and oldest_wait_s < self._dynamic_batch_delay_s
            and not hit_incompatible_front
        )
        if should_wait_for_more:
            return None

        batch_items: list[tuple[bytes | None, Any]] = []
        for _ in range(batch_len):
            item_identity, item_req, _ = self.waiting_queue.popleft()
            batch_items.append((item_identity, item_req))
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

            for resolution in self.server_args.warmup_resolutions:
                width, height = _parse_size(resolution)
                task_type = self.server_args.pipeline_config.task_type

                if task_type in (
                    ModelTaskType.I2I,
                    ModelTaskType.TI2I,
                    ModelTaskType.I2V,
                    ModelTaskType.TI2V,
                ):
                    uploads_dir = os.path.join("outputs", "uploads")
                    os.makedirs(uploads_dir, exist_ok=True)
                    input_path = asyncio.run(
                        save_image_to_path(
                            MINIMUM_PICTURE_BASE64_FOR_WARMUP,
                            os.path.join(uploads_dir, "warmup_image.jpg"),
                        )
                    )
                    req = Req(
                        data_type=task_type.data_type(),
                        width=width,
                        height=height,
                        prompt="",
                        negative_prompt="",
                        image_path=[input_path],
                        is_warmup=True,
                    )
                else:
                    req = Req(
                        data_type=task_type.data_type(),
                        width=width,
                        height=height,
                        prompt="",
                        is_warmup=True,
                    )
                self.waiting_queue.append((None, req, time.monotonic()))
            # if server is warmed-up, set this flag to avoid req-based warmup
            self.warmed_up = True

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
        identity, req = recv_reqs[0]
        if isinstance(req, Req):
            warmup_req = deepcopy(req)
            warmup_req.set_as_warmup()
            recv_reqs.insert(0, (identity, warmup_req))
            self._warmup_total = 1
            self._warmup_processed = 0
            self.warmed_up = True
        return recv_reqs

    def recv_reqs(self) -> List[tuple[bytes, Any]]:
        """
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj
        """
        if self.receiver is not None:
            try:
                try:
                    identity, _, payload = self.receiver.recv_multipart(zmq.NOBLOCK)
                    recv_reqs = pickle.loads(payload)
                except zmq.Again:
                    recv_reqs = []
            except zmq.ZMQError:
                # re-raise or handle appropriately to let the outer loop continue
                raise

            if recv_reqs:
                # Ensure recv_reqs is a list
                if not isinstance(recv_reqs, list):
                    recv_reqs = [recv_reqs]

                # Pack with identity for rank 0
                recv_reqs = [(identity, req) for req in recv_reqs]
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

        logger.debug(
            f"Rank 0 scheduler listening on tcp://*:{self.server_args.scheduler_port}"
        )

        while self._running:
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
                if self.waiting_queue and self._dynamic_batch_delay_s > 0:
                    # Avoid busy-spin while waiting for max-batch-delay to elapse.
                    time.sleep(min(0.001, self._dynamic_batch_delay_s))
                continue

            reqs = [item[1] for item in items]

            try:
                processed_req = reqs[0]
                handler = self.request_handlers.get(type(processed_req))
                if handler:
                    handler_result = handler(reqs)
                else:
                    handler_result = OutputBatch(
                        error=f"Unknown request type: {type(processed_req)}"
                    )
            except Exception as e:
                logger.error(
                    f"Error executing request in scheduler event loop: {e}",
                    exc_info=True,
                )
                # Determine appropriate error response format
                handler_result = (
                    OutputBatch(error=str(e))
                    if reqs and isinstance(reqs[0], Req)
                    else OutputBatch(error=str(e))
                )

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
                    # log warmup info
                    is_warmup = (
                        processed_req.is_warmup
                        if isinstance(processed_req, Req)
                        else False
                    )
                    if is_warmup:
                        if output_batch.error is None:
                            if self._warmup_total > 0:
                                logger.info(
                                    f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processed in {GREEN}%.2f{RESET} seconds",
                                    output_batch.timings.total_duration_s,
                                )
                            else:
                                logger.info(
                                    f"Warmup req processed in {GREEN}%.2f{RESET} seconds",
                                    output_batch.timings.total_duration_s,
                                )
                        else:
                            if self._warmup_total > 0:
                                logger.info(
                                    f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processing failed"
                                )
                            else:
                                logger.info(f"Warmup req processing failed")

                    self.return_result(output_batch, identity, is_warmup=is_warmup)
            except zmq.ZMQError as e:
                # Reply failed; log and keep loop alive to accept future requests
                logger.error(f"ZMQ error sending reply: {e}")
                continue

        if self.receiver is not None:
            self.receiver.close()
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
