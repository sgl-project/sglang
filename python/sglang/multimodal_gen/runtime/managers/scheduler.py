# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import pickle
import tempfile
from collections import deque
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
from sglang.multimodal_gen.runtime.utils.trace_wrapper import DiffStage, trace_slice

logger = init_logger(__name__)

MINIMUM_PICTURE_BASE64_FOR_WARMUP = "data:image/jpg;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Placeholder negative_prompt used in synthesized warmup Reqs when
# --enable-cfg-parallel is on. A non-empty, real word (vs "" or " ") so
# every tokenizer backend emits a predictable, non-degenerate token
# sequence — rank 1's uncond branch then produces a valid tensor for
# _combine_cfg_parallel's all-reduce.
DEFAULT_PLACEHOLDER_PROMPT = "warmup"


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

        # FIFO, new reqs are appended
        self.waiting_queue: deque[tuple[bytes, Any]] = deque()

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

    def _dispatch_request(self, reqs: list[Any]) -> OutputBatch:
        """dispatch req to its registered handler"""
        req_or_group = reqs[0]
        if isinstance(req_or_group, list):
            return self._handle_generation(reqs)

        handler = self.request_handlers.get(type(req_or_group))
        if handler is None:
            return OutputBatch(error=f"Unknown request type: {type(req_or_group)}")
        return handler(reqs)

    def _log_warmup_result(self, output_batch: OutputBatch, is_warmup: bool) -> None:
        if not is_warmup:
            return

        if output_batch.error is None:
            if self._warmup_total > 0:
                logger.info(
                    f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processed in {GREEN}%.2f{RESET} seconds",
                    output_batch.metrics.total_duration_s,
                )
            else:
                logger.info(
                    f"Warmup req processed in {GREEN}%.2f{RESET} seconds",
                    output_batch.metrics.total_duration_s,
                )
        else:
            if self._warmup_total > 0:
                logger.info(
                    f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processing failed"
                )
            else:
                logger.info("Warmup req processing failed")

    def _handle_generation(self, reqs: list[Any]):
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

        # Diffusion dispatches one generation request at a time, so reqs[0]
        # always carries the trace context for the entire batch.
        req = reqs[0]
        req.trace_ctx.rebuild_thread_context()
        with trace_slice(
            req.trace_ctx,
            DiffStage.SCHEDULER_DISPATCH,
            thread_finish_flag=True,
        ):
            return self.worker.execute_forward(reqs)

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

    def get_next_batch_to_run(self) -> list[tuple[bytes, Any]] | None:
        """pull a req from waiting_queue"""
        if not self.waiting_queue:
            return None

        # pop the first (earliest)
        item = self.waiting_queue.popleft()

        return [item]

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
                self.waiting_queue.append((None, req))
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

    def recv_reqs(self) -> List[tuple[bytes, Any]]:
        """
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj
        """
        if self.receiver is not None:
            try:
                try:
                    # Accept valid REQ envelopes only, ignore malformed/probe frames.
                    parts = self.receiver.recv_multipart(zmq.NOBLOCK)
                    identity, payload = parts[0], parts[-1]

                    # Ignore malformed probes or non-pickle data
                    recv_reqs = pickle.loads(payload) if len(parts) > 2 else []
                except (zmq.Again, pickle.UnpicklingError, IndexError, EOFError):
                    recv_reqs = []
            except zmq.ZMQError:
                # re-raise or handle appropriately to let the outer loop continue
                raise

            if recv_reqs:
                if isinstance(recv_reqs, list) and all(
                    isinstance(req, Req) for req in recv_reqs
                ):
                    recv_reqs = [(identity, recv_reqs)]
                else:
                    if not isinstance(recv_reqs, list):
                        recv_reqs = [recv_reqs]
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
                self.waiting_queue.extend(new_reqs)
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
                continue

            identities = [item[0] for item in items]
            reqs = [item[1] for item in items]

            try:
                req_or_group = reqs[0]
                is_warmup = self._is_warmup_item(req_or_group)
                output_batch = self._dispatch_request(reqs)
            except Exception as e:
                logger.error(
                    f"Error executing request in scheduler event loop: {e}",
                    exc_info=True,
                )
                output_batch = OutputBatch(error=str(e))

            # 3. return results
            try:
                self._log_warmup_result(output_batch, is_warmup)

                # TODO: Support sending back to multiple identities if batched
                self.return_result(output_batch, identities[0], is_warmup=is_warmup)
            except zmq.ZMQError as e:
                # Reply failed; log and keep loop alive to accept future requests
                logger.error(f"ZMQ error sending reply: {e}")
                continue

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
