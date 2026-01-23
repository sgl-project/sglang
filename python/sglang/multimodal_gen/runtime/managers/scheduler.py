# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import pickle
from collections import deque
from copy import deepcopy
from typing import Any, List

import zmq

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
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
        }

        # FIFO, new reqs are appended
        self.waiting_queue: deque[tuple[bytes, Req]] = deque()

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

    def get_next_batch_to_run(self) -> list[tuple[bytes, Req]] | None:
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
                self.waiting_queue.append((None, req))
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
                processed_req = reqs[0]
                handler = self.request_handlers.get(type(processed_req))
                if handler:
                    output_batch = handler(reqs)
                else:
                    output_batch = OutputBatch(
                        error=f"Unknown request type: {type(processed_req)}"
                    )
            except Exception as e:
                logger.error(
                    f"Error executing request in scheduler event loop: {e}",
                    exc_info=True,
                )
                # Determine appropriate error response format
                output_batch = (
                    OutputBatch(error=str(e))
                    if reqs and isinstance(reqs[0], Req)
                    else OutputBatch(error=str(e))
                )

            # 3. return results
            try:
                # log warmup info
                is_warmup = (
                    processed_req.is_warmup if isinstance(processed_req, Req) else False
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

                # TODO: Support sending back to multiple identities if batched
                self.return_result(output_batch, identities[0], is_warmup=is_warmup)
            except zmq.ZMQError as e:
                # Reply failed; log and keep loop alive to accept future requests
                logger.error(f"ZMQ error sending reply: {e}")
                continue

        logger.info("Scheduler event loop terminated.")
        if self.receiver is not None:
            self.receiver.close()
        self.context.term()

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
