# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import pickle
from collections import deque
from typing import Any, List

import zmq

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    MergeLoraWeightsReq,
    SetLoraReq,
    UnmergeLoraWeightsReq,
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
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
        endpoint = server_args.scheduler_endpoint()
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
        }

        # FIFO, new reqs are appended
        self.waiting_queue: deque[tuple[bytes, Req]] = deque()

    def _handle_set_lora(self, reqs: List[Any]):
        # TODO: return set status
        req = reqs[0]
        self.worker.set_lora(req.lora_nickname, req.lora_path, req.target, req.strength)
        return {"status": "ok"}

    def _handle_merge_lora(self, reqs: List[Any]):
        req = reqs[0]
        self.worker.merge_lora_weights(req.target, req.strength)
        return {"status": "ok"}

    def _handle_unmerge_lora(self, reqs: List[Any]):
        req = reqs[0]
        self.worker.unmerge_lora_weights(req.target)
        return {"status": "ok"}

    def _handle_generation(self, reqs: List[Req]):
        return self.worker.execute_forward(reqs)

    def return_result(self, output_batch: OutputBatch, identity: bytes | None = None):
        """
        replies to client, only on rank 0
        """
        if self.receiver is not None and identity is not None:
            self.receiver.send_multipart([identity, b"", pickle.dumps(output_batch)])

    def get_next_batch_to_run(self) -> list[tuple[bytes, Req]] | None:
        """pull a req from waiting_queue"""
        if not self.waiting_queue:
            return None

        # pop the first (earliest)
        item = self.waiting_queue.popleft()

        return [item]

    def recv_reqs(self) -> List[tuple[bytes, Any]]:
        """
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj
        """
        if self.receiver is not None:
            try:
                identity, _, payload = self.receiver.recv_multipart()
                recv_reqs = pickle.loads(payload)
            except zmq.ZMQError:
                # re-raise or handle appropriately to let the outer loop continue
                raise

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
                # after processing input reqs
                self.waiting_queue.extend(new_reqs)
            except Exception as e:
                logger.error(
                    f"Error receiving requests in scheduler event loop: {e}",
                    exc_info=True,
                )
                continue

            # 2: execute, make sure a reply is always sent
            while self.waiting_queue:
                items = self.get_next_batch_to_run()
                if not items:
                    break

                identities = [item[0] for item in items]
                reqs = [item[1] for item in items]

                try:
                    first_req = reqs[0]
                    handler = self.request_handlers.get(type(first_req))
                    if handler:
                        output_batch = handler(reqs)
                    else:
                        output_batch = {
                            "status": "error",
                            "message": f"Unknown request type: {type(first_req)}",
                        }
                except Exception as e:
                    logger.error(
                        f"Error executing request in scheduler event loop: {e}",
                        exc_info=True,
                    )
                    # Determine appropriate error response format
                    output_batch = (
                        OutputBatch(error=str(e))
                        if reqs and isinstance(reqs[0], Req)
                        else {"status": "error", "message": str(e)}
                    )

                try:
                    # TODO: Support sending back to multiple identities if batched
                    self.return_result(output_batch, identities[0])
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

    def _execute_on_rank0(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute task locally on the rank 0 worker."""
        method = payload["method"]
        kwargs = {k: v for k, v in payload.items() if k != "method"}
        handler = getattr(self.worker, method, None)
        if handler:
            result = handler(**kwargs)
            return {"status": "ok", "result": result}
        return {"status": "error", "error": f"Unknown method: {method}"}

    def _collect_slave_results(self) -> List[dict[str, Any]]:
        """Collect results from all slave worker processes."""
        results = []
        for pipe in self.result_pipes_from_slaves:
            results.append(pipe.recv())
        return results
