# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from typing import Any

import zmq

from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
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
            self.receiver, actual_endpoint = get_zmq_socket(
                self.context, zmq.REP, endpoint, True
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

    def return_result(self, output_batch: OutputBatch):
        """
        replies to client, only on rank 0
        """
        if self.receiver is not None:
            self.receiver.send_pyobj(output_batch)

    def recv_reqs(self):
        """
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj
        """
        if self.receiver is not None:
            recv_reqs = self.receiver.recv_pyobj()
            assert isinstance(recv_reqs, list)
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

    # TODO: queueing, cancellation
    def event_loop(self) -> None:
        """
        The main event loop that listens for ZMQ requests.
        Handles abortion
        """

        logger.info(
            f"Rank 0 scheduler listening on tcp://*:{self.server_args.scheduler_port}"
        )

        while self._running:
            reqs = None
            # 1: receive requests
            try:
                reqs = self.recv_reqs()
            except Exception as e:
                logger.error(
                    f"Error receiving requests in scheduler event loop: {e}",
                    exc_info=True,
                )
                continue

            # 2: execute, make sure a reply is always sent
            try:
                output_batch = self.worker.execute_forward(reqs, self.server_args)
            except Exception as e:
                logger.error(
                    f"Error executing forward in scheduler event loop: {e}",
                    exc_info=True,
                )
                output_batch = OutputBatch(error=str(e))

            try:
                self.return_result(output_batch)
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

    def _collect_slave_results(self) -> list[dict[str, Any]]:
        """Collect results from all slave worker processes."""
        results = []
        for pipe in self.result_pipes_from_slaves:
            results.append(pipe.recv())
        return results
