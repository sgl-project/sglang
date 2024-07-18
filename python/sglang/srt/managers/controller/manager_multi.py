"""
A controller that manages multiple data parallel workers.
Each data parallel worker can manage multiple tensor parallel workers.
"""

import logging
import queue
from enum import Enum, auto
from typing import Dict

import zmq

from sglang.srt.managers.controller.dp_worker import (
    DataParallelWorkerThread,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    FlushCacheReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.utils import get_exception_traceback

logger = logging.getLogger("srt.controller")


class LoadBalanceMethod(Enum):
    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class ControllerMulti:
    """A controller that manages multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        model_overide_args,
        load_balance_method: str,
    ):
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(load_balance_method)

        # Init communication
        context = zmq.Context()
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.router_port}")

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Start data parallel workers
        self.workers: Dict[int, DataParallelWorkerThread] = {}
        tp_size = server_args.tp_size

        def start_dp_worker(i):
            try:
                gpu_ids = list(range(i * tp_size, (i + 1) * tp_size))
                worker_thread = DataParallelWorkerThread(
                    server_args,
                    port_args,
                    model_overide_args,
                    gpu_ids,
                    i,
                )
                worker_thread.start()
                self.workers[i] = worker_thread
            except Exception:
                logger.error(
                    f"Failed to start local worker {i}\n{get_exception_traceback()}"
                )

        for i in range(server_args.dp_size):
            start_dp_worker(i)

        # Parallel launch is slower, probably due to the disk bandwidth limitations.
        # with ThreadPoolExecutor(server_args.dp_size) as executor:
        #     executor.map(start_dp_worker, range(server_args.dp_size))

    def have_any_live_worker(self):
        return any(worker_thread.liveness for worker_thread in self.workers.values())

    def put_req_to_worker(self, worker_id, req):
        self.workers[worker_id].request_queue.put(req)

    def round_robin_scheduler(self, input_requests):
        available_workers = list(self.workers.keys())
        for r in input_requests:
            self.put_req_to_worker(available_workers[self.round_robin_counter], r)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                available_workers
            )

    def shortest_queue_scheduler(self, input_requests):
        for r in input_requests:
            worker = min(
                self.workers, key=lambda w: self.workers[w].request_queue.qsize()
            )
            self.put_req_to_worker(worker, r)

    def remove_dead_workers(self):
        recv_reqs = []
        for i in list(self.workers.keys()):
            worker_thread = self.workers[i]
            if not worker_thread.liveness:
                worker_thread.join()
                # move unsuccessful requests back to the queue
                while not worker_thread.request_queue.empty():
                    recv_reqs.append(worker_thread.request_queue.get())
                del self.workers[i]
                logger.info(f"Stale worker {i} removed")
        return recv_reqs

    def loop_for_forward(self):
        while True:
            left_reqs = self.remove_dead_workers()

            if self.have_any_live_worker():
                recv_reqs = self.recv_requests() + left_reqs

                if recv_reqs:
                    self.dispatching(recv_reqs)
            # else:
            #    logger.error("There is no live worker.")

    def recv_requests(self):
        recv_reqs = []

        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break

            if isinstance(recv_req, FlushCacheReq):
                # TODO(lsyin): apply more specific flushCacheReq
                for worker in self.workers.keys():
                    self.put_req_to_worker(worker, recv_req)
            elif isinstance(recv_req, AbortReq):
                in_queue = False
                for i, req in enumerate(recv_reqs):
                    if req.rid == recv_req.rid:
                        recv_reqs[i] = recv_req
                        in_queue = True
                        break
                if not in_queue:
                    # Send abort req to all TP groups
                    for worker in self.workers.keys():
                        self.put_req_to_worker(worker, recv_req)
            elif isinstance(recv_req, TokenizedGenerateReqInput):
                recv_reqs.append(recv_req)
            else:
                logger.error(f"Invalid object: {recv_req}")

        return recv_reqs


def start_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    model_overide_args: dict,
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        controller = ControllerMulti(server_args, port_args, model_overide_args)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    try:
        controller.loop_for_forward()
    except Exception:
        logger.error("Exception in ControllerMulti:\n" + get_exception_traceback())
    finally:
        pass
