"""
A controller that manages multiple data parallel workers.
Each data parallel worker can manage multiple tensor parallel workers.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from typing import Dict

import zmq
import zmq.asyncio

from sglang.global_config import global_config
from sglang.srt.managers.controller.dp_worker import (
    DataParallelWorkerThread,
    start_data_parallel_worker,
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


class Controller:
    def __init__(
        self,
        load_balance_method: str,
        server_args: ServerArgs,
        port_args: PortArgs,
        model_overide_args,
    ):
        self.load_balance_method = LoadBalanceMethod.from_str(load_balance_method)
        self.server_args = server_args
        self.port_args = port_args

        if self.load_balance_method == LoadBalanceMethod.ROUND_ROBIN:
            self.round_robin_counter = 0

        self.dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
        }
        self.dispatching = self.dispatch_lookup[self.load_balance_method]

        # Init communication
        context = zmq.asyncio.Context()
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.router_port}")

        # Init status
        self.recv_reqs = []

        # Start data parallel workers
        self.workers: Dict[int, DataParallelWorkerThread] = {}
        tp_size = server_args.tp_size

        def start_dp_worker(i):
            try:
                gpu_ids = list(range(i * tp_size, (i + 1) * tp_size))
                worker_thread = start_data_parallel_worker(
                    server_args, port_args, model_overide_args, gpu_ids, i
                )
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

    async def round_robin_scheduler(self, input_requests):
        available_workers = list(self.workers.keys())
        for r in input_requests:
            self.put_req_to_worker(available_workers[self.round_robin_counter], r)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                available_workers
            )
        return

    async def shortest_queue_scheduler(self, input_requests):
        for r in input_requests:
            worker = min(
                self.workers, key=lambda w: self.workers[w].request_queue.qsize()
            )
            self.put_req_to_worker(worker, r)
        return

    async def remove_dead_workers(self):
        for i in list(self.workers.keys()):
            worker_thread = self.workers[i]
            if not worker_thread.liveness:
                worker_thread.join()
                # move unsuccessful requests back to the queue
                while not worker_thread.request_queue.empty():
                    self.recv_reqs.append(worker_thread.request_queue.get())
                del self.workers[i]
                logger.info(f"Stale worker {i} removed")

    async def loop_for_forward(self):
        while True:
            await self.remove_dead_workers()

            if self.have_any_live_worker():
                next_step_input = list(self.recv_reqs)
                self.recv_reqs = []
                if next_step_input:
                    await self.dispatching(next_step_input)
            # else:
            #    logger.error("There is no live worker.")

            await asyncio.sleep(global_config.wait_for_new_request_delay)

    async def loop_for_recv_requests(self):
        while True:
            recv_req = await self.recv_from_tokenizer.recv_pyobj()
            if isinstance(recv_req, FlushCacheReq):
                # TODO(lsyin): apply more specific flushCacheReq
                for worker_thread in self.workers.values():
                    worker_thread.request_queue.put(recv_req)
            elif isinstance(recv_req, TokenizedGenerateReqInput):
                self.recv_reqs.append(recv_req)
            elif isinstance(recv_req, AbortReq):
                in_queue = False
                for i, req in enumerate(self.recv_reqs):
                    if req.rid == recv_req.rid:
                        self.recv_reqs[i] = recv_req
                        in_queue = True
                        break
                if not in_queue:
                    # Send abort req to all TP groups
                    for worker in list(self.workers.keys()):
                        self.put_req_to_worker(worker, recv_req)
            else:
                logger.error(f"Invalid object: {recv_req}")


def start_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    model_overide_args=None,
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        controller = Controller(
            server_args.load_balance_method, server_args, port_args, model_overide_args
        )
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(controller.loop_for_recv_requests())
    loop.run_until_complete(controller.loop_for_forward())
