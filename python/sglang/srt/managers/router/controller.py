import asyncio
import logging
from enum import Enum, auto
from typing import Dict

import zmq
import zmq.asyncio

from sglang.srt.managers.io_struct import (
    FlushCacheReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.router.worker import (
    WorkerThread,
    connect_and_start_remote_worker,
    start_local_worker,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.utils import get_exception_traceback

logger = logging.getLogger("srt.controller")


class DispatchMethod(Enum):
    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid dispatch method: {method}") from exc


class Controller:
    def __init__(
        self,
        dispatch_method: str,
        server_args: ServerArgs,
        port_args: PortArgs,
        model_overide_args,
    ):
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)
        self.server_args = server_args
        self.port_args = port_args

        if self.dispatch_method == DispatchMethod.ROUND_ROBIN:
            self.round_robin_counter = 0

        self.dispatch_lookup = {
            DispatchMethod.ROUND_ROBIN: self.round_robin_scheduler,
            DispatchMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
        }
        self.dispatching = self.dispatch_lookup[self.dispatch_method]

        # Init communication
        context = zmq.asyncio.Context()
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.router_port}")

        # Init status
        self.recv_reqs = []

        # start local workers
        self.workers: Dict[int, WorkerThread] = {}
        tp_size = server_args.tp_size
        for i in range(server_args.dp_size):
            try:
                gpu_ids = list(range(i * tp_size, (i + 1) * tp_size))
                worker_thread = start_local_worker(
                    server_args, port_args, model_overide_args, gpu_ids, i
                )
                self.workers[i] = worker_thread
            except Exception:
                logger.error(
                    f"Failed to start local worker {i}\n{get_exception_traceback()}"
                )

    def set_dispatch_method(self, dispatch_method: str):
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)
        self.dispatching = self.dispatch_lookup[self.dispatch_method]

    def have_any_live_worker(self):
        return any(worker_thread.liveness for worker_thread in self.workers.values())

    def alloc_worker_id(self):
        if len(self.workers) == 0:
            return 0
        return max(self.workers.keys()) + 1

    def register_worker(
        self,
        worker_host,
        worker_port,
        server_args: ServerArgs,
        model_overide_args,
        gpu_ids,
    ):
        # resolve default gpu ids for remote worker
        if gpu_ids is None:
            gpu_ids = list(range(0, server_args.tp_size))

        if len(gpu_ids) != server_args.tp_size:
            logger.error(
                f"Invalid gpu nums for {worker_host}:{worker_port}, "
                f"expect {server_args.tp_size}gpus but got {len(gpu_ids)}"
            )
            return

        try:
            new_worker_id = self.alloc_worker_id()
            worker_thread = connect_and_start_remote_worker(
                worker_host,
                worker_port,
                server_args,
                model_overide_args,
                gpu_ids,
                new_worker_id,
                self.port_args.detokenizer_port,
            )
            self.workers[new_worker_id] = worker_thread
        except Exception:
            logger.error(
                f"Failed to connect to worker {worker_host}:{worker_port}\n"
                f"{get_exception_traceback()}"
            )

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

    async def get_worker_status(self, worker_id: int):
        try:
            return await asyncio.wait_for(
                self.workers[worker_id].model_client.get_status(), timeout=0.1
            )
        except asyncio.TimeoutError:
            return None

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

            await asyncio.sleep(0.001)

    async def loop_for_recv_requests(self):
        while True:
            recv_req = await self.recv_from_tokenizer.recv_pyobj()
            if isinstance(recv_req, FlushCacheReq):
                # TODO(lsyin): apply more specific flushCacheReq
                for worker_thread in self.workers.values():
                    worker_thread.request_queue.put(recv_req)
            elif isinstance(recv_req, TokenizedGenerateReqInput):
                self.recv_reqs.append(recv_req)
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
            server_args.dispatch_method, server_args, port_args, model_overide_args
        )
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(controller.loop_for_recv_requests())
    loop.run_until_complete(controller.loop_for_forward())
