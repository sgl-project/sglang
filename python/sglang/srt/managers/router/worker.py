import asyncio
import logging
import queue
import threading
from typing import List

import rpyc
import uvloop
import zmq
from rpyc.utils.classic import obtain

from sglang.srt.managers.router.model_tp import ModelTpClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import connect_to_rpyc_service
from sglang.utils import get_exception_traceback
from sglang import global_config

logger = logging.getLogger("worker")
CHECKING_INTERVAL = 5

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class RemoteWorkerService(rpyc.Service):
    class exposed_ModelTpClient(ModelTpClient):
        def __init__(
            self, server_args, port_args, model_overide_args, gpu_ids, worker_id
        ):
            server_args = obtain(server_args)
            logging.basicConfig(
                level=getattr(logging, server_args.log_level.upper()),
                format="%(message)s",
            )

            super().__init__(
                server_args, port_args, model_overide_args, gpu_ids, worker_id
            )

            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            def async_loop_wrap(func_name):
                def _func(*args, **kwargs):
                    return self.loop.run_until_complete(
                        getattr(self, func_name)(*args, **kwargs)
                    )

                return _func

            self.exposed_step = async_loop_wrap("step")


class WorkerThread(threading.Thread):
    def __init__(
        self,
        worker_id: int,
        request_queue: queue.Queue,
        detokenizer_port: int,
    ):
        super(WorkerThread, self).__init__()
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.liveness = True
        self.request_dependency_time = global_config.request_dependency_time

        context = zmq.asyncio.Context()
        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(f"tcp://127.0.0.1:{detokenizer_port}")

        self.step = None

    async def loop_for_forward(self):
        while self.liveness:
            requests = []
            while not self.request_queue.empty():
                requests.append(self.request_queue.get())
            try:
                out_pyobjs = await self.step(requests)
            except Exception:
                for r in requests:
                    self.request_queue.put(r)
                logger.error(
                    f"Worker thread {self.worker_id}: "
                    f"failed to get back from Model Server\n"
                    f"{get_exception_traceback()}"
                )
                self.liveness = False

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)

            # async sleep for receiving the subsequent request and avoiding cache miss
            if len(out_pyobjs) != 0:
                has_finished = any([obj.finished for obj in out_pyobjs])
                if has_finished:
                    await asyncio.sleep(self.request_dependency_time)
            await asyncio.sleep(0.0006)

    async def monitoring(self):
        while True:
            await asyncio.sleep(CHECKING_INTERVAL)
            # can plug in monitoring logic here

    def run(self):
        logger.info(f"WorkerThread {self.worker_id} start")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(self.monitoring())
        loop.run_until_complete(self.loop_for_forward())


def start_local_worker(
    server_args: ServerArgs,
    port_args: PortArgs,
    model_overide_args,
    gpu_ids: List[int],
    worker_id: int,
):
    model_tp_client = ModelTpClient(
        server_args,
        port_args.model_port_args[worker_id],
        model_overide_args,
        gpu_ids,
        worker_id,
    )
    worker_thread = WorkerThread(
        worker_id=worker_id,
        request_queue=queue.Queue(),
        detokenizer_port=port_args.detokenizer_port,
    )
    worker_thread.step = model_tp_client.step
    worker_thread.start()
    return worker_thread


def connect_and_start_remote_worker(
    worker_host: str,
    worker_port: int,
    server_args: ServerArgs,
    model_overide_args,
    gpu_ids: List[int],
    worker_id: int,
    detokenizer_port: int,
):
    proxy = connect_to_rpyc_service(worker_port, host=worker_host)
    model_tp_client = proxy.ModelTpClient(
        server_args, None, model_overide_args, gpu_ids, worker_id
    )

    worker_thread = WorkerThread(
        worker_id=worker_id,
        request_queue=queue.Queue(),
        detokenizer_port=detokenizer_port,
    )

    def async_wrap(func_name):
        async def _func(*args, **kwargs):
            f = rpyc.async_(getattr(model_tp_client, func_name))
            return obtain(f(*args, **kwargs).value)

        return _func

    worker_thread.step = async_wrap("step")

    worker_thread.start()
    return worker_thread
