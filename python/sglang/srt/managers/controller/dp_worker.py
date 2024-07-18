"""A data parallel worker thread."""

import logging
import queue
import threading
from typing import List

import zmq

from sglang.srt.managers.controller.tp_worker import (
    broadcast_recv_input, launch_tp_servers, ModelTpServer
)
from sglang.srt.managers.io_struct import BatchTokenIDOut
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import kill_parent_process
from sglang.utils import get_exception_traceback

logger = logging.getLogger("srt.controller")
CHECKING_INTERVAL = 5


class DataParallelWorkerThread(threading.Thread):
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        model_overide_args: dict,
        gpu_ids: List[int],
        worker_id: int,
    ):
        # Parse args
        super(DataParallelWorkerThread, self).__init__()
        self.worker_id = worker_id
        self.request_queue = queue.Queue()
        self.liveness = True

        # Init communication
        context = zmq.Context()
        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(f"tcp://127.0.0.1:{port_args.detokenizer_port}")

        # Launch other tp ranks
        self.tp_procs = []
        if server_args.tp_size > 1:
            tp_rank_range = range(1, server_args.tp_size)
            self.tp_procs = launch_tp_servers(
                gpu_ids,
                tp_rank_range,
                server_args,
                port_args.model_port_args[0],
                model_overide_args,
            )

        # Launch tp rank 0
        self.tp_server = ModelTpServer(
            gpu_ids[0],
            0,
            server_args,
            port_args.model_port_args[0],
            model_overide_args,
        )
        self.tp_cpu_group = self.tp_server.model_runner.tp_group.cpu_group

    def run(self):
        while self.liveness:
            recv_reqs = []
            while not self.request_queue.empty():
                recv_reqs.append(self.request_queue.get())

            out_pyobjs: List[BatchTokenIDOut] = []
            try:
                out_pyobjs = self.tp_server.exposed_step(recv_reqs)
            except Exception:
                for r in recv_reqs:
                    self.request_queue.put(r)
                logger.error(
                    f"Worker thread {self.worker_id}: "
                    f"failed to get back from Model Server\n"
                    f"{get_exception_traceback()}"
                )
                self.liveness = False
                # Crash the whole server when there are any errors.
                # TODO(lianmin): make this an option.
                kill_parent_process()
                return

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)
