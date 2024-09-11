"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""A controller that manages a group of tensor parallel workers."""

import logging
import multiprocessing as mp
from typing import List

import torch.multiprocessing as multiprocessing
import zmq

from sglang.srt.managers.speculative_utils import SpecInfoPipline
from sglang.srt.managers.speculative_worker import SpecDraftServer
from sglang.srt.managers.tp_worker import (
    ModelTpServer,
    broadcast_recv_input,
    launch_tp_servers,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, kill_parent_process
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class ControllerSingle:
    """A controller that manages a group of tensor parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        model_overide_args: dict,
        gpu_ids: List[int],
        is_data_parallel_worker: bool,
        dp_worker_id: int,
        mp_queue: multiprocessing.Queue,
        spec_queue: SpecInfoPipline,
        init_flag: multiprocessing.Event = None,
    ):
        # Parse args
        self.tp_size = server_args.tp_size
        self.is_dp_worker = is_data_parallel_worker
        self.dp_worker_id = dp_worker_id
        self.mp_queue = mp_queue

        # Init inter-process communication
        context = zmq.Context(2)

        if not self.is_dp_worker:
            self.recv_from_tokenizer = context.socket(zmq.PULL)
            self.recv_from_tokenizer.bind(
                f"tcp://127.0.0.1:{port_args.controller_port}"
            )

        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(
            f"tcp://127.0.0.1:{port_args.detokenizer_port}"
        )

        # Launch other tp ranks
        tp_size_local = server_args.tp_size // server_args.nnodes
        self.tp_procs = []
        if tp_size_local > 1:
            tp_rank_range = range(1, tp_size_local)
            self.tp_procs = launch_tp_servers(
                gpu_ids,
                tp_rank_range,
                server_args,
                port_args.nccl_ports[dp_worker_id],
                model_overide_args,
            )

        # Launch tp rank 0
        self.tp_server = ModelTpServer(
            gpu_ids[0],
            0,
            server_args,
            port_args.nccl_ports[dp_worker_id],
            model_overide_args,
            spec_queue,
        )
        self.tp_cpu_group = self.tp_server.model_runner.tp_group.cpu_group
        if init_flag is not None:
            init_flag.set()

    def loop_for_forward(self):
        while True:
            if not self.is_dp_worker:
                recv_reqs = self.recv_requests_from_zmq()
            else:
                recv_reqs = self.recv_requests_from_mp_queue()

            if self.tp_size > 1:
                broadcast_recv_input(recv_reqs, 0, self.tp_cpu_group)

            out_pyobjs = self.tp_server.exposed_step(recv_reqs)

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)

    def recv_requests_from_zmq(self):
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)

        return recv_reqs

    def recv_requests_from_mp_queue(self):
        recv_reqs = []
        while not self.mp_queue.empty():
            recv_reqs.append(self.mp_queue.get())
        return recv_reqs


def start_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer: mp.connection.Connection,
    model_overide_args: dict,
    is_data_parallel_worker: bool = False,
    gpu_ids: List[int] = None,
    dp_worker_id: int = None,
    queue: mp.connection.Connection = None,
):
    """Start a controller process."""
    if is_data_parallel_worker:
        logger_prefix = f" DP{dp_worker_id} TP0"
    else:
        logger_prefix = " TP0"
    configure_logger(server_args, prefix=logger_prefix)

    if not is_data_parallel_worker:
        tp_size_local = server_args.tp_size // server_args.nnodes
        gpu_ids = [i for _ in range(server_args.nnodes) for i in range(tp_size_local)]
        dp_worker_id = 0
        queue = None

    try:
        spec_queue = None
        flag = None
        if server_args.speculative_algorithm is not None:
            spec_queue = SpecInfoPipline()
            flag = multiprocessing.Event()

        controller = ControllerSingle(
            server_args,
            port_args,
            model_overide_args,
            gpu_ids,
            is_data_parallel_worker,
            dp_worker_id,
            queue,
            spec_queue,
            flag,
        )

        if server_args.speculative_algorithm is not None:
            flag.wait()
            # draft process should be launch after target process.
            proc = multiprocessing.Process(
                target=start_spec_controller_process,
                args=(
                    server_args,
                    port_args,
                    pipe_writer,
                    model_overide_args,
                    True,
                    gpu_ids,
                    dp_worker_id,
                    queue,
                    spec_queue,
                ),
            )
            proc.start()
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    try:
        controller.loop_for_forward()
    except Exception:
        logger.error("Exception in ControllerSingle:\n" + get_exception_traceback())
    finally:
        kill_parent_process()


class ControllerSingleSpecDraft(ControllerSingle):
    """A controller that manages a group of tensor parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        model_overide_args: dict,
        gpu_ids: List[int],
        is_data_parallel_worker: bool,
        dp_worker_id: int,
        mp_queue: multiprocessing.Queue,
        spec_queue: SpecInfoPipline,
    ):
        # Parse args
        self.tp_size = server_args.tp_size
        self.is_dp_worker = is_data_parallel_worker
        self.dp_worker_id = dp_worker_id

        self.mp_queue = spec_queue.draft_input_queue
        self.spec_server = SpecDraftServer(
            gpu_ids[0],
            0,
            server_args,
            port_args.nccl_ports[dp_worker_id * 2 + 1],
            model_overide_args,
            spec_queue,
        )

    def loop_for_forward(self):
        while True:
            recv_reqs = self.recv_requests_from_mp_queue()
            self.spec_server.exposed_step(recv_reqs)

    def recv_requests_from_mp_queue(self):
        recv_reqs = []
        while not self.mp_queue.empty():
            recv_reqs.append(self.mp_queue.get())
        return recv_reqs


def start_spec_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer: mp.connection.Connection,
    model_overide_args: dict,
    is_data_parallel_worker: bool = False,
    gpu_ids: List[int] = None,
    dp_worker_id: int = None,
    queue: mp.connection.Connection = None,
    spec_queue: SpecInfoPipline = None,
):
    """Start a controller process."""
    if is_data_parallel_worker:
        logger_prefix = f" Spec {dp_worker_id} TP0"
    else:
        logger_prefix = " Spec "
    configure_logger(server_args, prefix=logger_prefix)

    try:
        controller = ControllerSingleSpecDraft(
            server_args,
            port_args,
            model_overide_args,
            gpu_ids,
            is_data_parallel_worker,
            dp_worker_id,
            queue,
            spec_queue,
        )
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise
    finally:
        kill_parent_process()

    pipe_writer.send("draft init ok")

    try:
        controller.loop_for_forward()
    except Exception:
        logger.error("Exception in ControllerSingle:\n" + get_exception_traceback())
    finally:
        kill_parent_process()
