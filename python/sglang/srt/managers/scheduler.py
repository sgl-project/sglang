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
import multiprocessing
from typing import List

import zmq

from sglang.srt.managers.tp_worker import (
    ModelTpServer,
    broadcast_recv_input,
    launch_tp_servers,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, kill_parent_process
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class Scheduler:

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        tp_rank: int,
    ):
        # Parse args
        self.tp_rank = tp_rank

        # Init inter-process communication
        context = zmq.Context(2)

        if server_args.dist_init_addr:
            host, port = server_args.dist_init_addr.split(":")
            bind_address = f"tcp://{host}:{port + 1}"
        else:
            bind_address = f"tcp://127.0.0.1:{port_args.tokenizer_broadcast_port}"
        self.recv_from_tokenizer = context.socket(zmq.SUB)
        self.recv_from_tokenizer.bind(bind_address)
        self.recv_from_tokenizer.setsockopt_string(
            zmq.SUBSCRIBE, ""
        )  # Subscribe to all messages

        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(
            f"tcp://127.0.0.1:{port_args.detokenizer_port}"
        )

        # Launch a tp server
        self.tp_server = ModelTpServer(
            0,
            0,
            server_args,
            port_args.nccl_ports[0],
        )

    def event_loop(self):
        while True:
            recv_reqs = self.recv_requests_from_zmq()

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


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    tp_rank: int,
    pipe_writer: multiprocessing.connection.Connection,
):
    configure_logger(server_args)

    try:
        scheduler = Scheduler(server_args, port_args, tp_rank)
        pipe_writer.send("ready")
        scheduler.event_loop()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
