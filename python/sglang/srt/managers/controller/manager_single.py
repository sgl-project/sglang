"""A controller that manages a group of tensor parallel workers."""

import logging
import os

import zmq

from sglang.srt.managers.controller.tp_worker import (
    broadcast_recv_input, launch_tp_servers, ModelTpServer
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import kill_parent_process
from sglang.utils import get_exception_traceback

logger = logging.getLogger("srt.controller")


class ControllerSingle:
    """A controller that manages a group of tensor parallel workers."""

    def __init__(
        self, server_args: ServerArgs, port_args: PortArgs, model_overide_args: dict
    ):
        # Parse args
        self.tp_size = server_args.tp_size

        # Init communication
        context = zmq.Context(2)
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.router_port}")

        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(
            f"tcp://127.0.0.1:{port_args.detokenizer_port}"
        )

        # Init model server
        tp_size_local = server_args.tp_size // server_args.nnodes
        gpu_ids = [i for _ in range(server_args.nnodes) for i in range(tp_size_local)]

        # Launch other tp ranks
        self.tp_procs = []
        if tp_size_local > 1:
            tp_rank_range = range(1, tp_size_local)
            self.tp_procs = launch_tp_servers(
                gpu_ids,
                tp_rank_range,
                server_args,
                port_args.nccl_ports[0],
                model_overide_args,
            )

        # Launch tp rank 0
        self.tp_server = ModelTpServer(
            gpu_ids[0],
            0,
            server_args,
            port_args.nccl_ports[0],
            model_overide_args,
        )
        self.tp_cpu_group = self.tp_server.model_runner.tp_group.cpu_group

    def loop_for_forward(self):
        while True:
            recv_reqs = self.recv_requests()

            if self.tp_size > 1:
                broadcast_recv_input(recv_reqs, 0, self.tp_cpu_group)

            out_pyobjs = self.tp_server.exposed_step(recv_reqs)

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)

    def recv_requests(self):
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)

        return recv_reqs


def start_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    model_overide_args: dict
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        controller = ControllerSingle(server_args, port_args, model_overide_args)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    try:
        controller.loop_for_forward()
    except Exception:
        logger.error("Exception in ControllerSingle:\n" + get_exception_traceback())
    finally:
        for t in controller.tp_procs:
            os.kill(t.pid, 9)
        kill_parent_process()
