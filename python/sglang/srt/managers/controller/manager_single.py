"""A controller that manages a group of tensor parallel workers."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import uvloop
import zmq
import zmq.asyncio

from sglang.global_config import global_config
from sglang.srt.managers.controller.tp_worker import ModelTpServer
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import kill_parent_process
from sglang.utils import get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger("srt.controller")


class ControllerSingle:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs, model_overide_args: dict):
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
        self.model_server = ModelTpServer(
            gpu_ids[0],
            0,
            server_args,
            port_args.model_port_args[0],
            model_overide_args,
        )

        # Init status
        self.recv_reqs = []

    def loop_for_forward(self):
        while True:
            recv_reqs = self.recv_requests()
            out_pyobjs = self.model_server.exposed_step(recv_reqs)

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)

    def recv_requests(self):
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                recv_reqs.append(recv_req)
            except zmq.ZMQError:
                break
        return recv_reqs

def start_controller_process(
    server_args: ServerArgs, port_args: PortArgs, pipe_writer, model_overide_args: dict
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
        kill_parent_process()
