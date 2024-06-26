"""A controller that manages a group of tensor parallel workers."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import uvloop
import zmq
import zmq.asyncio

from sglang.global_config import global_config
from sglang.srt.managers.controller.tp_worker import ModelTpClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import kill_parent_process
from sglang.utils import get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger("srt.controller")


class ControllerSingle:
    def __init__(self, model_client: ModelTpClient, port_args: PortArgs):
        # Init communication
        context = zmq.asyncio.Context(2)
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.router_port}")

        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(
            f"tcp://127.0.0.1:{port_args.detokenizer_port}"
        )

        # Init status
        self.model_client = model_client
        self.recv_reqs = []

        # Init some configs
        self.request_dependency_delay = global_config.request_dependency_delay

    async def loop_for_forward(self):
        while True:
            next_step_input = list(self.recv_reqs)
            self.recv_reqs = []
            out_pyobjs = await self.model_client.step(next_step_input)

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)

            # async sleep for receiving the subsequent request and avoiding cache miss
            slept = False
            if len(out_pyobjs) != 0:
                has_finished = any(
                    [obj.finished_reason is not None for obj in out_pyobjs]
                )
                if has_finished:
                    if self.request_dependency_delay > 0:
                        slept = True
                        await asyncio.sleep(self.request_dependency_delay)

            if not slept:
                await asyncio.sleep(global_config.wait_for_new_request_delay)

    async def loop_for_recv_requests(self):
        while True:
            recv_req = await self.recv_from_tokenizer.recv_pyobj()
            self.recv_reqs.append(recv_req)


def start_controller_process(
    server_args: ServerArgs, port_args: PortArgs, pipe_writer, model_overide_args
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        tp_size_local = server_args.tp_size // server_args.nnodes
        model_client = ModelTpClient(
            [i for _ in range(server_args.nnodes) for i in range(tp_size_local)],
            server_args,
            port_args.model_port_args[0],
            model_overide_args,
        )
        controller = ControllerSingle(model_client, port_args)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    loop = asyncio.new_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=256))
    asyncio.set_event_loop(loop)
    loop.create_task(controller.loop_for_recv_requests())
    try:
        loop.run_until_complete(controller.loop_for_forward())
    except Exception:
        logger.error("Exception in ControllerSingle:\n" + get_exception_traceback())
    finally:
        kill_parent_process()