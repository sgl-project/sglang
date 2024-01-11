import asyncio
import logging
from typing import List, Tuple
import time

import uvloop
import zmq
import zmq.asyncio
from sglang.srt.managers.router.model_rpc import ModelRpcClient
from sglang.srt.managers.io_struct import BackendConfig, DEFAULT_BACKEND_CONFIG
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class RouterManager:
    def __init__(self, model_client: ModelRpcClient, port_args: PortArgs):
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

        self.model_rpc_sleep_time = DEFAULT_BACKEND_CONFIG.model_rpc_sleep_time  # 0.001
        self.adjust_time_out = DEFAULT_BACKEND_CONFIG.backend_adjust_timeout  # 60.0
        self.last_config_time = time.time()

    async def loop_for_forward(self):
        while True:
            next_step_input = list(self.recv_reqs)
            self.recv_reqs = []
            out_pyobjs = await self.model_client.step(next_step_input)

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)

            if len(out_pyobjs) != 0:
                await asyncio.sleep(0.03)

            # if timeout, reset backend config
            if time.time() - self.last_config_time > self.adjust_time_out:
                print("reset backend config to default")
                self.model_rpc_sleep_time = DEFAULT_BACKEND_CONFIG.model_rpc_sleep_time
                self.adjust_time_out = DEFAULT_BACKEND_CONFIG.backend_adjust_timeout
                self.last_config_time = time.time()

            # await for a while to accept input requests
            await asyncio.sleep(self.model_rpc_sleep_time)

    async def loop_for_recv_requests(self):
        while True:
            recv_req = await self.recv_from_tokenizer.recv_pyobj()
            if isinstance(recv_req, BackendConfig):
                print(f"reset backend config. {recv_req}")
                self.model_rpc_sleep_time = recv_req.model_rpc_sleep_time
                self.adjust_time_out = recv_req.backend_adjust_timeout
                self.last_config_time = time.time()
            else:
                self.recv_reqs.append(recv_req)


def start_router_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        model_client = ModelRpcClient(server_args, port_args)
        router = RouterManager(model_client, port_args)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_recv_requests())
    loop.run_until_complete(router.loop_for_forward())
