import asyncio
import sys
import zmq
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import queue
import requests
from dataclasses import asdict
import uvloop
import threading
import signal
import setproctitle
import psutil

from sglang.srt.managers.io_struct import (
    PrefillOnlyInput,
    SamplingParams,
)
from sglang.srt.utils import (
    get_zmq_socket,
    kill_process_tree,
    configure_logger,
)
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


class HTTPClientManager:
    def __init__(
        self,
        port_args: PortArgs,
        max_workers=10
    ):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.response_queue = queue.Queue()
        self.asyncio_tasks = set()

        # status
        self.running = False

        # Init inter-process communication
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_out_bound_ipc_name, True
        )

        self._req_dispatcher = TypeBasedDispatcher(
            [
                (PrefillOnlyInput, self._handle_prefill_request),
            ]
        )

    def event_loop(self):
        """The event loop that handles requests"""
        while True:
            logger.info("============== start recv obj ================")
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            logger.info("============== finally recv obj ================")
            print(recv_obj)
            self._req_dispatcher(recv_obj)

    def _handle_prefill_request(self, prefill_input: PrefillOnlyInput):
        try:
            if not prefill_input.prefill_instance_ip_port or len(prefill_input.prefill_instance_ip_port) != 2:
                return {"status": "error", "message": "Invalid prefill instance IP:port"}
            ip, port = prefill_input.prefill_instance_ip_port
            url = f"http://{ip}:{port}/process_prefill_only"
            data = asdict(prefill_input)
            data['sampling_params'] = convert_sampling_params(data.get('sampling_params'))

            logger.info("!!!!!!!!!!!")
            response = requests.post(url, json=data, timeout=0.5)
            logger.info(response)
            logger.info("!!!!!!!!!!!")
            return {"status": "success", "status_code": response.status_code}
        except Exception as e:
            logger.error("!!!!!! error ", str(e))
            return {"status": "error", "message": str(e)}


def convert_sampling_params(obj):
    """Convert SamplingParams and nested objects to dictionaries"""
    if isinstance(obj, SamplingParams):  # Replace with actual class check
        # Use __dict__ to get attributes of regular Python classes
        return {k: convert_sampling_params(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [convert_sampling_params(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_sampling_params(v) for k, v in obj.items()}
    else:
        return obj  # Base types (str, int, etc.) pass through

def run_http_client_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    setproctitle.setproctitle("sglang::http_client")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = HTTPClientManager(port_args)
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"HTTPClientManager hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
