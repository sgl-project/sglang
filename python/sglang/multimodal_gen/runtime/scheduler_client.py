# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio

import zmq
import zmq.asyncio

from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Using a singleton pattern to hold the ZMQ context and the socket connected to the scheduler
class SchedulerClient:
    """
    A gateway for Scheduler, forwarding the ForwardBatch from http endpoints (or somewhere else) to background scheduler, with TCP socket
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SchedulerClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        # Ensure the initialization runs only once for the singleton instance
        if getattr(self, "_init_done", False):
            return
        # Queue + worker to strictly serialize ZeroMQ REQ/REP interactions
        self._request_queue = asyncio.Queue()
        self._worker_task = None
        self._closing = False
        self._init_done = True

    def initialize(self, server_args: ServerArgs):
        self.server_args = server_args
        self.context = zmq.asyncio.Context()
        # This is the REQ socket used to connect to the backend Scheduler
        self.scheduler_socket = self.context.socket(zmq.REQ)
        scheduler_endpoint = server_args.scheduler_endpoint()
        self.scheduler_socket.connect(scheduler_endpoint)
        logger.info(
            f"Scheduler client connected to backend scheduler at {scheduler_endpoint}"
        )
        # Worker will be lazily started on the first forward call to ensure a running loop exists

    async def forward(self, batch: Req) -> Req:
        """Enqueue a request to the backend Scheduler and await the reply."""
        if self._closing:
            raise RuntimeError(
                "SchedulerClient is closing; cannot forward new requests"
            )

        await self._ensure_worker_started()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._request_queue.put((batch, future))
        return await future

    async def _ensure_worker_started(self):
        # Start the worker only once and only when an event loop is running
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self):
        while True:
            try:
                item = await self._request_queue.get()
                try:
                    batch, future = item
                except Exception:
                    # Malformed queue item; skip
                    self._request_queue.task_done()
                    continue

                try:
                    await self.scheduler_socket.send_pyobj(batch)
                    response = await self.scheduler_socket.recv_pyobj()
                    if not future.done():
                        future.set_result(response)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
                finally:
                    self._request_queue.task_done()
            except asyncio.CancelledError:
                # Drain remaining items with cancellation error to avoid hanging waiters
                while True:
                    try:
                        batch, future = self._request_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    try:
                        if not future.done():
                            future.set_exception(asyncio.CancelledError())
                    finally:
                        self._request_queue.task_done()
                raise

    def close(self):
        self._closing = True
        # Cancel worker if running
        if self._worker_task is not None:
            self._worker_task.cancel()
        try:
            self.scheduler_socket.close()
        finally:
            try:
                self.context.term()
            except Exception:
                pass


# Singleton instance
scheduler_client = SchedulerClient()


async def run_zeromq_broker(server_args: ServerArgs):
    """
    This function runs as a background task in the FastAPI process.
    It listens for TCP requests from offline clients (e.g., DiffGenerator).
    """
    ctx = zmq.asyncio.Context()
    # This is the REP socket that listens for requests from DiffGenerator
    socket = ctx.socket(zmq.REP)
    broker_endpoint = f"tcp://*:{server_args.broker_port}"
    socket.bind(broker_endpoint)
    logger.info(f"ZMQ Broker is listening for offline jobs on {broker_endpoint}")

    while True:
        try:
            # 1. Receive a request from an offline client
            request_batch = await socket.recv_pyobj()
            logger.info("Broker received an offline job from a client.")

            # 2. Forward the request to the main Scheduler via the shared client
            response_batch = await scheduler_client.forward(request_batch)

            # 3. Send the Scheduler's reply back to the offline client
            await socket.send_pyobj(response_batch)

        except Exception as e:
            logger.error(f"Error in ZMQ Broker: {e}", exc_info=True)
            # A reply must be sent to prevent the client from hanging
            await socket.send_pyobj({"status": "error", "message": str(e)})
