import pickle
import time
from typing import Any, Generator

import zmq
import zmq.asyncio

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
            payload = await socket.recv()
            request_batch = pickle.loads(payload)
            logger.info("Broker received an offline job from a client.")

            # 2. Forward the request to the main Scheduler via the shared client
            response_batch = await async_scheduler_client.forward(request_batch)

            # 3. Send the Scheduler's reply back to the offline client
            await socket.send(pickle.dumps(response_batch))

        except Exception as e:
            logger.error(f"Error in ZMQ Broker: {e}", exc_info=True)
            # A reply must be sent to prevent the client from hanging
            try:
                await socket.send(pickle.dumps({"status": "error", "message": str(e)}))
            except Exception:
                pass


class SchedulerClient:
    """
    A synchronous, singleton client for communicating with the Scheduler service.
    Designed for use in DiffGenerator, where synchronous usage is preferred
    """

    def __init__(self):
        self.context = None
        self.scheduler_socket = None
        self.server_args = None

    def initialize(self, server_args: ServerArgs):
        if self.context is not None and not self.context.closed:
            logger.warning("SchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.server_args = server_args
        self.context = zmq.Context()
        self.scheduler_socket = self.context.socket(zmq.REQ)

        # Set socket options for the main communication socket
        self.scheduler_socket.setsockopt(zmq.LINGER, 0)

        # 100 minute timeout for generation
        self.scheduler_socket.setsockopt(zmq.RCVTIMEO, 6000000)

        scheduler_endpoint = self.server_args.scheduler_endpoint
        self.scheduler_socket.connect(scheduler_endpoint)
        logger.debug(
            f"SchedulerClient connected to backend scheduler at {scheduler_endpoint}"
        )

    def forward(self, batch: Any) -> Any:
        """Sends a batch or request to the scheduler and waits for the response."""
        try:
            self.scheduler_socket.send_pyobj(batch)
            output_batch = self.scheduler_socket.recv_pyobj()
            return output_batch
        except zmq.error.Again:
            logger.error("Timeout waiting for response from scheduler.")
            raise TimeoutError("Scheduler did not respond in time.")

    def generate_streaming(self, batch: Any) -> Generator[Any, None, Any]:
        """
        Sends a streaming request and yields PartialOutputBatch objects as they arrive.
        The final PartialOutputBatch has is_final=True.
        After the generator is exhausted, the final OutputBatch is available via
        the generator's return value (StopIteration.value).
        """
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            PartialOutputBatch,
        )

        # 100-minute deadline matches the scheduler socket timeout
        _STREAMING_TIMEOUT_S = 6_000

        req = batch[0] if isinstance(batch, list) else batch
        sub_socket = self.context.socket(zmq.SUB)
        sub_socket.setsockopt(zmq.LINGER, 0)
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, req.request_id)
        sub_socket.connect(self.server_args.streaming_endpoint)

        poller = zmq.Poller()
        poller.register(sub_socket, zmq.POLLIN)

        try:
            self.scheduler_socket.send_pyobj(batch)

            deadline = time.monotonic() + _STREAMING_TIMEOUT_S
            while True:
                remaining_ms = (deadline - time.monotonic()) * 1000
                if remaining_ms <= 0:
                    raise TimeoutError(
                        f"Streaming timed out after {_STREAMING_TIMEOUT_S}s "
                        "waiting for is_final frame."
                    )
                # Block for up to 100 ms so the thread yields the GIL and
                # other threads / signals can run between frames.
                ready = dict(poller.poll(timeout=min(remaining_ms, 100)))
                if sub_socket not in ready:
                    continue
                _, payload = sub_socket.recv_multipart()
                partial: PartialOutputBatch = pickle.loads(payload)
                yield partial
                if partial.is_final:
                    break

            return self.scheduler_socket.recv_pyobj()
        finally:
            sub_socket.close()

    def ping(self) -> bool:
        """
        Checks if the scheduler server is alive using a temporary socket.
        """
        if self.context is None or self.context.closed:
            logger.error("Cannot ping: client is not initialized.")
            return False

        ping_socket = self.context.socket(zmq.REQ)
        ping_socket.setsockopt(zmq.LINGER, 0)
        ping_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2-second timeout for pings

        endpoint = self.server_args.scheduler_endpoint

        try:
            ping_socket.connect(endpoint)
            ping_socket.send_pyobj({"method": "ping"})
            ping_socket.recv_pyobj()
            return True
        except zmq.error.Again:
            return False
        finally:
            ping_socket.close()

    def close(self):
        """Closes the socket and terminates the context."""
        if self.scheduler_socket:
            self.scheduler_socket.close()
            self.scheduler_socket = None
        if self.context:
            self.context.term()
            self.context = None


class AsyncSchedulerClient:
    """
    An asynchronous, singleton client for communicating with the Scheduler service.
    Designed for use in asynchronous environments like FastAPI entrypoints.

    To support high concurrency, it creates a new REQ socket for each request
    rather than sharing a single one (which would cause ZMQ state errors).
    """

    def __init__(self):
        self.context = None
        self.server_args = None

    def initialize(self, server_args: ServerArgs):
        if self.context is not None and not self.context.closed:
            logger.warning(
                "AsyncSchedulerClient is already initialized. Re-initializing."
            )
            self.close()

        self.server_args = server_args
        self.context = zmq.asyncio.Context()
        logger.debug("AsyncSchedulerClient initialized with zmq.asyncio.Context")

    async def forward(self, batch: Any) -> Any:
        """Sends a batch or request to the scheduler and waits for the response."""
        if self.context is None:
            raise RuntimeError(
                "AsyncSchedulerClient is not initialized. Call initialize() first."
            )

        # Create a temporary REQ socket for this request to allow concurrency
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        # 100 minute timeout
        socket.setsockopt(zmq.RCVTIMEO, 6000000)

        endpoint = self.server_args.scheduler_endpoint
        socket.connect(endpoint)

        try:
            await socket.send(pickle.dumps(batch))
            payload = await socket.recv()
            return pickle.loads(payload)
        except zmq.error.Again:
            logger.error("Timeout waiting for response from scheduler.")
            raise TimeoutError("Scheduler did not respond in time.")
        finally:
            socket.close()

    async def ping(self) -> bool:
        """
        Checks if the scheduler server is alive using a temporary socket.
        """
        if self.context is None or self.context.closed:
            logger.error("Cannot ping: client is not initialized.")
            return False

        ping_socket = self.context.socket(zmq.REQ)
        ping_socket.setsockopt(zmq.LINGER, 0)
        ping_socket.setsockopt(zmq.RCVTIMEO, 2000)

        endpoint = self.server_args.scheduler_endpoint

        try:
            ping_socket.connect(endpoint)
            await ping_socket.send(pickle.dumps({"method": "ping"}))
            await ping_socket.recv()
            return True
        except zmq.error.Again:
            return False
        finally:
            ping_socket.close()

    def close(self):
        """Closes the socket and terminates the context."""
        if self.context:
            self.context.term()
            self.context = None


# Singleton instances for easy access
async_scheduler_client = AsyncSchedulerClient()
sync_scheduler_client = SchedulerClient()
