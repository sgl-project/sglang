# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from typing import Any

import zmq

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SyncSchedulerClient:
    """
    A synchronous, singleton client for communicating with the Scheduler service.
    Designed for use in synchronous environments like the DiffGenerator or standalone scripts.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SyncSchedulerClient, cls).__new__(cls)
        return cls._instance

    def initialize(self, server_args: ServerArgs):
        if hasattr(self, "context") and not self.context.closed:
            logger.warning(
                "SyncSchedulerClient is already initialized. Re-initializing."
            )
            self.close()

        self.server_args = server_args
        self.context = zmq.Context()  # Standard synchronous context
        self.scheduler_socket = self.context.socket(zmq.REQ)

        # Set socket options for the main communication socket
        self.scheduler_socket.setsockopt(zmq.LINGER, 0)
        self.scheduler_socket.setsockopt(
            zmq.RCVTIMEO, 6000000
        )  # 10 minute timeout for generation

        scheduler_endpoint = self.server_args.scheduler_endpoint()
        self.scheduler_socket.connect(scheduler_endpoint)
        logger.debug(
            f"SyncSchedulerClient connected to backend scheduler at {scheduler_endpoint}"
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

    def ping(self) -> bool:
        """
        Checks if the scheduler server is alive using a temporary socket.
        This avoids interfering with the state of the main REQ/REP socket.
        """
        if not hasattr(self, "context") or self.context.closed:
            logger.error("Cannot ping: client is not initialized.")
            return False

        ping_socket = self.context.socket(zmq.REQ)
        ping_socket.setsockopt(zmq.LINGER, 0)
        ping_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2-second timeout for pings

        endpoint = self.server_args.scheduler_endpoint()

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
        if hasattr(self, "scheduler_socket"):
            self.scheduler_socket.close()
        if hasattr(self, "context"):
            self.context.term()


# Singleton instance for easy access
sync_scheduler_client = SyncSchedulerClient()
