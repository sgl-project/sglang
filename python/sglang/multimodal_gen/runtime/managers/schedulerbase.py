# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from abc import ABC
from typing import TypeVar

import zmq

from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import init_logger

logger = init_logger(__name__)

_R = TypeVar("_R")


class SchedulerBase(ABC):
    """
    Abstract base class for all schedulers.
    """

    def __init__(self, server_args: "ServerArgs"):
        """
        Initialize the scheduler.

        Args:
            server_args: The inference arguments
        """
        self.server_args = server_args
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.server_args.scheduler_endpoint())

    @classmethod
    def get_class(cls, server_args: "ServerArgs") -> type["SchedulerBase"]:
        """
        Get the scheduler class based on the server arguments.
        """
        if server_args.distributed_executor_backend == "mp":
            from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler

            # For now, always return the new Scheduler
            return Scheduler
        else:
            raise ValueError(
                f"Unsupported distributed executor backend: {server_args.distributed_executor_backend}"
            )

    # @abstractmethod
    def start(self) -> None:
        """
        Start the scheduler service.
        """
        raise NotImplementedError

    def execute_forward(self, batch: Req, server_args: "ServerArgs") -> OutputBatch:
        """
        Execute a forward pass. This method now sends a request over ZMQ.
        """
        payload = {"method": "execute_forward", "batch": batch}
        self.socket.send_pyobj(payload)
        output_batch = self.socket.recv_pyobj()
        return output_batch

    def set_lora_adapter(
        self, lora_nickname: str, lora_path: str | None = None
    ) -> None:
        """
        Set the LoRA adapter.
        """
        payload = {
            "method": "set_lora_adapter",
            "lora_nickname": lora_nickname,
            "lora_path": lora_path,
        }
        self.socket.send_pyobj(payload)
        self.socket.recv_pyobj()  # Wait for confirmation

    # @abstractmethod
    def unmerge_lora_weights(self) -> None:
        """
        Unmerge the LoRA weights for the workers.
        """
        raise NotImplementedError

    # @abstractmethod
    def merge_lora_weights(self) -> None:
        """
        Merge the LoRA weights for the workers.
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """
        Shutdown the scheduler.
        """
        logger.info("Shutting down scheduler client.")
        payload = {"method": "shutdown"}
        self.socket.send_pyobj(payload)
        self.socket.recv_pyobj()  # Wait for shutdown confirmation
        self.socket.close()
        self.context.term()
