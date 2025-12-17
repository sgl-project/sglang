# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.distributed as dist


class CommunicationBackend(ABC):
    """
    Abstract base class for communication backends.
    This abstraction allows switching between different transport layers
    (e.g., torch.distributed, Mooncake, etc.) without changing the application logic.
    """

    @abstractmethod
    def send_object(
        self, obj: Any, dst: int, group: Optional[dist.ProcessGroup] = None
    ) -> None:
        """
        Send a python object to a destination rank.

        Args:
            obj: The object to send. Must be picklable.
            dst: The destination rank (global rank).
            group: The process group to use for communication.
        """
        pass

    @abstractmethod
    def recv_object(self, src: int, group: Optional[dist.ProcessGroup] = None) -> Any:
        """
        Receive a python object from a source rank.

        Args:
            src: The source rank (global rank).
            group: The process group to use for communication.

        Returns:
            The received object.
        """
        pass

    @abstractmethod
    def send_tensor(
        self, tensor: torch.Tensor, dst: int, group: Optional[dist.ProcessGroup] = None
    ) -> None:
        """
        Send a tensor to a destination rank.
        """
        pass

    @abstractmethod
    def recv_tensor(
        self, tensor: torch.Tensor, src: int, group: Optional[dist.ProcessGroup] = None
    ) -> None:
        """
        Receive a tensor from a source rank into the provided tensor buffer.
        """
        pass

    @abstractmethod
    def broadcast_object(
        self, obj: Any, src: int, group: Optional[dist.ProcessGroup] = None
    ) -> Any:
        """
        Broadcast an object from src to all other ranks in the group.
        """
        pass


class TorchDistributedBackend(CommunicationBackend):
    """
    Implementation of CommunicationBackend using torch.distributed.
    """

    def send_object(
        self, obj: Any, dst: int, group: Optional[dist.ProcessGroup] = None
    ) -> None:
        # torch.distributed.send_object_list sends a list of objects.
        # We wrap our object in a list.
        # Note: send_object_list is a collective in some backends or point-to-point helper in others?
        # Actually, for P2P object sending, torch provides `dist.send` for tensors.
        # `dist.broadcast_object_list` is for broadcast.
        # There isn't a direct `dist.send_object`.
        # We usually serialize it to a ByteTensor and send it.
        # However, newer torch versions might have helpers.
        # Let's use the standard serialization approach for robustness or `rpc` if available.
        # But `rpc` is a different module.
        # Let's implement a simple pickle-based sender using dist.send(tensor).

        # A simpler way is using `dist.isend` if we want async, but let's stick to sync for now.

        # Serialization
        import pickle

        buffer = pickle.dumps(obj)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Send size first
        size_tensor = torch.tensor(
            [tensor.numel()], dtype=torch.long, device=tensor.device
        )
        dist.send(size_tensor, dst, group=group)

        # Send data
        dist.send(tensor, dst, group=group)

    def recv_object(self, src: int, group: Optional[dist.ProcessGroup] = None) -> Any:
        import pickle

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Receive size
        size_tensor = torch.tensor([0], dtype=torch.long, device=device)
        dist.recv(size_tensor, src, group=group)

        size = size_tensor.item()

        # Receive data
        data_tensor = torch.empty(size, dtype=torch.uint8, device=device)
        dist.recv(data_tensor, src, group=group)

        # Deserialize
        buffer = data_tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
        return obj

    def send_tensor(
        self, tensor: torch.Tensor, dst: int, group: Optional[dist.ProcessGroup] = None
    ) -> None:
        dist.send(tensor, dst, group=group)

    def recv_tensor(
        self, tensor: torch.Tensor, src: int, group: Optional[dist.ProcessGroup] = None
    ) -> None:
        dist.recv(tensor, src, group=group)

    def broadcast_object(
        self, obj: Any, src: int, group: Optional[dist.ProcessGroup] = None
    ) -> Any:
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=src, group=group)
        return obj_list[0]


_BACKEND: Optional[CommunicationBackend] = None


def get_backend() -> CommunicationBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = TorchDistributedBackend()
    return _BACKEND


def set_backend(backend: CommunicationBackend):
    global _BACKEND
    _BACKEND = backend
