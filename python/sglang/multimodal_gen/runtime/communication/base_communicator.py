"""
Abstract base class for Disaggregation Communication.

This module defines the interface for communication between different components
(e.g., Non-DiT encoder/decoder and DiT denoiser) in a disaggregated architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch.distributed import ProcessGroup


class DisaggCommunicator(ABC):
    """
    Abstract base class for disaggregation communication.

    This class handles:
    1. Topology management (who is Encoder, who is DiT)
    2. Inter-group P2P communication (Send/Recv tensors)
    3. Intra-group collective communication (Broadcast inputs to SP/TP group)
    """

    @abstractmethod
    def initialize_topology(self, server_args: Any) -> None:
        """
        Initialize the distributed topology based on server arguments.

        This should setup:
        - self.non_dit_group: ProcessGroup for Encoder/VAE ranks
        - self.dit_group: ProcessGroup for DiT ranks
        - self.rank_role: "non_dit" or "dit"
        """
        pass

    @abstractmethod
    def get_my_group(self) -> Optional[ProcessGroup]:
        """Return the ProcessGroup this rank belongs to."""
        pass

    @abstractmethod
    def is_dit_rank(self) -> bool:
        """Return True if this rank is part of the DiT group."""
        pass

    @abstractmethod
    def is_non_dit_rank(self) -> bool:
        """Return True if this rank is part of the Non-DiT group."""
        pass

    @abstractmethod
    def send_to_dit(
        self, tensor: torch.Tensor, metadata: Optional[Dict] = None
    ) -> None:
        """
        Send a tensor from Non-DiT (Encoder) to DiT group.
        Usually called by Non-DiT rank 0.
        """
        pass

    @abstractmethod
    def recv_from_non_dit(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """
        Receive a tensor from Non-DiT group at DiT group.
        Usually called by DiT rank 0 (or all DiT ranks if broadcast is built-in).
        """
        pass

    @abstractmethod
    def send_to_non_dit(self, tensor: torch.Tensor) -> None:
        """
        Send a tensor from DiT group to Non-DiT group (VAE).
        Usually called by DiT rank 0.
        """
        pass

    @abstractmethod
    def recv_from_dit(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """
        Receive a tensor from DiT group at Non-DiT group.
        Usually called by Non-DiT rank 0.
        """
        pass

    @abstractmethod
    def broadcast_in_group(
        self, tensor: torch.Tensor, src_rank_in_group: int = 0
    ) -> None:
        """
        Broadcast a tensor within the current group (e.g. from DiT Master to Workers).
        """
        pass
