# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

import torch


class DiffusionCache(ABC):
    """Base class for managing diffusion timestep caching.

    Subclasses define specific strategies for deciding when to skip
    computation and how to store/retrieve hidden states.
    """

    @abstractmethod
    def maybe_reset(self, **kwargs) -> None:
        """Resets the internal cache state for a new generation sequence.

        Args:
            **kwargs: Additional parameters that may be helpful.
        """

    @abstractmethod
    def should_skip(self, **kwargs) -> bool:
        """Determines if the current timestep computation can be skipped.

        Args:
            **kwargs: Additional parameters that may be helpful.

        Returns:
            bool: True if the timestep should be skipped, False otherwise.
        """

    @abstractmethod
    def write(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        **kwargs
    ) -> None:
        """Cache the result of a full forward pass to the cache state.

        Args:
            hidden_states: Output of the transformer blocks.
            original_hidden_states: Input from before the transformer blocks.
            **kwargs: Additional parameters that may be helpful.
        """

    @abstractmethod
    def read(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Computes an approximation of the forward pass using cached data. Reads from the cache.

        Args:
            hidden_states: The current input/intermediate hidden states.
            **kwargs: Additional parameters for the retrieval strategy.

        Returns:
            torch.Tensor: The approximated output of the forward pass.
        """

    def calibrate(self, **kwargs) -> None:
        """Performs a calibration step to learn cache thresholds or values.

        Args:
            **kwargs: Additional parameters that may be helpful.
        """
        pass
