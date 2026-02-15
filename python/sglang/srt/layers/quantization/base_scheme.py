# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe import MoeRunnerConfig

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

__all__ = ["BaseLinearScheme", "BaseMoEScheme"]


class BaseLinearScheme(ABC):
    """
    Abstract class used to describe the weight creation and forward pass
    of different quantization schemes.
    """

    @abstractmethod
    def create_weights(self, *args, **kwargs):
        """
        Weight creation for the particular scheme. Inputs to this function

        """
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        """
        Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.

        :param layer: torch.nn.Module with the registered weights and
            other parameters relevant to the particular scheme.
        :param x: input to the layer
        :param bias: bias parameter

        """
        raise NotImplementedError


class BaseMoEScheme(ABC):
    """
    Abstract class used to describe the weight creation and forward pass
    of different quantization schemes.
    """

    @abstractmethod
    def create_weights(self, *args, **kwargs):
        """
        Weight creation for the particular scheme. Inputs to this function

        """
        raise NotImplementedError

    @abstractmethod
    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        """
        Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.

        :param layer: torch.nn.Module with the registered weights and
            other parameters relevant to the particular scheme.
        :param x: input to the layer
        :param bias: bias parameter

        """
        raise NotImplementedError
