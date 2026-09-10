# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

__all__ = ["BaseLinearScheme", "BaseMoEScheme"]


class BaseLinearScheme(ABC):
    """
    Abstract class used to describe the weight creation and forward pass
    of different quantization schemes.
    """

    # Schemes whose parameters only implement the v2 loader API
    # (load_{column,row,merged_column,qkv}_weight) set this so LinearBase
    # routes them through weight_loader_v2 without flipping the loader for
    # every scheme that shares the same LinearMethod class.
    requires_weight_loader_v2: bool = False

    @classmethod
    def get_min_capability(cls) -> int:
        """
        Get minimum device capability.

        Declared rather than abstract: only the config classes that gate on
        device capability call it, and schemes outside those families are
        never asked.
        """
        raise NotImplementedError

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

    # Whether the fused W13 parameter is laid out as [up; gate] rather than
    # the default [gate; up]. Read off the scheme by the MoE method that
    # owns it and forwarded to the FusedMoE weight loader.
    load_up_proj_weight_first: bool = False

    @classmethod
    def get_min_capability(cls) -> int:
        """
        Get minimum device capability.

        Declared rather than abstract: only the config classes that gate on
        device capability call it, and schemes outside those families are
        never asked.
        """
        raise NotImplementedError

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

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """
        Build the MoeRunner this scheme dispatches through.

        Declared rather than abstract: schemes whose family drives the runner
        from the MoE method instead of the scheme do not implement it.
        """
        raise NotImplementedError

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        """
        Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.

        Declared rather than abstract: schemes whose family drives the runner
        from the MoE method instead of the scheme do not implement it.

        :param layer: torch.nn.Module with the registered weights and
            other parameters relevant to the particular scheme.
        :param dispatch_output: output of the token dispatcher

        """
        raise NotImplementedError
