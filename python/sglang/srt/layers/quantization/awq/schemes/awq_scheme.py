# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

__all__ = ["AWQLinearSchemeBase", "AWQMoESchemeBase"]


class AWQLinearSchemeBase(BaseLinearScheme):
    @abstractmethod
    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        raise NotImplementedError


class AWQMoESchemeBase(BaseMoEScheme):
    @abstractmethod
    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        raise NotImplementedError
