from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, TypeGuard, Union, runtime_checkable

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLOutput,
        DeepEPNormalOutput,
        StandardDispatchOutput,
    )


@dataclass
class DispatchOutputChecker:
    @staticmethod
    def format_is_standard(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[StandardDispatchOutput]:
        return dispatch_output.format == DispatchOutputFormat.standard

    @staticmethod
    def format_is_deepep_normal(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPNormalOutput]:
        return dispatch_output.format == DispatchOutputFormat.deepep_normal

    @staticmethod
    def format_is_deepep_ll(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPLLOutput]:
        return dispatch_output.format == DispatchOutputFormat.deepep_ll

    @staticmethod
    def format_is_deepep(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[Union[DeepEPNormalOutput, DeepEPLLOutput]]:
        return dispatch_output.format in [
            DispatchOutputFormat.deepep_normal,
            DispatchOutputFormat.deepep_ll,
        ]


class DispatchOutputFormat(Enum):
    standard = auto()
    deepep_normal = auto()
    deepep_ll = auto()


@runtime_checkable
class DispatchOutput(Protocol):
    """Protocol for dispatch outputs in different formats."""

    @property
    def format(self) -> DispatchOutputFormat: ...


class BaseDispatcherConfig(ABC):
    """Base class for dispatcher configs."""

    pass


class BaseDispatcher(ABC):
    """Base class for dispatchers."""

    @abstractmethod
    def dispatch(self, *args, **kwargs) -> DispatchOutput:
        pass

    @abstractmethod
    def combine(self, *args, **kwargs) -> torch.Tensor:
        pass
