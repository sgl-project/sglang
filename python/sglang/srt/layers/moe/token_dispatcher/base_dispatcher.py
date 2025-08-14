from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, TypeGuard, Union, runtime_checkable

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        AscendDeepEPLLOutput,
        DeepEPLLOutput,
        DeepEPNormalOutput,
        StandardDispatchOutput,
    )


class DispatchOutputChecker:

    @staticmethod
    def format_is_standard(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[StandardDispatchOutput]:
        return dispatch_output.format.is_standard()

    @staticmethod
    def format_is_deepep_normal(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPNormalOutput]:
        return dispatch_output.format.is_deepep_normal()

    @staticmethod
    def format_is_deepep_ll(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPLLOutput]:
        return dispatch_output.format.is_deepep_ll()

    @staticmethod
    def format_is_deepep(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[Union[DeepEPNormalOutput, DeepEPLLOutput]]:
        return dispatch_output.format.is_deepep()

    @staticmethod
    def format_is_ascent_ll(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[AscendDeepEPLLOutput]:
        return dispatch_output.format.is_ascent_ll()


class DispatchOutputFormat(Enum):

    STANDARD = auto()
    DEEPEP_NORMAL = auto()
    DEEPEP_LL = auto()
    ASCENT_LL = auto()

    def is_standard(self) -> bool:
        return self == DispatchOutputFormat.STANDARD

    def is_deepep_normal(self) -> bool:
        return self == DispatchOutputFormat.DEEPEP_NORMAL

    def is_deepep_ll(self) -> bool:
        return self == DispatchOutputFormat.DEEPEP_LL

    def is_deepep(self) -> bool:
        return self in [
            DispatchOutputFormat.DEEPEP_NORMAL,
            DispatchOutputFormat.DEEPEP_LL,
        ]

    def is_ascent_ll(self) -> bool:
        return self == DispatchOutputFormat.ASCENT_LL


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
