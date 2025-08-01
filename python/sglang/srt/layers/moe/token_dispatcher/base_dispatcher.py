from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Protocol, runtime_checkable

import torch


class MoEA2ABackend(Enum):
    none = "none"
    deepep = "deepep"

    def is_none(self):
        return self == MoEA2ABackend.none

    def is_deepep(self):
        return self == MoEA2ABackend.deepep


class DispatchOutputFormat(Enum):
    standard = auto()
    deepep_normal = auto()
    deepep_ll = auto()

    def is_standard(self) -> bool:
        return self == DispatchOutputFormat.standard

    def is_deepep_normal(self) -> bool:
        return self == DispatchOutputFormat.deepep_normal

    def is_deepep_ll(self) -> bool:
        return self == DispatchOutputFormat.deepep_ll


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
