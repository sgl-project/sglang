from abc import ABC
from dataclasses import dataclass
from typing import Iterator, Tuple

import torch
from sglang.srt.model_executor.memory_transfer import AsyncToCudaManager, CombinedManager


class ModelWeightUpdater:
    def __init__(self, init_pin_memory: bool):
        self._state: _State = _StateIdle()
        self._manager_transfer_manager = AsyncToCudaManager() if init_pin_memory else CombinedManager.init_pin_memory_and_to_cuda()
        self._model_weight_source = _ModelWeightSourcePinnedMemory() if init_pin_memory else _ModelWeightSourceVanilla()

    def start_prepare(self):
        TODO


class _State(ABC):
    pass


@dataclass
class _StateIdle(_State):
    pass


@dataclass
class _StateAwaitMemoryTransfer(_State):
    pass


@dataclass
class _StatePrepared(_State):
    named_tensors: List[Tuple[str, torch.Tensor]]


class _ModelWeightSourceBase(ABC):
    def get_all_weights(self) -> Iterator[Tuple[str, torch.Tensor]]:
        raise NotImplementedError


class _ModelWeightSourceVanilla(_ModelWeightSourceBase):
    def get_all_weights(self) -> Iterator[Tuple[str, torch.Tensor]]:
        yield TODO


class _ModelWeightSourcePinnedMemory(_ModelWeightSourceBase):
    def get_all_weights(self) -> Iterator[Tuple[str, torch.Tensor]]:
        yield TODO
