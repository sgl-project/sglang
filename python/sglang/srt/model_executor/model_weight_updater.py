from abc import ABC
from typing import Iterator, Tuple

import torch


class ModelWeightUpdater:
    def __init__(self, init_pin_memory: bool):
        self._manager_transfer_manager = TODO
        self._model_weight_source = _ModelWeightSourcePinnedMemory() if init_pin_memory else _ModelWeightSourceVanilla()


class _ModelWeightSourceBase(ABC):
    def get_all_weights(self) -> Iterator[Tuple[str, torch.Tensor]]:
        raise NotImplementedError


class _ModelWeightSourceVanilla(_ModelWeightSourceBase):
    def get_all_weights(self) -> Iterator[Tuple[str, torch.Tensor]]:
        yield TODO


class _ModelWeightSourcePinnedMemory(_ModelWeightSourceBase):
    def get_all_weights(self) -> Iterator[Tuple[str, torch.Tensor]]:
        yield TODO
