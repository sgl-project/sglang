from abc import ABC
from dataclasses import dataclass
from typing import Iterator, Tuple, List, Callable, Iterable

import torch
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_executor.memory_transfer import AsyncToCudaManager, CombinedManager
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype


class ModelWeightUpdater:
    def __init__(self, init_pin_memory: bool, weight_filter: Callable[[str], bool]):
        self._weight_filter = weight_filter

        self._state: _State = _StateIdle()
        self._memory_transfer_manager = AsyncToCudaManager() if init_pin_memory else CombinedManager.init_pin_memory_and_to_cuda()
        self._model_weight_source = _ModelWeightSourcePinnedMemory() if init_pin_memory else _ModelWeightSourceVanilla()

    def start_prepare(self):
        assert isinstance(self._state, _StateIdle)

        all_weights_iterator = self._model_weight_source.get_all_weights()
        interesting_weights = [(name, weight) for name, weight in all_weights_iterator if self._weight_filter(name)]
        self._memory_transfer_manager.enqueue(interesting_weights)

        self._state = _StateAwaitMemoryTransfer()

    def event_loop_step(self):
        memory_transfer_outputs = self._memory_transfer_manager.get_outputs()
        assert len(memory_transfer_outputs) in {0, 1}
        if len(memory_transfer_outputs) == 0:
            return False

        memory_transfer_output = memory_transfer_outputs[0]
        self._state = _StatePrepared(named_tensors=memory_transfer_output)
        return True

    def act(self):
        assert isinstance(self._state, _StatePrepared)

        named_tensors = self._state.named_tensors

        # TODO further extract such common operations during weight loading
        with set_default_torch_dtype(TODO):
            DefaultModelLoader.load_weights_and_postprocess(model, named_tensors, target_device)

        self._state = _StateIdle()


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
        TODO_with_set_default_torch_dtype
        raise NotImplementedError


class _ModelWeightSourceVanilla(_ModelWeightSourceBase):
    def get_all_weights(self) -> Iterator[Tuple[str, torch.Tensor]]:
        load_config = LoadConfig(load_format=load_format)
        loader = get_model_loader(load_config)
        assert isinstance(loader, DefaultModelLoader)
        with set_default_torch_dtype(model_config.dtype):
            return loader._get_weights_iterator(DefaultModelLoader.Source.init_new(model_config, model))


class _ModelWeightSourcePinnedMemory(_ModelWeightSourceBase):
    def __init__(self):
        vanilla = _ModelWeightSourceVanilla()
        self._all_weights = _named_tensors_pin_memory(list(vanilla.get_all_weights()))

    def get_all_weights(self) -> Iterator[Tuple[str, torch.Tensor]]:
        return TODO


def _named_tensors_pin_memory(named_tensors: Iterable[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
    return [(name, tensor.pin_memory()) for name, tensor in named_tensors]
