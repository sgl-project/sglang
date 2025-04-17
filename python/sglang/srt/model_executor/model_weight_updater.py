from abc import ABC
from dataclasses import dataclass
from typing import Tuple, List, Callable, Iterable

import torch
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.memory_transfer import AsyncToCudaManager, CombinedManager
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype


class ModelWeightUpdater:
    def __init__(
        self,
        init_pin_memory: bool,
        weight_filter: Callable[[str], bool],
        load_format: str,
        model_config: ModelConfig,
        model,
        device,
    ):
        self._weight_filter = weight_filter
        self._model_config = model_config
        self._model = model
        self._device = device

        ModelWeightSourceCls = _ModelWeightSourcePinnedMemory if init_pin_memory else _ModelWeightSourceVanilla
        self._model_weight_source = ModelWeightSourceCls(load_format=load_format, model_config=model_config,
                                                         model=model)
        self._memory_transfer_manager = AsyncToCudaManager() if init_pin_memory else CombinedManager.init_pin_memory_and_to_cuda()

        self._state: _State = _StateIdle()

    def start_prepare(self):
        assert isinstance(self._state, _StateIdle)

        all_weights_iterator = self._model_weight_source.get_all_weights()
        interesting_weights = [(name, weight) for name, weight in all_weights_iterator if self._weight_filter(name)]
        self._memory_transfer_manager.enqueue(interesting_weights)

        self._state = _StateAwaitMemoryTransfer()

    def event_loop_step(self):
        TODO_maybe_rename
        TODO_maybe_change_output

        memory_transfer_outputs = self._memory_transfer_manager.get_outputs()
        assert len(memory_transfer_outputs) in {0, 1}
        if len(memory_transfer_outputs) == 0:
            return False

        assert isinstance(self._state, _StateAwaitMemoryTransfer)
        memory_transfer_output = memory_transfer_outputs[0]
        self._state = _StatePrepared(named_tensors=memory_transfer_output)
        return True

    def act(self):
        assert isinstance(self._state, _StatePrepared)

        target_device = torch.device(self._device)
        named_tensors = self._state.named_tensors

        # TODO further extract such common operations during weight loading
        with set_default_torch_dtype(self._model_config.dtype):
            DefaultModelLoader.load_weights_and_postprocess(self._model, named_tensors, target_device)

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
    def get_all_weights(self) -> Iterable[Tuple[str, torch.Tensor]]:
        raise NotImplementedError


class _ModelWeightSourceVanilla(_ModelWeightSourceBase):
    def __init__(self, load_format: str, model_config: ModelConfig, model):
        self._load_format = load_format
        self._model_config = model_config
        self._model = model

    def get_all_weights(self) -> Iterable[Tuple[str, torch.Tensor]]:
        load_config = LoadConfig(load_format=self._load_format)
        loader = get_model_loader(load_config)
        assert isinstance(loader, DefaultModelLoader)
        with set_default_torch_dtype(self._model_config.dtype):
            yield from loader._get_weights_iterator(DefaultModelLoader.Source.init_new(self._model_config, self._model))


class _ModelWeightSourcePinnedMemory(_ModelWeightSourceBase):
    def __init__(self, *args, **kwargs):
        vanilla = _ModelWeightSourceVanilla(*args, **kwargs)
        self._all_weights = _named_tensors_pin_memory(list(vanilla.get_all_weights()))

    def get_all_weights(self) -> Iterable[Tuple[str, torch.Tensor]]:
        return self._all_weights


def _named_tensors_pin_memory(named_tensors: Iterable[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
    return [(name, tensor.pin_memory()) for name, tensor in named_tensors]
