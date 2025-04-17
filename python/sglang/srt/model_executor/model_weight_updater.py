import logging
from abc import ABC
from dataclasses import dataclass
from typing import Tuple, List, Iterable

import torch
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.memory_transfer import AsyncToCudaManager, CombinedManager, SimpleCachingAllocator
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype

logger = logging.getLogger(__name__)


class ModelWeightUpdater:
    def __init__(
            self,
            init_pin_memory: bool,
            load_format: str,
            model_config: ModelConfig,
            model,
            device,
    ):
        self._model_config = model_config
        self._model = model
        self._device = device

        self._all_weights_and_info = _get_all_weights_and_info(load_format=load_format, model_config=model_config,
                                                               model=model,
                                                               pin_memory=init_pin_memory)
        self._transfer_allocator = SimpleCachingAllocator(device="cuda")
        self._memory_transfer_manager = AsyncToCudaManager(
            self._transfer_allocator) if init_pin_memory else CombinedManager.init_pin_memory_and_to_cuda(
            self._transfer_allocator)

        self._state: _State = _StateIdle()

    def start_prepare(self, weight_filter):
        assert isinstance(self._state, _StateIdle)

        self._transfer_allocator.mark_all_unused()
        interesting_weights = [(name, weight) for name, weight, info in self._all_weights_and_info if
                               weight_filter(name, info)]
        self._memory_transfer_manager.enqueue(interesting_weights)

        self._state = _StateAwaitMemoryTransfer()

    def poll_prepare_end(self):
        memory_transfer_outputs = self._memory_transfer_manager.get_outputs()
        assert len(memory_transfer_outputs) in {0, 1}
        if len(memory_transfer_outputs) == 0:
            return False

        self._handle_memory_transfer_output(memory_transfer_outputs[0])
        return True

    def _handle_memory_transfer_output(self, memory_transfer_output):
        assert isinstance(self._state, _StateAwaitMemoryTransfer)
        self._state = _StatePrepared(named_tensors=memory_transfer_output)

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


def _get_all_weights_and_info(load_format: str, model_config: ModelConfig, model, pin_memory: bool):
    load_config = LoadConfig(load_format=load_format)
    loader = get_model_loader(load_config)
    assert isinstance(loader, DefaultModelLoader)
    with set_default_torch_dtype(model_config.dtype):
        all_weights = list(loader._get_weights_iterator(DefaultModelLoader.Source.init_new(model_config, model)))

    if pin_memory:
        all_weights = _named_tensors_pin_memory(all_weights)

    all_weights_and_info = [
        (name, weight, model.get_param_name_info(name))
        for name, weight in all_weights
    ]

    return all_weights_and_info


def _named_tensors_pin_memory(named_tensors: Iterable[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
    return [(name, tensor.pin_memory()) for name, tensor in named_tensors]
