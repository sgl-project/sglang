from typing import List, Tuple

import torch

NamedTensors = List[Tuple[str, torch.Tensor]]


class TensorOperationManagerBase:
    def enqueue(self, named_tensors: NamedTensors):
        raise NotImplementedError

    def get_outputs(self) -> List[NamedTensors]:
        raise NotImplementedError


class CombinedManager(TensorOperationManagerBase):
    def __init__(self, manager_a: TensorOperationManagerBase, manager_b: TensorOperationManagerBase):
        # For simplicity, only support chaining 2 managers, but can be extended to N
        self._manager_a = manager_a
        self._manager_b = manager_b

    @classmethod
    def init_pin_memory_and_to_cuda(cls):
        return cls(manager_a=AsyncPinMemoryManager(), manager_b=AsyncToCudaManager())

    def enqueue(self, named_tensors: NamedTensors):
        self._manager_a.enqueue(named_tensors)

    def get_outputs(self) -> List[NamedTensors]:
        return TODO


class AsyncPinMemoryManager(TensorOperationManagerBase):
    def enqueue(self, named_tensors: NamedTensors):
        TODO

    def get_outputs(self) -> List[NamedTensors]:
        return TODO


class AsyncToCudaManager(TensorOperationManagerBase):
    def enqueue(self, named_tensors: NamedTensors):
        TODO

    def get_outputs(self) -> List[NamedTensors]:
        return TODO
