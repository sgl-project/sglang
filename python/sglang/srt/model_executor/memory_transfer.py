from typing import List, Tuple

import torch

NamedTensors = List[Tuple[str, torch.Tensor]]


class TensorOperationManagerBase:
    def enqueue(self, named_tensors: NamedTensors):
        raise NotImplementedError

    def get_outputs(self) -> List[NamedTensors]:
        raise NotImplementedError


class CombinedManager(TensorOperationManagerBase):
    @classmethod
    def init_pin_memory_and_to_cuda(cls):
        return cls(manager_a=AsyncPinMemoryManager(), manager_b=AsyncToCudaManager())

    def enqueue(self, named_tensors: NamedTensors):
        TODO

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
