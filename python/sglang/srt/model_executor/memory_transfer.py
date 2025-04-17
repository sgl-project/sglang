from typing import List, Tuple

import torch

NamedTensors = List[Tuple[str, torch.Tensor]]


class TensorOperationManagerBase:
    def enqueue(self, named_tensors: NamedTensors):
        raise NotImplementedError

    def get_outputs(self) -> List[NamedTensors]:
        raise NotImplementedError


class CombinedManager(TensorOperationManagerBase):
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
