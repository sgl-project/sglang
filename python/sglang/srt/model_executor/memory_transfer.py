from typing import List, Tuple

import torch

NamedTensors = List[Tuple[str, torch.Tensor]]


class TensorOperationManagerBase:
    def enqueue(self, named_tensors: NamedTensors):
        raise NotImplementedError

    def get_outputs(self) -> List[NamedTensors]:
        raise NotImplementedError


class AsyncPinMemoryManager(TensorOperationManagerBase):
    TODO


class AsyncToCudaManager(TensorOperationManagerBase):
    TODO
