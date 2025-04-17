from dataclasses import dataclass
from typing import List, Tuple, Optional

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
        outputs_a = self._manager_a.get_outputs()
        for output_a in outputs_a:
            self._manager_b.enqueue(output_a)

        return self._manager_b.get_outputs()


class AsyncPinMemoryManager(TensorOperationManagerBase):
    def enqueue(self, named_tensors: NamedTensors):
        TODO

    def get_outputs(self) -> List[NamedTensors]:
        return TODO


class AsyncToCudaManager(TensorOperationManagerBase):
    def __init__(self):
        self._inflight_tasks = []
        self._alt_stream: Optional[torch.cuda.Stream] = None

    def enqueue(self, named_tensors: NamedTensors):
        self._auto_create_stream()

        self._alt_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._alt_stream):
            output_named_tensors = [
                (name, tensor.to("cuda", non_blocking=True))
                for name, tensor in named_tensors
            ]
            finish_event = torch.cuda.Event()
            finish_event.record()

        self._inflight_tasks.append(_AsyncToCudaTask(
            finish_event=finish_event,
            input_named_tensors=named_tensors,
            output_named_tensors=output_named_tensors,
        ))

    def get_outputs(self) -> List[NamedTensors]:
        outputs = []
        while len(self._inflight_tasks) > 0:
            TODO
        return outputs

    def _auto_create_stream(self):
        if self._alt_stream is None:
            self._alt_stream = torch.cuda.Stream()


@dataclass
class _AsyncToCudaTask:
    finish_event: torch.cuda.Event
    input_named_tensors: NamedTensors
    output_named_tensors: NamedTensors
