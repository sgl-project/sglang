import queue
from dataclasses import dataclass
from queue import SimpleQueue
from threading import Thread
from typing import List, Tuple, Optional

import torch

NamedTensors = List[Tuple[str, torch.Tensor]]


# For simplicity, classes here does not have tagging etc
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
    def __init__(self):
        self._input_queue = SimpleQueue()
        self._output_queue = SimpleQueue()
        self._background_thread = None

    def enqueue(self, named_tensors: NamedTensors):
        self._auto_create_background_thread()
        self._input_queue.put_nowait(named_tensors)

    def get_outputs(self) -> List[NamedTensors]:
        outputs = []
        while True:
            try:
                outputs.append(self._output_queue.get_nowait())
            except queue.Empty:
                break
        return outputs

    def _auto_create_background_thread(self):
        if self._background_thread is not None:
            return

        self._background_thread = Thread(target=self._background_thread_entrypoint)
        self._background_thread.start()

    def _background_thread_entrypoint(self):
        TODO


class AsyncToCudaManager(TensorOperationManagerBase):
    def __init__(self):
        self._inflight_tasks: List[_AsyncToCudaTask] = []
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
        while len(self._inflight_tasks) > 0 and self._inflight_tasks[0].finish_event.query():
            task = self._inflight_tasks.pop(0)
            outputs.append(self._handle_one_output(task))
        return outputs

    def _handle_one_output(self, task: "_AsyncToCudaTask"):
        torch.cuda.current_stream().wait_stream(self._alt_stream)
        return task.output_named_tensors

    def _auto_create_stream(self):
        if self._alt_stream is None:
            self._alt_stream = torch.cuda.Stream()


@dataclass
class _AsyncToCudaTask:
    finish_event: torch.cuda.Event
    input_named_tensors: NamedTensors
    output_named_tensors: NamedTensors
