"""FIFO full-row activation transport and CUDA storage dependencies for VPP."""

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


class _CommStream:
    def __init__(self, pp_group):
        device = torch.device(pp_group.device)
        self.module = torch.get_device_module(device)
        self.stream = (
            self.module.Stream(device=device) if device.type != "cpu" else None
        )

    def context(self):
        return (
            self.module.stream(self.stream)
            if self.stream is not None
            else nullcontext()
        )

    def record_event(self):
        if self.stream is None:
            return None
        event = self.module.Event()
        event.record(self.stream)
        return event


@dataclass
class VPPReceivedActivation:
    tensors: dict[str, Any]
    recv_event: Any = None

    def wait_on(self, forward_stream) -> None:
        """Order compute after receive and protect storage on the compute stream."""
        if self.recv_event is not None:
            forward_stream.wait_event(self.recv_event)
        for tensor in self.tensors.values():
            if isinstance(tensor, torch.Tensor) and not tensor.is_cpu:
                tensor.record_stream(forward_stream)


class VPPActivationReceiver:
    """One posted FIFO receive; metadata progress never blocks the control loop.

    take() may block on CPU metadata, but GPU payload completion is represented
    by a local event. Call wait_on() before the forward consumes this result.
    The sender transmits full tensors on each TP lane, without TP all-gather.
    """

    def __init__(self, pp_group, *, tag: int = 0):
        self.pp_group = pp_group
        self.tag = tag
        self._comm = _CommStream(pp_group)
        self._handle = None

    def start(self) -> None:
        if self._handle is not None:
            raise RuntimeError("VPP activation receiver is already started")
        self._post()

    def _post(self) -> None:
        with self._comm.context():
            self._handle = self.pp_group.recv_tensor_dict_async(tag=self.tag)

    def progress(self) -> bool:
        if self._handle is None:
            raise RuntimeError("VPP activation receiver is not started")
        with self._comm.context():
            return self._handle.poll_metadata_ready()

    def take(self, *, post_next: bool = True) -> VPPReceivedActivation:
        """Consume one activation; end a finite wave with post_next=False.

        The next wave calls start() again. Ending the receive chain leaves no
        metadata waiter outstanding across an idle period or process shutdown.
        """
        if self._handle is None:
            raise RuntimeError("VPP activation receiver is not started")
        with self._comm.context():
            tensors = self._handle.wait()
            result = VPPReceivedActivation(tensors, self._comm.record_event())
            self._handle = None
            if post_next:
                self._post()
        return result


class VPPSendWindow:
    """Bound the number of submitted activation sends, identically on every TP.

    acquire() releases oldest sends in submission order, never by local polling.
    Its returned event must be waited on before reusing released send buffers.
    For shared CUDA-graph output storage, call wait_for_reuse() before replay:
    it fences all outstanding readers, including sends still inside the window.
    """

    def __init__(self, pp_group, max_inflight: int, *, tag: int = 0):
        if type(max_inflight) is not int or max_inflight <= 0:
            raise ValueError("max_inflight must be a positive integer")
        self.pp_group = pp_group
        self.tag = tag
        self.max_inflight = max_inflight
        self._comm = _CommStream(pp_group)
        self._pending = deque()

    def __len__(self) -> int:
        return len(self._pending)

    def _release_oldest(self) -> None:
        items = self._pending[0]
        for item in items:
            item.wait()
            tensor = item.payload
            if isinstance(tensor, torch.Tensor) and not tensor.is_cpu:
                # Work.wait only orders the stream. record_stream keeps the
                # allocator from recycling storage while NCCL still reads it.
                tensor.record_stream(self._comm.stream)
        self._pending.popleft()

    def acquire(self):
        """Reserve capacity before forward; return a released-buffer fence."""
        if len(self._pending) < self.max_inflight:
            return None
        with self._comm.context():
            self._release_oldest()
            return self._comm.record_event()

    def send(self, tensors: dict[str, Any], forward_event=None) -> None:
        if len(self._pending) >= self.max_inflight:
            raise RuntimeError("Acquire VPP send credit before forwarding")
        if self._comm.stream is not None and forward_event is None:
            raise ValueError("A device activation send requires a forward event")
        with self._comm.context():
            if forward_event is not None:
                if self._comm.stream is None:
                    raise ValueError("A forward event requires a device stream")
                self._comm.stream.wait_event(forward_event)
            items = self.pp_group.send_tensor_dict(
                tensors, async_send=True, tag=self.tag
            )
            if not isinstance(items, list):
                raise RuntimeError("VPP activation send requires a distributed peer")
            self._pending.append(items)

    def wait_for_reuse(self):
        """Fence all sent storage before graph replay or shutdown."""
        if not self._pending:
            return None
        with self._comm.context():
            while self._pending:
                self._release_oldest()
            return self._comm.record_event()
