"""Keep HCCL-registered graph communication buffers outside pauseable pools.

Only capture warmup/capture uses staging. Eager execution is unchanged. Buffers
are shared by matching operations on the same stream and live with the group;
the captured copies preserve graph intermediates and output ownership.
"""

import threading
from contextlib import contextmanager

import torch
import torch.distributed as dist

_capture = threading.local()


@contextmanager
def stage_graph_collectives(enabled):
    previous = getattr(_capture, "enabled", False)
    _capture.enabled = bool(enabled)
    try:
        yield
    finally:
        _capture.enabled = previous


class GraphCollectiveStaging:
    def __init__(self, group):
        self.group = group
        self.buffers = {}

    def _buffers(self, operation, output, input):
        stream = torch.npu.current_stream(input.device)
        key = (
            operation,
            tuple(input.shape),
            tuple(output.shape),
            input.dtype,
            output.dtype,
            input.device,
            stream.npu_stream,
        )
        buffers = self.buffers.get(key)
        if buffers is None:
            # NPUCudaGraphBackend runs warmups on this same capture stream.
            # A missed warmup must not silently allocate registered HCCL memory
            # into the graph's pauseable allocator region.
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError(
                    "HCCL graph staging shape was not warmed before capture"
                )
            buffers = (
                torch.empty(tuple(input.shape), dtype=input.dtype, device=input.device),
                torch.empty(
                    tuple(output.shape), dtype=output.dtype, device=output.device
                ),
            )
            self.buffers[key] = buffers
        return buffers

    def all_gather_into_tensor(self, output, input):
        if not getattr(_capture, "enabled", False):
            return dist.all_gather_into_tensor(output, input, group=self.group)
        source, target = self._buffers("all_gather", output, input)
        source.copy_(input)
        dist.all_gather_into_tensor(target, source, group=self.group)
        output.copy_(target)

    def reduce_scatter_tensor(self, output, input):
        if not getattr(_capture, "enabled", False):
            return dist.reduce_scatter_tensor(output, input, group=self.group)
        source, target = self._buffers("reduce_scatter", output, input)
        source.copy_(input)
        dist.reduce_scatter_tensor(target, source, group=self.group)
        output.copy_(target)
