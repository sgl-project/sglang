"""Stage HCCL buffers outside the memory-saver pool of a captured NPU graph."""

from contextlib import contextmanager
from threading import local

import torch
import torch.distributed as dist

_capture = local()


class HCCLGraphStaging:
    """Own the staging tensors for one graph, including across offload cycles."""

    def __init__(self, memory_saver):
        self._memory_saver = memory_saver
        self._buffers = {}
        self._warming_up = True

    @contextmanager
    def activate(self):
        previous = getattr(_capture, "staging", None)
        _capture.staging = self
        try:
            yield
        finally:
            _capture.staging = previous

    def finish_warmup(self):
        self._warming_up = False

    def run(self, collective, output, input_, group):
        stream = torch.npu.current_stream(input_.device)
        key = (
            collective,
            group,
            stream.npu_stream,
            input_.device,
            input_.shape,
            input_.dtype,
            output.device,
            output.shape,
            output.dtype,
        )
        if key not in self._buffers:
            if not self._warming_up:
                raise RuntimeError(
                    f"HCCL graph staging was not warmed up for {collective.__name__} "
                    f"with shapes {tuple(input_.shape)} and {tuple(output.shape)} "
                    "on the current stream"
                )
            with self._memory_saver.disable():
                self._buffers[key] = (
                    torch.empty(input_.shape, dtype=input_.dtype, device=input_.device),
                    torch.empty(output.shape, dtype=output.dtype, device=output.device),
                )
        send, recv = self._buffers[key]
        send.copy_(input_)
        collective(recv, send, group=group)
        output.copy_(recv)


def run_hccl_graph_collective(collective, output, input_, *, group):
    staging = getattr(_capture, "staging", None)
    if staging is not None and dist.get_backend(group) == "hccl":
        staging.run(collective, output, input_, group)
    else:
        collective(output, input_, group=group)
