"""Streams for NCCL EP communication and eligible TBO compute lanes.

The caller owns submission ordering. A send waits for its producer; completion
joins back to the consumer. In particular, completion must NOT wait for work
submitted by the compute stream between send_only and complete.
"""

from contextlib import contextmanager

import torch

from sglang.srt.runtime_context import get_flags, get_resources


class NcclEpStream:
    def __init__(self, device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Initialize the NCCL EP side stream during warmup")
        self.stream = torch.cuda.Stream(device=device)

    @contextmanager
    def send(self, *inputs):
        producer = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(producer)
        # Also required during capture: otherwise the graph pool may reuse a
        # producer's temporary allocation before the side-stream copy reads it.
        # A Python reference through complete() is not a GPU completion fence.
        for tensor in inputs:
            tensor.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            yield self.stream

    def complete(self, handle):
        consumer = torch.cuda.current_stream(self.stream.device)
        with torch.cuda.stream(self.stream):
            handle.complete(config=0, stream=self.stream.cuda_stream)
        consumer.wait_stream(self.stream)


def get_nccl_ep_stream(device, instance_id=0, *, role="communication"):
    if not get_flags().moe.nccl_ep_multistream:
        return None
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = f"nccl_ep_stream_{device.index}_{role}_{instance_id}"
    buffers = get_resources().buffers
    if key not in buffers:
        buffers[key] = NcclEpStream(device)
    return buffers[key]


def destroy_nccl_ep_streams():
    # Called after Graph executables and native groups have been retired.
    buffers = get_resources().buffers
    for key in list(buffers):
        if isinstance(key, str) and key.startswith("nccl_ep_stream_"):
            buffers[key].stream.synchronize()
            del buffers[key]
