"""Batch short KV ranges while preserving registered, layer-major Host tensors."""

from __future__ import annotations

import torch
from sglang.srt.training_capture.protocol import ContractError


class KVStaging:
    def __init__(self, device_tensors, host_tensors):
        self.device_tensors = device_tensors
        self.host_tensors = host_tensors
        first = next(iter(device_tensors.values()))
        self.device = first.device
        self.capacity = first.shape[0]
        self.start = self.end = 0
        self.stream = None
        self.transfer_uncertain = False

    def export(self, exporter, slots, *, start, end):
        if start != self.end or slots.numel() != end - start:
            raise ContractError("staged KV ranges must append without gaps")
        stream = (
            torch.cuda.current_stream(self.device)
            if self.device.type == "cuda"
            else None
        )
        if self.stream is not None and stream != self.stream:
            try:
                stream.wait_stream(self.stream)
            except Exception:
                self.transfer_uncertain = True
                raise
        self.stream = stream
        offset = 0
        while self.end < end:
            pending = self.end - self.start
            remaining = end - self.end
            # Large prefill ranges already amortize DMA without device staging.
            if not pending and remaining >= self.capacity:
                exporter.export(slots[offset:], self.host_tensors, self.end, end)
                self.start = self.end = end
                return
            count = min(self.capacity - pending, remaining)
            exporter.export(
                slots[offset : offset + count],
                self.device_tensors,
                pending,
                pending + count,
            )
            self.end += count
            offset += count
            if self.end - self.start == self.capacity:
                self.flush()

    def flush(self):
        """Caller selects the producer stream and fences partial failures."""
        count = self.end - self.start
        if not count:
            return
        for name, source in self.device_tensors.items():
            self.host_tensors[name][self.start : self.end].copy_(
                source[:count], non_blocking=source.is_cuda
            )
        self.start = self.end
