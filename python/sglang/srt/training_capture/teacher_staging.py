"""Bounded request-owned teacher rows, flushed to registered Host tensors."""

from __future__ import annotations

import torch
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.teacher import TeacherRows


class TeacherStaging:
    def __init__(self, device_tensors, host_tensors):
        self.device_tensors = device_tensors
        self.host_tensors = host_tensors
        first = next(iter(device_tensors.values()))
        self.device = first.device
        self.capacity = first.shape[0]
        self.start = self.end = 0
        self.stream = None
        self.transfer_uncertain = False

    def append(self, rows: TeacherRows, *, row: int, start: int, count: int):
        if start != self.end or count < 1:
            raise ContractError("staged teacher rows must append without gaps")
        sources = {
            "teacher_topk_ids": rows.token_ids,
            "teacher_topk_logits": rows.logits,
            "teacher_logsumexp": rows.logsumexp,
        }
        source_device = rows.logits.device
        for name, source in sources.items():
            target = self.device_tensors[name]
            if (
                source.device != source_device
                or (source_device != self.device and source_device.type != "cpu")
                or source.dtype != target.dtype
                or source.shape[1:] != target.shape[1:]
                or not 0 <= row < row + count <= source.shape[0]
                or start + count > self.host_tensors[name].shape[0]
            ):
                raise ContractError("teacher staging source differs from its buffer")
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
        # P/D's first teacher row already arrives on Host. Keep it there, after
        # flushing any preceding device rows, and advance the same row ledger.
        host_source = source_device.type == "cpu" and self.device.type != "cpu"
        if host_source:
            self.flush()
        remaining = count
        while remaining:
            pending = self.end - self.start
            direct = host_source or (not pending and remaining >= self.capacity)
            take = remaining if direct else min(self.capacity - pending, remaining)
            destinations = self.host_tensors if direct else self.device_tensors
            offset = self.end if direct else pending
            for name, source in sources.items():
                destinations[name][offset : offset + take].copy_(
                    source[row : row + take], non_blocking=source.is_cuda
                )
                if source.is_cuda:
                    source.record_stream(stream)
            self.end += take
            row += take
            remaining -= take
            if direct:
                self.start = self.end
            elif self.end - self.start == self.capacity:
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
