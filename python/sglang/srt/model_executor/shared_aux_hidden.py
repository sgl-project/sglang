"""Persistent DFlash target outputs shared across graph batch sizes."""

import torch


class SharedAuxHiddenBuffers:
    def __init__(self):
        self._buffers = {}

    def get(self, stream, *, rows, max_rows, width, dtype, device):
        if not 0 <= rows <= max_rows:
            raise ValueError("auxiliary output rows exceed graph capacity")
        key = (stream, max_rows, width, dtype, torch.device(device))
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = torch.empty((max_rows, width), dtype=dtype, device=device)
            self._buffers[key] = buffer
        return buffer[:rows]
