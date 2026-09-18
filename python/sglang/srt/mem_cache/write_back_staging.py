from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WriteBackStaging:
    """Persistent transfer buffers, owned by the device pool across KV sizing."""

    buffers: tuple[torch.Tensor, ...]

    @classmethod
    def allocate(
        cls,
        shapes: tuple[tuple[int, ...], ...],
        *,
        dtype: torch.dtype,
        device: str,
    ) -> WriteBackStaging:
        return cls(
            tuple(torch.empty(shape, dtype=dtype, device=device) for shape in shapes)
        )

    @property
    def nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self.buffers)

    def views(
        self, shapes: tuple[tuple[int, ...], ...], *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        if len(shapes) != len(self.buffers):
            raise ValueError("HiCache staging buffer count changed after preparation")
        for buffer, shape in zip(self.buffers, shapes):
            if (
                buffer.dtype != dtype
                or tuple(buffer.shape[1:]) != shape[1:]
                or buffer.shape[0] < shape[0]
            ):
                raise ValueError("HiCache staging geometry changed after preparation")
        return tuple(buffer[: shape[0]] for buffer, shape in zip(self.buffers, shapes))
