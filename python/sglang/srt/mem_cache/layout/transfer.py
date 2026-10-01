"""Byte layouts for pieces of a page-major allocation.

These describe payloads inside an already registered raw allocation. They are
not separate registration regions: layer views overlap, and the stride between
pages/slots is generally larger than an individual tensor's payload.
"""

from dataclasses import dataclass
from math import prod
from typing import Optional, Tuple


@dataclass(frozen=True)
class TransferTensor:
    name: str
    layer_id: int
    offset_bytes: int
    shape: Tuple[int, ...]
    itemsize: int
    slice_axis: Optional[int] = None
    # Full (unsharded) dimensions of independently sharded conv components.
    shard_groups: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.offset_bytes < 0 or self.itemsize <= 0:
            raise ValueError("Invalid transfer tensor offset or item size")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError("Transfer tensors must have nonempty positive shapes")
        if self.slice_axis is not None and not 0 <= self.slice_axis < len(self.shape):
            raise ValueError("Transfer tensor slice axis is out of range")

    @property
    def row_bytes(self) -> int:
        return prod(self.shape) * self.itemsize

    @property
    def slice_dim(self) -> int:
        return 0 if self.slice_axis is None else self.shape[self.slice_axis]

    @property
    def outer_count(self) -> int:
        return 1 if self.slice_axis is None else prod(self.shape[: self.slice_axis])


@dataclass(frozen=True)
class TransferLayout:
    # Distance between physical pages (KV) or physical slots (state).
    block_bytes: int
    rows_per_block: int
    tensors: Tuple[TransferTensor, ...]

    def __post_init__(self):
        if self.block_bytes <= 0 or self.rows_per_block <= 0 or not self.tensors:
            raise ValueError("Invalid transfer block layout")
        end = 0
        for tensor in sorted(self.tensors, key=lambda t: t.offset_bytes):
            if tensor.offset_bytes < end:
                raise ValueError("Transfer tensor payloads overlap")
            end = tensor.offset_bytes + self.rows_per_block * tensor.row_bytes
        if end > self.block_bytes:
            raise ValueError("Transfer tensor extends past its block")

    def address(self, base: int, block: int, tensor: int, row: int = 0) -> int:
        """Address a physical block; callers translate virtual IDs first."""
        if block < 0 or not 0 <= row < self.rows_per_block:
            raise ValueError("Invalid physical block or row")
        entry = self.tensors[tensor]
        return (
            base + block * self.block_bytes + entry.offset_bytes + row * entry.row_bytes
        )
