"""Stable allocations outside the movable KV chain, owned by its plan."""

from contextlib import nullcontext
from typing import Sequence

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.kv.decl import TensorPartDecl
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class FixedMemory:
    """Typed fixed rows allocated before capture, with declaration-driven reset."""

    def __init__(
        self,
        parts: Sequence[TensorPartDecl],
        *,
        rows: int,
        device: str,
        enable_memory_saver: bool = False,
    ):
        self.parts = tuple(parts)
        if len({p.name for p in parts}) != len(parts):
            raise ValueError("fixed part names must be unique")
        if rows < 1:
            raise ValueError("fixed memory needs the padding row")
        self.rows = rows
        self.device = device
        self.tensors = {}
        if not self.parts:
            return
        adapter = TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        enabled, pool, _ = maybe_init_custom_mem_pool(device=device)
        with (
            adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            torch.cuda.use_mem_pool(pool) if enabled else nullcontext(),
        ):
            for part in self.parts:
                self.tensors[part.name] = torch.full(
                    part.tensor_shape(rows),
                    part.reset_value,
                    dtype=part.dtype,
                    device=device,
                )

    @property
    def nbytes(self) -> int:
        return self.rows * sum(part.row_bytes for part in self.parts)

    def reset_rows(self, rows) -> None:
        if not self.parts:
            return
        if not isinstance(rows, torch.Tensor):
            rows = torch.tensor(rows, dtype=torch.int64)
        indices = rows.to(device=self.device, dtype=torch.int64, non_blocking=True)
        for part in self.parts:
            self.tensors[part.name].index_fill_(
                part.row_axis, indices, part.reset_value
            )

    def copy_rows(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Snapshot source rows before writing, including overlapping remaps."""
        for part in self.parts:
            tensor = self.tensors[part.name]
            values = tensor.index_select(part.row_axis, src.to(torch.int64))
            tensor.index_copy_(part.row_axis, dst.to(torch.int64), values)
