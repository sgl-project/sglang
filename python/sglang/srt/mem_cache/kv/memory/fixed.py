"""Stable allocations outside the movable KV chain, owned by its plan."""

import math
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

    def regions(self, parts: Sequence[TensorPartDecl]):
        """Contiguous fixed rows in stable part-major, then layer order."""
        from sglang.srt.mem_cache.kv.storage import Region

        ptrs, lens, items, dims, outer, layers = [], [], [], [], [], []
        for part in parts:
            tensor = self.tensors[part.name]
            views = tensor.unbind(0) if part.layer_ids else (tensor,)
            layer_ids = part.layer_ids or (-1,)
            for layer_id, view in zip(layer_ids, views):
                if view.numel() == 0:
                    continue
                assert view.is_contiguous(), "fixed transfer rows must be contiguous"
                ptrs.append(view.data_ptr())
                lens.append(view.nbytes)
                items.append(view[0].nbytes)
                dims.append(
                    0 if part.slice_axis is None else part.shape[part.slice_axis]
                )
                outer.append(
                    1
                    if part.slice_axis is None
                    else math.prod(part.shape[: part.slice_axis])
                )
                layers.append(layer_id)
        return Region(
            ptrs=tuple(ptrs),
            lens=tuple(lens),
            item_lens=tuple(items),
            dim_per_tensor=tuple(dims),
            slice_outer_counts=tuple(outer),
            layer_ids=tuple(layers),
        )
