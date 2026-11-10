from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Sequence, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

L2_PREFETCH_CHUNK_BYTES = 4096


@cache_once
def _l2_prefetch_module() -> Module:
    return load_jit(
        "l2_prefetch",
        cuda_files=["speculative/l2_prefetch.cuh"],
        cuda_wrappers=[("run", "L2PrefetchKernel::run")],
    )


def l2_prefetch_chunks(
    spans: Iterable[Tuple[int, int]], chunk_bytes: int = L2_PREFETCH_CHUNK_BYTES
) -> List[Tuple[int, int]]:
    if chunk_bytes <= 0 or chunk_bytes % 16:
        raise ValueError(
            f"chunk_bytes must be a positive multiple of 16, got {chunk_bytes}"
        )
    chunks = []
    for addr, nbytes in spans:
        if nbytes <= 0:
            continue
        lo, hi = addr & ~15, (addr + nbytes + 15) & ~15
        chunks.extend((a, min(chunk_bytes, hi - a)) for a in range(lo, hi, chunk_bytes))
    return chunks


def build_l2_prefetch_table(
    tensors: Sequence[torch.Tensor], chunk_bytes: int = L2_PREFETCH_CHUNK_BYTES
) -> torch.Tensor:
    if not tensors:
        raise ValueError("no tensors to prefetch")
    spans = [(t.data_ptr(), t.numel() * t.element_size()) for t in tensors]
    rows = l2_prefetch_chunks(spans, chunk_bytes=chunk_bytes)
    return torch.tensor(rows, dtype=torch.int64, device=tensors[0].device).reshape(
        -1, 2
    )


def l2_prefetch(table: torch.Tensor) -> None:
    _l2_prefetch_module().run(table)
