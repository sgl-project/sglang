# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import collections
import contextlib

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


# (cu_seqlens tensor, per-sequence lengths on the host) for the extend call in
# flight: lets the chunk index builders skip a GPU->host .tolist() sync.
_CPU_LENS_HINT = None
# pinned sources of in-flight non_blocking H2D copies: freeing one early lets the
# host allocator hand the buffer out again before the DMA has read it
_PINNED_KEEPALIVE = collections.deque(maxlen=64)


@contextlib.contextmanager
def cpu_seqlens_hint(cu_seqlens: torch.Tensor, lens_cpu):
    global _CPU_LENS_HINT
    prev = _CPU_LENS_HINT
    _CPU_LENS_HINT = (cu_seqlens, [int(x) for x in lens_cpu])
    try:
        yield
    finally:
        _CPU_LENS_HINT = prev


def _hinted_lens(cu_seqlens):
    h = _CPU_LENS_HINT
    if h is None:
        return None
    t, lens = h
    if cu_seqlens is t or (
        cu_seqlens.data_ptr() == t.data_ptr()
        and cu_seqlens.numel() == t.numel()
        and cu_seqlens.dtype == t.dtype
    ):
        return lens
    return None


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    lens = _hinted_lens(cu_seqlens)
    if lens is not None:
        indices = torch.cat(
            [torch.arange(-(-n // chunk_size)) for n in lens]
            or [torch.zeros(0, dtype=torch.long)]
        )
        out = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        pinned = out.to(cu_seqlens.dtype).pin_memory()
        _PINNED_KEEPALIVE.append(pinned)
        return pinned.to(cu_seqlens.device, non_blocking=True)
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
