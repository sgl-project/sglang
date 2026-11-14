# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
) -> torch.LongTensor:
    # cu_seqlens_cpu: the same offsets on the host. The indices are then built
    # from them and copied in asynchronously instead of reading cu_seqlens back,
    # which waits for the device.
    lens = prepare_lens(cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu)
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()]
    )
    indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
    if cu_seqlens_cpu is None:
        return indices.to(cu_seqlens)
    return (
        indices.to(cu_seqlens.dtype)
        .pin_memory()
        .to(cu_seqlens.device, non_blocking=True)
    )


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    # The leading zero is padded on the device: a host-built one would be
    # copied in synchronously.
    return torch.nn.functional.pad(
        triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0)
    ).cumsum(-1)
