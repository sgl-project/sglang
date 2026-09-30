# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level API for the FlyDSL UltraQuant 4-bit decode specialization.

Replaces stage 1 of the Triton UltraQuant decode on gfx950 for head_dim 256
and GQA 6/8/16. Stage 2 is sglang's existing decode reducer: the kernel writes
``attn_logits``/``attn_lse`` in that reducer's layout.

The kernel rotates the query in-register, so it takes the RAW query.
"""

from __future__ import annotations

import functools
import importlib.util

import torch

from sglang.srt.layers.quantization.ultraquant_tensor import code_bytes, n_groups
from sglang.srt.utils import next_power_of_2

_HEAD_SIZE = 256
_SUPPORTED_GQA = (6, 8, 16)
# The kernel addresses every buffer through a 32-bit byte offset.
_MAX_BUFFER_BYTES = 1 << 32

# Partition p walks the KV in blocks p, p + P, p + 2P, ... of this many tokens,
# so the reducer must be told which partitions hold data (`strided_block_kv`).
ULTRAQUANT_DECODE_BLOCK_KV = 256

# Past about this many workgroups per CU, extra splits no longer hide memory
# latency and only add reducer work.
_WORKGROUPS_PER_CU = 8
# Bounds the fp32 split buffers kept for the largest graph batch.
_MAX_KV_SPLITS = 256


def ultraquant_decode_max_kv_splits(
    base_max_kv_splits: int, max_context_len: int
) -> int:
    """Split-buffer width: one partition per tile-group at the max context."""
    tile_groups = -(-max(max_context_len, 1) // ULTRAQUANT_DECODE_BLOCK_KV)
    return max(base_max_kv_splits, min(_MAX_KV_SPLITS, next_power_of_2(tile_groups)))


@functools.cache
def ultraquant_decode_num_kv_splits(
    batch_size: int,
    num_kv_heads: int,
    min_kv_splits: int,
    max_kv_splits: int,
    core_count: int,
) -> int:
    """Splits for one launch: enough workgroups to fill the GPU, and no more."""
    if core_count <= 0:
        return max_kv_splits
    target = _WORKGROUPS_PER_CU * core_count
    splits = next_power_of_2(-(-target // max(batch_size * num_kv_heads, 1)))
    return max(min_kv_splits, min(max_kv_splits, splits))


@functools.cache
def _rocm_arch(device_index: int) -> str | None:
    properties = torch.cuda.get_device_properties(device_index)
    return getattr(properties, "gcnArchName", "").split(":")[0] or None


@functools.cache
def _flydsl_available() -> bool:
    return importlib.util.find_spec("flydsl") is not None


@functools.cache
def _kernel():
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from .kernels import ultraquant_decode_hd256 as kmod

    assert kmod.HEAD_SIZE == _HEAD_SIZE
    assert kmod.KV_COMPUTE_BLOCK == ULTRAQUANT_DECODE_BLOCK_KV
    return kmod.create_ultraquant_decode_hd256_kernel, _run_compiled


def is_flydsl_ultraquant_decode_supported(
    head_dim: int,
    query_group_size: int,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> bool:
    """Return whether this gfx950-only specialization covers the given shape."""
    if (
        head_dim != _HEAD_SIZE
        or query_group_size not in _SUPPORTED_GQA
        or dtype != torch.bfloat16
    ):
        return False
    if not torch.cuda.is_available() or not _flydsl_available():
        return False
    index = device.index if device is not None else None
    if index is None:
        index = torch.cuda.current_device()
    return _rocm_arch(index) == "gfx950"


def flydsl_ultraquant_decode_fits(
    num_seqs: int, num_q_heads: int, split_stride: int
) -> bool:
    """Whether ``attn_logits`` for this batch fits the kernel's 32-bit offsets."""
    return num_seqs * num_q_heads * split_stride * _HEAD_SIZE * 4 <= _MAX_BUFFER_BYTES


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    # Strides are baked into the kernel, so the layout must be dense.
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"`{name}` must be a contiguous {dtype} tensor of shape {list(shape)} "
            f"on {device}, got {tensor.dtype} {list(tensor.shape)} with strides "
            f"{tensor.stride()} on {tensor.device}."
        )


def _check_split_buffer(
    name: str,
    tensor: torch.Tensor,
    num_seqs: int,
    trailing: tuple[int, ...],
    device: torch.device,
) -> None:
    """Check a split-K output whose batch dim may be padded past ``num_seqs``."""
    if tensor.shape[0] < num_seqs:
        raise ValueError(
            f"`{name}` covers {tensor.shape[0]} sequences, need {num_seqs}."
        )
    _check_tensor(name, tensor, (tensor.shape[0], *trailing), torch.float32, device)


def flydsl_ultraquant_decode(
    query: torch.Tensor,
    k_code_buffer: torch.Tensor,
    k_scale_buffer: torch.Tensor,
    v_code_buffer: torch.Tensor,
    v_scale_buffer: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    num_splits: int | None = None,
) -> None:
    """Run stage 1 of the UltraQuant decode, filling ``attn_logits``/``attn_lse``.

    ``query`` is the raw (unrotated) bf16 ``[num_seqs, num_q_heads, 256]``
    tensor. The first ``num_splits`` partitions (default: all
    ``attn_logits.shape[2]``) are each written; reduce them with
    ``num_kv_splits`` set to that count and
    ``strided_block_kv=ULTRAQUANT_DECODE_BLOCK_KV``.
    """
    device = query.device
    if query.dim() != 3:
        raise ValueError(f"`query` must have rank 3, got rank {query.dim()}.")
    num_seqs, num_q_heads, head_dim = query.shape
    num_kv_heads = k_code_buffer.shape[1]
    if num_q_heads % num_kv_heads:
        raise ValueError(
            f"`query` head count {num_q_heads} is not a multiple of the "
            f"{num_kv_heads} KV heads."
        )
    query_group_size = num_q_heads // num_kv_heads
    if not is_flydsl_ultraquant_decode_supported(
        head_dim, query_group_size, query.dtype, device
    ):
        raise RuntimeError(
            "`flydsl_ultraquant_decode` requires flydsl on a gfx950 GPU, a bf16 "
            f"query, head_dim 256 and GQA {_SUPPORTED_GQA}; got "
            f"dtype={query.dtype}, head_dim={head_dim}, GQA={query_group_size}."
        )
    # The two outer query strides are baked in; the head dim must be dense.
    if query.stride(2) != 1:
        raise ValueError("`query` must be contiguous along the head dim.")

    split_stride = attn_logits.shape[2]
    if num_splits is None:
        num_splits = split_stride
    if not 0 < num_splits <= split_stride:
        raise ValueError(
            f"`num_splits` must be in [1, {split_stride}], got {num_splits}."
        )
    _check_split_buffer(
        "attn_logits",
        attn_logits,
        num_seqs,
        (num_q_heads, split_stride, head_dim),
        device,
    )
    _check_split_buffer(
        "attn_lse", attn_lse, num_seqs, (num_q_heads, split_stride), device
    )
    if not flydsl_ultraquant_decode_fits(num_seqs, num_q_heads, split_stride):
        raise ValueError(
            f"`attn_logits` for {num_seqs} sequences overflows the 32-bit buffer "
            "offset this kernel addresses with."
        )
    if (
        kv_indptr.numel() < num_seqs + 1
        or kv_indptr.dtype != torch.int32
        or kv_indptr.device != device
        or kv_indptr.stride(-1) != 1
    ):
        raise ValueError(
            "`kv_indptr` must be a contiguous int32 tensor on the query device "
            f"with at least {num_seqs + 1} entries, got {list(kv_indptr.shape)} "
            f"{kv_indptr.dtype}."
        )
    # The kernel reads the low dword of each entry, so the width is load-bearing.
    if (
        kv_indices.dtype != torch.int64
        or kv_indices.device != device
        or kv_indices.stride(-1) != 1
    ):
        raise ValueError(
            "`kv_indices` must be a contiguous int64 tensor on the query device."
        )

    num_slots = k_code_buffer.shape[0]
    for name, buf, row in (
        ("k_code_buffer", k_code_buffer, code_bytes(head_dim)),
        ("v_code_buffer", v_code_buffer, code_bytes(head_dim)),
        ("k_scale_buffer", k_scale_buffer, n_groups(head_dim)),
        ("v_scale_buffer", v_scale_buffer, n_groups(head_dim)),
    ):
        _check_tensor(name, buf, (num_slots, num_kv_heads, row), torch.uint8, device)
    # Codes are the largest of the four KV buffers, so they bound the pool.
    if k_code_buffer.numel() > _MAX_BUFFER_BYTES - 1:
        raise ValueError(
            f"UltraQuant KV pool is {k_code_buffer.numel()} B per code buffer, "
            "which overflows the 32-bit buffer offset this kernel addresses with."
        )

    create_kernel, run_compiled = _kernel()
    launch = create_kernel(
        num_kv_heads=num_kv_heads,
        num_partitions=num_splits,
        softmax_scale=float(softmax_scale),
        query_group_size=query_group_size,
        stride_q_seq=query.stride(0),
        stride_q_head=query.stride(1),
        split_stride=split_stride,
    )
    with torch.cuda.device(device):
        run_compiled(
            launch,
            attn_logits,
            attn_lse,
            query,
            k_code_buffer,
            k_scale_buffer,
            v_code_buffer,
            v_scale_buffer,
            kv_indptr,
            kv_indices,
            num_seqs,
            torch.cuda.current_stream(device),
        )
