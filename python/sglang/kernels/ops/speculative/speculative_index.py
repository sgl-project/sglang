"""Cached-descriptor CUDA kernels for EAGLE decode index preparation."""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs
from sglang.srt.utils.common import is_cuda

_INDEX_DTYPES = (torch.int32, torch.int64)
_is_cuda = is_cuda()
_descriptors: dict[str, tuple[torch.Tensor, object]] = {}


def _runtime_optimization_enabled() -> bool:
    """Keep the generic index kernels behind the runtime master switch."""
    return envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()


@cache_once
def _module():
    return load_jit(
        "speculative_index",
        cuda_files=["elementwise/speculative_index.cuh"],
        cuda_wrappers=[
            ("chain_tree", "SpeculativeIndex::chain_tree"),
            ("extend_layout", "SpeculativeIndex::extend_layout"),
            ("draft_locs", "SpeculativeIndex::draft_locs"),
        ],
    )


def _descriptor(name: str, num_entries: int):
    entry = _descriptors.get(name)
    if entry is None:
        tensor = torch.zeros((num_entries,), dtype=torch.int64)
        entry = _descriptors[name] = (tensor, tensor.numpy())
    return entry


def _same_cuda_device(anchor: torch.Tensor, *others: torch.Tensor | None) -> bool:
    """Raw pointers below are launched on ``anchor``'s CUDA device."""
    return anchor.is_cuda and all(
        tensor is None or tensor.device == anchor.device for tensor in others
    )


def chain_tree(
    *,
    bonus,
    draft,
    seq,
    mask,
    positions,
    retrieve,
    tokens,
    req,
    table,
    out_cache_loc,
    mrope,
) -> bool:
    if not _is_cuda or not _runtime_optimization_enabled():
        return False
    required = (bonus, draft, seq, mask, positions, retrieve, tokens)
    if not all(isinstance(tensor, torch.Tensor) for tensor in required):
        return False
    cache_tensors = (req, table, out_cache_loc)
    if any(tensor is None for tensor in cache_tensors) != all(
        tensor is None for tensor in cache_tensors
    ):
        return False
    optional = cache_tensors + (mrope,)
    if not all(
        tensor is None or isinstance(tensor, torch.Tensor) for tensor in optional
    ) or not _same_cuda_device(seq, *required, *optional):
        return False
    if draft.ndim != 2:
        return False
    batch_size, depth = draft.shape
    width = depth + 1
    num_tokens = batch_size * width
    if not (
        bonus.ndim == seq.ndim == 1
        and bonus.numel() == seq.numel() == batch_size
        and bonus.dtype in _INDEX_DTYPES
        and draft.dtype in _INDEX_DTYPES
        and seq.dtype in _INDEX_DTYPES
        and mask.dtype is torch.bool
        and mask.is_contiguous()
        and mask.numel() >= batch_size * width * width
        and positions.dtype is torch.int64
        and positions.is_contiguous()
        and positions.shape == (num_tokens,)
        and retrieve.dtype is torch.int64
        and retrieve.is_contiguous()
        and retrieve.shape == (3, batch_size, width)
        and tokens.dtype in _INDEX_DTYPES
        and tokens.is_contiguous()
        and tokens.shape == (num_tokens,)
        and (
            req is None
            or (
                req.ndim == 1
                and req.numel() == batch_size
                and req.dtype in _INDEX_DTYPES
                and table.ndim == 2
                and table.dtype in _INDEX_DTYPES
                and out_cache_loc.dtype is torch.int64
                and out_cache_loc.is_contiguous()
                and out_cache_loc.shape == (num_tokens,)
            )
        )
        and (
            mrope is None
            or (
                mrope.dtype is torch.int64
                and mrope.is_contiguous()
                and mrope.shape == (3, num_tokens)
            )
        )
    ):
        return False
    descriptor, values = _descriptor("chain_tree", 27)
    values[0] = bonus.data_ptr()
    values[1] = draft.data_ptr()
    values[2] = seq.data_ptr()
    values[3] = mask.data_ptr()
    values[4] = positions.data_ptr()
    values[5] = retrieve.data_ptr()
    values[6] = tokens.data_ptr()
    values[7] = req.data_ptr() if req is not None else 0
    values[8] = table.data_ptr() if table is not None else 0
    values[9] = out_cache_loc.data_ptr() if out_cache_loc is not None else 0
    values[10] = mrope.data_ptr() if mrope is not None else 0
    values[11] = bonus.dtype is torch.int64
    values[12] = bonus.stride(0)
    values[13] = draft.dtype is torch.int64
    values[14] = draft.stride(0)
    values[15] = draft.stride(1)
    values[16] = seq.dtype is torch.int64
    values[17] = seq.stride(0)
    values[18] = tokens.dtype is torch.int64
    values[19] = req is not None and req.dtype is torch.int64
    values[20] = req.stride(0) if req is not None else 0
    values[21] = table is not None and table.dtype is torch.int64
    values[22] = table.stride(0) if table is not None else 0
    values[23] = table.stride(1) if table is not None else 0
    values[24] = batch_size
    values[25] = width
    values[26] = (table is not None) | ((mrope is not None) << 1)
    _module().chain_tree(seq, descriptor)
    return True


def extend_layout(
    *, seq, width, prefix, extend, post, positions, starts, mrope
) -> bool:
    if not _is_cuda or not _runtime_optimization_enabled():
        return False
    required = (seq, prefix, extend, post, positions, starts)
    if not all(isinstance(tensor, torch.Tensor) for tensor in required) or not (
        mrope is None or isinstance(mrope, torch.Tensor)
    ):
        return False
    if not _same_cuda_device(seq, *required, mrope) or not (
        isinstance(width, int) and width > 0 and seq.ndim == 1
    ):
        return False
    batch_size = seq.numel()
    num_tokens = batch_size * width
    if not (
        seq.dtype in _INDEX_DTYPES
        and prefix.dtype is torch.int32
        and prefix.is_contiguous()
        and prefix.shape == (batch_size,)
        and extend.dtype is torch.int32
        and extend.is_contiguous()
        and extend.shape == (batch_size,)
        and post.dtype is seq.dtype
        and post.is_contiguous()
        and post.shape == (batch_size,)
        and positions.dtype is torch.int64
        and positions.is_contiguous()
        and positions.shape == (num_tokens,)
        and starts.dtype is torch.int32
        and starts.is_contiguous()
        and starts.shape == (batch_size,)
        and (
            mrope is None
            or (
                mrope.dtype is torch.int64
                and mrope.is_contiguous()
                and mrope.shape == (3, num_tokens)
            )
        )
    ):
        return False
    descriptor, values = _descriptor("extend_layout", 13)
    values[0] = seq.data_ptr()
    values[1] = prefix.data_ptr()
    values[2] = extend.data_ptr()
    values[3] = post.data_ptr()
    values[4] = positions.data_ptr()
    values[5] = starts.data_ptr()
    values[6] = mrope.data_ptr() if mrope is not None else 0
    values[7] = seq.dtype is torch.int64
    values[8] = seq.stride(0)
    values[9] = post.dtype is torch.int64
    values[10] = batch_size
    values[11] = width
    values[12] = mrope is not None
    _module().extend_layout(seq, descriptor)
    return True


def draft_locs(
    *,
    req,
    table,
    seq,
    out_cache_loc,
    pool_len,
    topk,
    steps,
    batch_size,
    positions,
    mrope,
) -> bool:
    if not _is_cuda or not _runtime_optimization_enabled():
        return False
    required = (req, table, seq, out_cache_loc)
    optional = (positions, mrope)
    if not all(isinstance(tensor, torch.Tensor) for tensor in required) or not all(
        tensor is None or isinstance(tensor, torch.Tensor) for tensor in optional
    ):
        return False
    if not _same_cuda_device(seq, *required, *optional) or not (
        isinstance(pool_len, int)
        and isinstance(topk, int)
        and isinstance(steps, int)
        and isinstance(batch_size, int)
        and pool_len > 0
        and topk > 0
        and steps > 0
        and batch_size >= 0
    ):
        return False
    num_positions = batch_size * topk
    num_cache_locs = num_positions * steps
    if not (
        req.dtype in _INDEX_DTYPES
        and req.is_contiguous()
        and req.shape == (batch_size,)
        and table.dtype in _INDEX_DTYPES
        and table.ndim == 2
        and table.is_contiguous()
        and table.shape[1] == pool_len
        and seq.dtype in _INDEX_DTYPES
        and seq.is_contiguous()
        and seq.shape == (batch_size,)
        and out_cache_loc.dtype is torch.int64
        and out_cache_loc.is_contiguous()
        and out_cache_loc.shape == (num_cache_locs,)
        and (
            positions is None
            or (
                positions.dtype in _INDEX_DTYPES
                and positions.is_contiguous()
                and positions.shape == (num_positions,)
            )
        )
        and (
            mrope is None
            or (
                positions is not None
                and mrope.dtype in _INDEX_DTYPES
                and mrope.is_contiguous()
                and mrope.shape == (3, num_positions)
            )
        )
    ):
        return False
    descriptor, values = _descriptor("draft_locs", 17)
    values[0] = req.data_ptr()
    values[1] = table.data_ptr()
    values[2] = seq.data_ptr()
    values[3] = out_cache_loc.data_ptr()
    values[4] = positions.data_ptr() if positions is not None else 0
    values[5] = mrope.data_ptr() if mrope is not None else 0
    values[6] = req.dtype is torch.int64
    values[7] = table.dtype is torch.int64
    values[8] = seq.dtype is torch.int64
    values[9] = positions is not None and positions.dtype is torch.int64
    values[10] = mrope is not None and mrope.dtype is torch.int64
    values[11] = pool_len
    values[12] = topk
    values[13] = steps
    values[14] = batch_size
    values[15] = positions is not None
    values[16] = mrope is not None
    _module().draft_locs(seq, descriptor)
    return True
