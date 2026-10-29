"""SGLANG_PERFLL_HOST_JIT_INDEX: the spec-v2 decode round's index kernels (draft cache
locations, chain-tree verify inputs, draft-extend layout) as one JIT CUDA module launched
from a cached host descriptor, with the Triton kernels' integer results. Each entry point
returns False without launching when an index tensor is not int32 / int64; the caller
then launches the Triton kernel."""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

ENABLED = envs.SGLANG_PERFLL_HOST_JIT_INDEX.get()
_INT = (torch.int32, torch.int64)
_desc: dict = {}


@cache_once
def _module():
    return load_jit(
        "perfll_spec_index",
        cuda_files=["elementwise/perfll_spec_index.cuh"],
        cuda_wrappers=[
            ("chain_tree", "PerfllSpecIndex::chain_tree"),
            ("extend_layout", "PerfllSpecIndex::extend_layout"),
            ("draft_locs", "PerfllSpecIndex::draft_locs"),
        ],
    )


def _view(name: str, n: int):
    # one host descriptor per entry point; C++ reads it before the launch returns
    entry = _desc.get(name)
    if entry is None:
        tensor = torch.zeros((n,), dtype=torch.int64)
        entry = _desc[name] = (tensor, tensor.numpy())
    return entry


def chain_tree(
    bonus, draft, seq, mask, positions, retrieve, tokens, req, table, out_cache_loc, mrope
) -> bool:
    if not (
        bonus.dtype in _INT
        and draft.dtype in _INT
        and seq.dtype in _INT
        and tokens.dtype in _INT
        and (req is None or req.dtype in _INT)
        and (table is None or table.dtype in _INT)
    ):
        return False
    bs, depth = draft.shape
    tensor, d = _view("chain_tree", 27)
    d[0] = bonus.data_ptr()
    d[1] = draft.data_ptr()
    d[2] = seq.data_ptr()
    d[3] = mask.data_ptr()
    d[4] = positions.data_ptr()
    d[5] = retrieve.data_ptr()
    d[6] = tokens.data_ptr()
    d[7] = req.data_ptr() if req is not None else 0
    d[8] = table.data_ptr() if table is not None else 0
    d[9] = out_cache_loc.data_ptr() if out_cache_loc is not None else 0
    d[10] = mrope.data_ptr() if mrope is not None else 0
    d[11] = bonus.dtype is torch.int64
    d[12] = bonus.stride(0)
    d[13] = draft.dtype is torch.int64
    d[14] = draft.stride(0)
    d[15] = draft.stride(1)
    d[16] = seq.dtype is torch.int64
    d[17] = seq.stride(0)
    d[18] = tokens.dtype is torch.int64
    d[19] = req is not None and req.dtype is torch.int64
    d[20] = req.stride(0) if req is not None else 0
    d[21] = table is not None and table.dtype is torch.int64
    d[22] = table.stride(0) if table is not None else 0
    d[23] = table.stride(1) if table is not None else 0
    d[24] = bs
    d[25] = depth + 1
    d[26] = (table is not None) | ((mrope is not None) << 1)
    _module().chain_tree(seq, tensor)
    return True


def extend_layout(seq, width, prefix, extend, post, positions, starts, mrope) -> bool:
    if seq.dtype not in _INT:
        return False
    tensor, d = _view("extend_layout", 13)
    d[0] = seq.data_ptr()
    d[1] = prefix.data_ptr()
    d[2] = extend.data_ptr()
    d[3] = post.data_ptr()
    d[4] = positions.data_ptr()
    d[5] = starts.data_ptr()
    d[6] = mrope.data_ptr() if mrope is not None else 0
    d[7] = seq.dtype is torch.int64
    d[8] = seq.stride(0)
    d[9] = post.dtype is torch.int64
    d[10] = seq.numel()
    d[11] = width
    d[12] = mrope is not None
    _module().extend_layout(seq, tensor)
    return True


def draft_locs(req, table, seq, out_cache_loc, pool_len, topk, steps, bs, positions, mrope) -> bool:
    if not (
        req.dtype in _INT
        and table.dtype in _INT
        and seq.dtype in _INT
        and out_cache_loc.dtype is torch.int64
        and (positions is None or positions.dtype in _INT)
        and (mrope is None or mrope.dtype in _INT)
        and req.is_contiguous()
        and seq.is_contiguous()
    ):
        return False
    tensor, d = _view("draft_locs", 17)
    d[0] = req.data_ptr()
    d[1] = table.data_ptr()
    d[2] = seq.data_ptr()
    d[3] = out_cache_loc.data_ptr()
    d[4] = positions.data_ptr() if positions is not None else 0
    d[5] = mrope.data_ptr() if mrope is not None else 0
    d[6] = req.dtype is torch.int64
    d[7] = table.dtype is torch.int64
    d[8] = seq.dtype is torch.int64
    d[9] = positions is not None and positions.dtype is torch.int64
    d[10] = mrope is not None and mrope.dtype is torch.int64
    d[11] = pool_len
    d[12] = topk
    d[13] = steps
    d[14] = bs
    d[15] = positions is not None
    d[16] = mrope is not None
    _module().draft_locs(seq, tensor)
    return True
