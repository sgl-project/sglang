"""Optional Flash-V100 grouped Page4 prefill for Qwen's structured QSA indices.

Uses the grouped planner/attention API from 1Cat-vLLM's Flash-V100 fork.
The caller must supply the allocator page size and compressed Page4 selections;
arbitrary token permutations or duplicate selections are not this API's input.

Gated by ``SGLANG_SM70_QSA_GROUPED_PREFILL``; a missing library, another shape
or the flag being unset keeps the native QSA prefill.
"""

from __future__ import annotations

import importlib.util
import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.kernels.sm70_paths import sm70_prebuilt
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_WORKSPACES = {}

_LIBRARY_NAME = "_sm70_grouped_qsa_v100.so"


@lru_cache(maxsize=1)
def load_grouped_qsa_ops():
    path = sm70_prebuilt(_LIBRARY_NAME)
    if not path.is_file():
        logger.warning(
            "Grouped QSA library is missing: %s; retaining native QSA.", path
        )
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "_sglang_grouped_qsa.flash_attn_v100_cuda", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not all(
            hasattr(module, name)
            for name in ["grouped_sparse_page4_plan_fwd", "grouped_sparse_page4_fwd"]
        ):
            raise RuntimeError("Flash-V100 library lacks grouped Page4 bindings")
        return module
    except Exception:
        logger.exception(
            "Grouped QSA library could not be loaded; retaining native QSA."
        )
        return None


def can_use_grouped_qsa(q, k, v, indices, page_size):
    return (
        envs.SGLANG_SM70_QSA_GROUPED_PREFILL.get()
        and page_size == 16
        and q.is_cuda
        and q.dtype == torch.float16
        and tuple(q.shape) == (2048, 6, 256)
        and tuple(indices.shape) == (2048, 2051)
        and q.is_contiguous()
        and indices.is_contiguous()
        and indices.dtype == torch.int32
        and k.dtype == v.dtype == torch.float16
        and k.device == v.device == indices.device == q.device
        and k.is_contiguous()
        and v.is_contiguous()
        and k.shape == v.shape
        and tuple(k.shape[1:]) == (1, 256)
        and k.shape[0] % page_size == 0
        and torch.cuda.get_device_capability(q.device) == (7, 0)
    )


@triton.jit
def _prepare_metadata(
    table,
    req,
    lens,
    blocks,
    positions,
    token_to_req,
    STRIDE: tl.constexpr,
    PAGES: tl.constexpr,
    ROWS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(req).to(tl.int64)
    slot = tl.load(
        table + request * STRIDE + offset * PAGE_SIZE, offset < PAGES, other=0
    )
    tl.store(blocks + offset, slot // PAGE_SIZE, offset < PAGES)
    position = tl.load(lens).to(tl.int64) - ROWS + offset
    tl.store(positions + offset, position, offset < ROWS)
    tl.store(token_to_req + offset, 0, offset < ROWS)


@triton.jit
def _mask_future(
    indices, lens, out, ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // WIDTH
    column = offset % WIDTH
    visible = tl.load(lens) - ROWS + row + 1
    token = tl.load(indices + offset, offset < ROWS * WIDTH, other=-1)
    valid = (
        (column < tl.minimum(WIDTH, tl.maximum(visible, 0)))
        & (token >= 0)
        & (token < visible)
    )
    tl.store(out + offset, tl.where(valid, token, -1), offset < ROWS * WIDTH)


def grouped_qsa_prefill(q, k, v, table, req, indices, lens, scale, page_size=16):
    """Read a paged FP16 cache using up to 512 full Page4 groups plus their tail."""
    ops = load_grouped_qsa_ops()
    if ops is None:
        return None
    rows = q.shape[0]
    assert rows > 0 and rows % 8 == 0 and indices.shape == (rows, 2051)
    assert page_size == 16 and k.shape[0] % page_size == 0
    assert q.is_contiguous() and indices.is_contiguous()
    assert req.numel() == lens.numel() == 1
    assert all(t.device == q.device for t in (k, v, table, req, indices, lens))
    groups = rows // 8
    pages = triton.cdiv(table.shape[1], page_size)
    key = (q.device.index, torch.cuda.current_stream(q.device).cuda_stream, rows, pages)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        workspace = dict(
            blocks=torch.empty((1, pages), dtype=torch.int32, device=q.device),
            positions=torch.empty(rows, dtype=torch.int64, device=q.device),
            token_to_req=torch.empty(rows, dtype=torch.int32, device=q.device),
            selected=torch.empty_like(indices, dtype=torch.int32),
            grouped_pages=torch.empty(
                (groups, 4160), dtype=torch.int32, device=q.device
            ),
            masks=torch.empty((groups, 4160), dtype=torch.uint32, device=q.device),
            lengths=torch.empty(groups, dtype=torch.int32, device=q.device),
            lse=torch.empty((rows, 6), dtype=torch.float32, device=q.device),
        )
        _WORKSPACES[key] = workspace
    _prepare_metadata[(triton.cdiv(max(rows, pages), 256),)](
        table,
        req,
        lens,
        workspace["blocks"],
        workspace["positions"],
        workspace["token_to_req"],
        table.stride(0),
        pages,
        rows,
        page_size,
        256,
    )
    _mask_future[(triton.cdiv(rows * 2051, 256),)](
        indices, lens, workspace["selected"], rows, 2051, 256
    )
    sequence_lengths = lens.to(dtype=torch.int32).contiguous()
    ops.grouped_sparse_page4_plan_fwd(
        workspace["selected"],
        workspace["blocks"],
        workspace["token_to_req"],
        workspace["positions"],
        sequence_lengths,
        workspace["grouped_pages"],
        workspace["masks"],
        workspace["lengths"],
        page_size,
        page_size // 4,
        k.shape[0] // page_size,
    )
    output = torch.empty_like(q)
    ops.grouped_sparse_page4_fwd(
        q,
        k.view(-1, 4, 1, 256),
        v.view(-1, 4, 1, 256),
        output,
        workspace["grouped_pages"],
        workspace["masks"],
        workspace["lengths"],
        workspace["lse"],
        float(scale),
    )
    logger.info_once(
        "SM70 QSA grouped Page4 prefill active: eight queries share selected KV pages."
    )
    return output
