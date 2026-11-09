"""Metal kernels for the DASH attention path on Apple MPS.

MSL sources live in ``dash.metal`` next to this module and are compiled at
runtime with ``torch.mps.compile_shader``: by ``compile_kernels`` when the
attention backend is constructed, or on first use. They run on torch MPS
tensors and are not part of the AOT ``sgl_kernel`` Metal build. Each wrapper
returns False when a shape is outside what the kernel was written for, and the
caller falls back to the torch implementation in ``mps.py``.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_SG = 4  # simdgroups per attend threadgroup
_ROUTE_THREADS = 256
# Block-table columns a route threadgroup keeps in threadgroup memory.
_ROUTE_MAX_W = 4096
_METAL_PATH = Path(__file__).with_name("dash.metal")


@functools.cache
def _metal_src() -> str:
    return _METAL_PATH.read_text(encoding="utf-8")


@functools.cache
def _attend_lib(head_dim: int, block_size: int):
    defs = (
        f"#define KERNEL_ATTEND 1\n#define D {head_dim}\n"
        f"#define PER {head_dim // 32}\n#define BS {block_size}\n#define SG {_SG}\n"
    )
    return torch.mps.compile_shader(defs + _metal_src())


@functools.cache
def _route_lib(index_dim: int, block_size: int):
    nt = _ROUTE_THREADS
    defs = (
        f"#define KERNEL_ROUTE 1\n#define DQ {index_dim}\n#define BS {block_size}\n"
        f"#define NT {nt}\n#define NSG {nt // 32}\n#define MAXW {_ROUTE_MAX_W}\n"
    )
    return torch.mps.compile_shader(defs + _metal_src())


@functools.cache
def _kbar_lib(index_dim: int, block_size: int):
    defs = (
        f"#define KERNEL_KBAR 1\n#define DQ {index_dim}\n#define BS {block_size}\n"
        f"#define TPL {(block_size + 31) // 32}\n"
    )
    return torch.mps.compile_shader(defs + _metal_src())


@functools.cache
def _rope_lib(head_dim: int):
    defs = (
        f"#define KERNEL_ROPE 1\n#define HD {head_dim}\n#define HALF {head_dim // 2}\n"
    )
    return torch.mps.compile_shader(defs + _metal_src())


@functools.cache
def _decode_meta_lib(block_size: int):
    nt = 256
    defs = (
        f"#define KERNEL_DECODE_META 1\n#define BS {block_size}\n"
        f"#define NT {nt}\n#define NSG {nt // 32}\n"
    )
    return torch.mps.compile_shader(defs + _metal_src())


def compile_kernels(*, head_dim: int, index_dim: int, block_size: int) -> None:
    """Precompile all DASH Metal shader libraries for the model's geometry."""
    _attend_lib(head_dim, block_size)
    _route_lib(index_dim, block_size)
    _kbar_lib(index_dim, block_size)
    _rope_lib(head_dim)
    if index_dim != head_dim:
        _rope_lib(index_dim)
    _decode_meta_lib(block_size)


def _row_major(t: torch.Tensor) -> bool:
    return t.dim() == 2 and t.stride(1) == 1


def _int32_table(t: torch.Tensor) -> torch.Tensor:
    """``t`` as the row-major int32 table the metadata kernels read.

    Wider tables are narrowed after a range check rather than reinterpreted.
    """
    if t.dtype == torch.int32 and _row_major(t):
        return t
    if t.dtype != torch.int32 and t.numel():
        lo, hi = (v.item() for v in torch.aminmax(t))
        info = torch.iinfo(torch.int32)
        if lo < info.min or hi > info.max:
            raise ValueError(f"block table values [{lo}, {hi}] do not fit in int32")
    return t.to(torch.int32).contiguous()


def _prefill_rows(rows, like):
    """``(row2req, cu_seqlens_q, context_lens)`` for the kernels, or dummies."""
    if rows is None:
        return like, like, like, 0
    row2req, cu, ctx = rows
    return row2req.to(torch.int32), cu.to(torch.int64), ctx.to(torch.int64), 1


def sparse_attend(
    q,
    k_cache,
    v_cache,
    idx,
    bias,
    block_table,
    seq_lens,
    out,
    *,
    num_kv_heads,
    scale,
    local_blocks,
    block_size,
    decode_query_len=1,
    rows=None,
) -> bool:
    """Run the Metal sparse attend; False when the inputs need the torch path.

    ``idx`` / ``bias`` are the [nq, T] slot rows shared by every kv head.
    Decode rows are ``decode_query_len`` per request; prefill passes ``rows``.
    """
    nq, num_heads, head_dim = q.shape
    if (
        head_dim % 32
        or head_dim > 256
        or q.dtype != torch.bfloat16
        or k_cache.dtype != torch.bfloat16
        or v_cache.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or idx.dtype != torch.int32
        or (bias is not None and bias.dtype != torch.float32)
        or block_table.dtype != torch.int32
        or not (q.is_contiguous() and out.is_contiguous())
        or not (k_cache.is_contiguous() and v_cache.is_contiguous())
        or not _row_major(idx)
        or not _row_major(block_table)
        or (bias is not None and not _row_major(bias))
    ):
        return False
    if nq == 0:
        return True
    lib = _attend_lib(head_dim, block_size)
    seq = seq_lens.to(torch.int64)
    bias_t = bias if bias is not None else idx.new_zeros(1, dtype=torch.float32)
    lib.dash_sparse_attend(
        q,
        k_cache,
        v_cache,
        idx,
        bias_t,
        block_table,
        seq,
        out,
        num_heads,
        num_kv_heads,
        idx.shape[1],
        idx.stride(0),
        bias_t.stride(0) if bias is not None else 0,
        block_table.stride(0),
        decode_query_len,
        local_blocks,
        int(bias is not None),
        float(scale),
        *_prefill_rows(rows, seq),
        threads=nq * num_heads * _SG * 32,
        group_size=_SG * 32,
    )
    return True


def route(
    idx_q,
    k_bar,
    block_table,
    seq_lens,
    image_allow,
    out_idx,
    out_bias,
    *,
    width,
    block_size,
    local_blocks,
    idx_scale,
    prior_floor,
    sigma,
    budget,
    M,
    decode_query_len=1,
    rows=None,
) -> bool:
    """Run the Metal router; False when the inputs need the torch path.

    Decode rows are ``decode_query_len`` per request; prefill passes ``rows``.
    """
    nd, index_dim = idx_q.shape
    if (
        width > _ROUTE_MAX_W
        or M > _ROUTE_THREADS
        or idx_q.dtype != torch.bfloat16
        or not idx_q.is_contiguous()
        or k_bar.dtype != torch.float32
        or not k_bar.is_contiguous()
        or block_table.dtype != torch.int32
        or not _row_major(block_table)
        or out_idx.dtype != torch.int32
        or out_bias.dtype != torch.float32
        or not (_row_major(out_idx) and _row_major(out_bias))
        or (
            image_allow is not None
            and image_allow.dtype not in (torch.bool, torch.int8)
        )
        or (image_allow is not None and not _row_major(image_allow))
    ):
        return False
    if nd == 0:
        return True
    if image_allow is not None and image_allow.dtype == torch.bool:
        image_allow = image_allow.view(torch.int8)
    img = image_allow if image_allow is not None else out_idx
    seq = seq_lens.to(torch.int64)
    _route_lib(index_dim, block_size).dash_route(
        idx_q,
        k_bar,
        block_table,
        seq,
        img,
        out_idx,
        out_bias,
        width,
        block_table.stride(0),
        image_allow.stride(0) if image_allow is not None else 0,
        out_idx.stride(0),
        out_bias.stride(0),
        decode_query_len,
        local_blocks,
        int(image_allow is not None),
        budget,
        M,
        float(idx_scale),
        float(prior_floor),
        1.0 / float(sigma),
        *_prefill_rows(rows, seq),
        threads=nd * _ROUTE_THREADS,
        group_size=_ROUTE_THREADS,
    )
    return True


def kbar_refresh(kv_cache, k_bar, head_cls, pages, tok) -> bool:
    """Run the Metal page-summary refresh; False when the torch path is needed."""
    _, block_size, index_dim = kv_cache.shape
    if (
        kv_cache.dtype != torch.bfloat16
        or not kv_cache.is_contiguous()
        or k_bar.dtype != torch.float32
        or not k_bar.is_contiguous()
        or block_size > 128
    ):
        return False
    n = pages.numel()
    if n == 0:
        return True
    _kbar_lib(index_dim, block_size).dash_kbar_refresh(
        kv_cache,
        head_cls.to(torch.bfloat16).reshape(-1).contiguous(),
        pages.to(torch.int64).contiguous(),
        tok.to(torch.int32).contiguous(),
        k_bar,
        threads=n * 32,
        group_size=32,
    )
    return True


def page2d_rope_pair(x, y, cos, sin, head_dim: int):
    """Page2D NeoX RoPE of contiguous bf16 ``x`` and ``y`` in one launch."""
    t = x.shape[0]
    nx, ny = x.shape[1] // head_dim, y.shape[1] // head_dim
    ox, oy = torch.empty_like(x), torch.empty_like(y)
    total = t * (nx + ny) * (head_dim // 2)
    _rope_lib(head_dim).page2d_rope_pair(
        x, y, cos, sin, ox, oy, nx, ny, total, threads=total
    )
    return ox, oy


def decode_meta(
    *,
    input_ids,
    block_table,
    seq_lens,
    slots,
    coord_rows,
    slot_tables,
    coord_tables,
    q_page,
    tok,
    page,
    image_token_id,
    page_token_id,
    per_page,
    num_slots,
    max_pages,
    block_size,
) -> None:
    """Metal ``dash_decode_meta``; every state table is updated in place."""
    num_reqs = int(seq_lens.shape[0])
    block_table = _int32_table(block_table)
    c_mark, c_pp, c_cnt = coord_tables
    _decode_meta_lib(block_size).dash_decode_meta(
        input_ids.to(torch.int64),
        block_table,
        seq_lens.to(torch.int64),
        slots,
        coord_rows.to(torch.int64),
        *slot_tables,
        c_mark,
        c_pp,
        c_cnt,
        q_page,
        tok,
        page,
        num_reqs,
        block_table.stride(0),
        c_cnt.stride(0),
        image_token_id,
        page_token_id or 0,
        per_page,
        num_slots,
        max_pages,
        block_table.shape[1],
        int(page_token_id is not None),
        threads=num_reqs,
    )


def decode_allow(
    *,
    block_table,
    slot_img_page,
    q_page,
    seq_lens,
    out,
    width,
    num_blocks,
    max_block,
    block_size,
) -> None:
    """Metal ``dash_decode_allow`` into the ``[R, max_block]`` int8 view ``out``."""
    num_reqs = int(q_page.shape[0])
    block_table = _int32_table(block_table)
    _decode_meta_lib(block_size).dash_decode_allow(
        block_table,
        slot_img_page,
        q_page,
        seq_lens.to(torch.int64),
        out,
        block_table.stride(0),
        out.stride(0),
        width,
        num_blocks,
        max_block,
        threads=num_reqs * 256,
        group_size=256,
    )
