"""MXFP4 index keys of the Kimi-K3.1 DSA indexer on gfx950, in AITER's FP4 MQA-logits pages.

A 64-key page holds the e2m1 payload as [4 chunks, 64 keys, 16 bytes] (4096 B) and the
ue8m0 exponents as [4 chunks, 64 keys] with the key axis a transposed 16 x 4 tile (256 B);
chunk c covers head dims [32c, 32c + 32). Decode and target verify score the pages with
AITER's row-group kernel, a request's rows off one read of each key; prefill gathers the
keys of every DCP rank and scores them, staged back into pages, with its prefill kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    _fp4_e2m1_code_rne,
    quantize_fp4_indexer_row,
)

PAGE_KEYS = 64
HEAD_DIM = 128
PAYLOAD_PAGE_BYTES = PAGE_KEYS * HEAD_DIM // 2
SCALE_PAGE_BYTES = PAGE_KEYS * HEAD_DIM // 32
# a gathered key: 64 payload bytes, then its 4 exponents in chunk order
ROW_BYTES = HEAD_DIM // 2 + HEAD_DIM // 32
_Q_HEADS_PER_PROGRAM = 16


@dataclass(frozen=True)
class Fp4IndexQuery:
    """MXFP4 indexer queries; they slice by row like the FP8 query tensor they replace."""

    payload: torch.Tensor  # [T, H, 64] e2m1
    # [T, 1, 4, 16, 4] ue8m0, head h of chunk c at [t, 0, c, h % 16, h // 16]
    scale: torch.Tensor

    @property
    def shape(self) -> torch.Size:
        return self.payload.shape

    @property
    def device(self) -> torch.device:
        return self.payload.device

    def __getitem__(self, rows: slice) -> Fp4IndexQuery:
        return Fp4IndexQuery(self.payload[rows], self.scale[rows])


class Fp4IndexKeys(NamedTuple):
    """Gathered FP4 keys staged into pages in order, the first num_keys of them real."""

    payload: torch.Tensor  # [P, 4096]
    scale: torch.Tensor  # [P, 256]
    num_keys: int


@triton.jit
def _q_fp4_head_gate_kernel(
    Q,
    G,
    QP,
    QS,
    W,
    HEADS: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    GATE_SCALE: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    HB: tl.constexpr,
):
    # HB of a token's heads, as quantize_fp4_indexer_row quantizes each of them; the
    # scale tile spans 64 heads and AITER reads the unused slots, which get 0
    t = tl.program_id(0)
    h = tl.program_id(1) * HB + tl.arange(0, HB)
    live = h < HEADS
    values = tl.load(
        Q + (t * HEADS + h)[:, None] * 128 + tl.arange(0, 128)[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)
    groups = tl.reshape(values, (HB, 4, 32))
    amax = tl.max(tl.abs(groups), axis=2)
    bits = tl.maximum(amax / 6.0, 1.0e-4).to(tl.int32, bitcast=True)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    scaled = groups / (exp << 23).to(tl.float32, bitcast=True)[:, :, None]
    lo, hi = tl.split(tl.reshape(scaled, (HB, 4, 16, 2)))
    packed = (_fp4_e2m1_code_rne(lo) & 0x0F) | ((_fp4_e2m1_code_rne(hi) & 0x0F) << 4)
    tl.store(
        QP + (t * HEADS + h)[:, None] * 64 + tl.arange(0, 64)[None, :],
        tl.reshape(packed, (HB, 64)),
        mask=live[:, None],
    )
    chunks = tl.arange(0, 4)
    tl.store(
        QS + t * 256 + chunks[None, :] * 64 + (h % 16 * 4 + h // 16)[:, None],
        tl.where(live[:, None], exp, 0).to(tl.uint8),
    )
    # the FP8 path's BF16 rounding point of the gate, without a query scale
    gate = tl.load(G + t * GATE_STRIDE + h, mask=live, other=0.0).to(tl.float32)
    gate = (gate * GATE_SCALE).to(tl.bfloat16).to(tl.float32)
    tl.store(
        W + t * HEADS + h, (gate * SOFTMAX_SCALE).to(W.dtype.element_ty), mask=live
    )


def indexer_q_fp4_head_gate(
    query: torch.Tensor,
    gate: torch.Tensor,
    softmax_scale: float,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[Fp4IndexQuery, torch.Tensor]:
    """MXFP4 of the no-RoPE indexer query and its weights
    float(bf16(gate * heads**-0.5)) * softmax_scale [T, H, 1], as indexer_q_quant_head_gate returns
    them but without a query scale (the kernels apply the group scales). The prefill kernel reads
    bf16 weights: pass softmax_scale 1 and apply it as its weight_scale."""
    assert query.is_cuda and query.dtype == torch.bfloat16
    assert query.ndim == 3 and query.shape[-1] == HEAD_DIM and query.is_contiguous()
    tokens, heads, _ = query.shape
    assert heads % 16 == 0 and heads <= 64, (
        f"{heads} heads: AITER takes 16-head groups up to 64"
    )
    assert gate.shape == (tokens, heads) and gate.stride(-1) == 1
    payload = torch.empty(
        (tokens, heads, HEAD_DIM // 2), dtype=torch.uint8, device=query.device
    )
    scale = torch.empty((tokens, 1, 4, 16, 4), dtype=torch.uint8, device=query.device)
    weights = torch.empty((tokens, heads, 1), dtype=weights_dtype, device=query.device)
    if tokens:
        hb = _Q_HEADS_PER_PROGRAM
        _q_fp4_head_gate_kernel[(tokens, 64 // hb)](
            query,
            gate,
            payload,
            scale,
            weights,
            heads,
            gate.stride(0),
            heads**-0.5,
            softmax_scale,
            hb,
        )
    return Fp4IndexQuery(payload, scale), weights


@triton.jit
def _store_fp4_index_k_kernel(K, LOC, PAYLOAD, SCALE, K_STRIDE: tl.constexpr):
    t = tl.program_id(0)
    loc = tl.load(LOC + t).to(tl.int64)
    if loc < 0:
        return
    values = tl.load(K + t * K_STRIDE + tl.arange(0, 128)).to(tl.float32)
    v0, v1 = tl.split(tl.reshape(values, (64, 2)))
    packed, packed_sf = quantize_fp4_indexer_row(
        values, v0, v1, BLOCK_N=128, GROUP_N=32, RNE=True
    )
    page, slot = loc // 64, loc % 64
    byte = tl.arange(0, 64)
    tl.store(PAYLOAD + page * 4096 + byte // 16 * 1024 + slot * 16 + byte % 16, packed)
    chunks = tl.arange(0, 4)
    tl.store(
        SCALE + page * 256 + chunks * 64 + slot % 16 * 4 + slot // 16,
        ((packed_sf >> (8 * chunks)) & 0xFF).to(tl.uint8),
    )


def store_fp4_index_k(
    key: torch.Tensor, payload: torch.Tensor, scale: torch.Tensor, loc: torch.Tensor
) -> None:
    """Quantize [T, 128] keys into their pages at row ``loc``; a negative loc skips the key."""
    assert key.shape[-1] == HEAD_DIM and key.stride(-1) == 1
    assert payload.dtype == scale.dtype == torch.uint8
    assert payload.shape[1:] == (PAYLOAD_PAGE_BYTES,) and scale.shape[1:] == (
        SCALE_PAGE_BYTES,
    )
    tokens = min(key.shape[0], loc.shape[0])
    if tokens:
        _store_fp4_index_k_kernel[(tokens,)](key, loc, payload, scale, key.stride(0))


@triton.jit
def _gather_fp4_rows_kernel(
    PAYLOAD, SCALE, TABLE, LENS, CU, OUT, TABLE_STRIDE, BLOCK: tl.constexpr
):
    b = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < tl.load(LENS + b)
    page = tl.load(TABLE + b * TABLE_STRIDE + rows // 64, valid, 0).to(tl.int64)
    slot = rows % 64
    dst = (tl.load(CU + b) + rows).to(tl.int64) * 68
    byte = tl.arange(0, 64)
    keys = tl.load(
        PAYLOAD
        + page[:, None] * 4096
        + byte[None, :] // 16 * 1024
        + slot[:, None] * 16
        + byte[None, :] % 16,
        valid[:, None],
    )
    tl.store(OUT + dst[:, None] + byte[None, :], keys, valid[:, None])
    chunks = tl.arange(0, 4)
    exps = tl.load(
        SCALE
        + page[:, None] * 256
        + chunks[None, :] * 64
        + slot[:, None] % 16 * 4
        + slot[:, None] // 16,
        valid[:, None],
    )
    tl.store(OUT + dst[:, None] + 64 + chunks[None, :], exps, valid[:, None])


def gather_fp4_index_k_rows(
    payload: torch.Tensor,
    scale: torch.Tensor,
    page_table: torch.Tensor,
    lens: torch.Tensor,
    total: int,
    max_len: int,
) -> torch.Tensor:
    """Each request's first ``lens`` keys from its pages, back to back as [total, 68] bytes."""
    out = torch.empty((total, ROW_BYTES), dtype=torch.uint8, device=payload.device)
    if total:
        cu = torch.cumsum(lens, 0, dtype=torch.int64) - lens
        block = 32
        _gather_fp4_rows_kernel[(lens.shape[0], triton.cdiv(max_len, block))](
            payload, scale, page_table, lens, cu, out, page_table.stride(0), block
        )
    return out


@lru_cache(maxsize=None)
def _rowgroup_plan(
    num_seqs: int, max_qlen: int, num_rows: int, heads: int, max_seq_len: int
):
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup import (
        make_fp4_mqa_plan,
    )

    return make_fp4_mqa_plan(
        num_seqs=num_seqs,
        max_qlen=max_qlen,
        num_rows=num_rows,
        heads=heads,
        page_size=PAGE_KEYS,
        max_seq_len=max_seq_len,
    )


def fp4_paged_mqa_logits(
    query: Fp4IndexQuery,
    weights: torch.Tensor,
    payload: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    row_ends: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """[rows, max_seq_len] logits of each request's ``query_start_loc`` rows over its
    ``block_tables`` pages; row r's columns past row_ends[r] are left unwritten."""
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup import (
        flydsl_pa_mqa_logits_fp4_rowgroup,
    )

    rows, heads = query.shape[:2]
    num_seqs = block_tables.shape[0]
    plan = _rowgroup_plan(
        num_seqs, max(1, rows // max(num_seqs, 1)), rows, heads, max_seq_len
    )
    logits = torch.empty(
        (rows, max_seq_len), dtype=torch.float32, device=payload.device
    )
    return flydsl_pa_mqa_logits_fp4_rowgroup(
        plan,
        query.payload,
        query.scale,
        payload.view(-1, 1, 4, PAGE_KEYS, 16),
        scale.view(-1, 1, 4, PAGE_KEYS),
        block_tables,
        weights,
        query_start_loc,
        row_ends,
        out=logits,
    )


@triton.jit
def _stage_fp4_rows_kernel(ROWS, PAYLOAD, SCALE, N, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = r < N
    src = r.to(tl.int64)[:, None] * 68
    page, slot = (r // 64).to(tl.int64)[:, None], (r % 64)[:, None]
    byte = tl.arange(0, 64)[None, :]
    keys = tl.load(ROWS + src + byte, valid[:, None])
    tl.store(
        PAYLOAD + page * 4096 + byte // 16 * 1024 + slot * 16 + byte % 16,
        keys,
        valid[:, None],
    )
    chunks = tl.arange(0, 4)[None, :]
    exps = tl.load(ROWS + src + 64 + chunks, valid[:, None])
    tl.store(
        SCALE + page * 256 + chunks * 64 + slot % 16 * 4 + slot // 16,
        exps,
        valid[:, None],
    )


def stage_fp4_index_k_rows(rows: torch.Tensor) -> Fp4IndexKeys:
    """[N, 68] keys -> the pages they fill in order."""
    n = rows.shape[0]
    pages = max(triton.cdiv(n, PAGE_KEYS), 1)
    payload = torch.empty(
        (pages, PAYLOAD_PAGE_BYTES), dtype=torch.uint8, device=rows.device
    )
    scale = torch.empty(
        (pages, SCALE_PAGE_BYTES), dtype=torch.uint8, device=rows.device
    )
    if n:
        block = 32
        _stage_fp4_rows_kernel[(triton.cdiv(n, block),)](rows, payload, scale, n, block)
    return Fp4IndexKeys(payload, scale, n)


def fp4_ragged_mqa_logits(
    query: Fp4IndexQuery,
    weights: torch.Tensor,
    weight_scale: float,
    keys: Fp4IndexKeys,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """Prefill logits [rows, num_keys] of each row over keys [ks, ke), as fp8_mqa_logits
    lays them out; AITER writes only each row's window. weights: bf16 [rows, H], scaled
    by weight_scale in the kernel."""
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
        flydsl_pa_mqa_logits_fp4_prefill,
    )

    rows = query.shape[0]
    pages = keys.payload.shape[0]
    device = keys.payload.device
    # AITER compiles the scorer per table width and its schedule per search depth and
    # bound alignment: a power-of-two table (one sequence over every staged page, plus
    # the block the scorer prefetches past the last) and fresh bounds keep it to one
    # variant per doubling of the context
    width = max(1024, 1 << (pages + 4 - 1).bit_length())
    table = torch.zeros((1, width), dtype=torch.int32, device=device)
    table[0, :pages] = torch.arange(pages, dtype=torch.int32, device=device)
    ks, ke = ks.clone(), ke.clone()
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device=device)
    # a precomputed schedule spares the kernel its -inf fill of every column
    _, cta_info, n_ctas = compute_prefill_schedule(
        row_to_batch, ks, ke, 256, max(4096, rows), width * PAGE_KEYS
    )
    logits = torch.empty((rows, keys.num_keys), dtype=torch.float32, device=device)
    return flydsl_pa_mqa_logits_fp4_prefill(
        query.payload,
        query.scale,
        keys.payload.view(pages, 1, 4, PAGE_KEYS, 16),
        keys.scale.view(pages, 1, 4, PAGE_KEYS),
        table,
        weights,
        row_to_batch,
        ks,
        ke,
        keys.num_keys,
        weight_scale=weight_scale,
        cta_info=cta_info,
        n_ctas=n_ctas,
        out=logits,
    )
