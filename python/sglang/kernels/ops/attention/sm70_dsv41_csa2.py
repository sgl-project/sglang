"""SM70 JIT loaders for DeepSeek-V4.1 CSA2 pack / indexer / sparse decode.

fp16 only. No PDL, TMA, wgmma, or DeepGEMM. Layouts match
``sm70_csa2_reference.py``: 288 B compressed KV, 68 B index K, 528 B SWA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

if TYPE_CHECKING:
    from tvm_ffi.module import Module

KV_ROW_BYTES = 288
INDEX_ROW_BYTES = 68
SWA_ROW_BYTES = 528
HEAD_DIM = 512
INDEX_DIM = 128
ROPE_DIM = 64


@cache_once
def _module() -> Module:
    cap = torch.cuda.get_device_capability()
    if cap[0] != 7:
        raise RuntimeError(
            f"sm70_dsv41 CSA2 kernels require SM70 (Volta); got SM{cap[0]}{cap[1]}"
        )
    return load_jit(
        "sm70_dsv41_csa2",
        cuda_files=[
            "sm70_dsv41_pack_kv_fp4.cuh",
            "sm70_dsv41_pack_index_k.cuh",
            "sm70_dsv41_index_logits.cuh",
            "sm70_dsv41_sparse_decode.cuh",
        ],
        cuda_wrappers=[
            ("pack_kv_fp4", "sm70_dsv41::pack_kv_fp4"),
            ("pack_kv_fp4_rope", "sm70_dsv41::pack_kv_fp4_rope"),
            ("pack_kv_fp4_at", "sm70_dsv41::pack_kv_fp4_at"),
            ("pack_swa_fp8", "sm70_dsv41::pack_swa_fp8"),
            ("pack_swa_fp8_rope", "sm70_dsv41::pack_swa_fp8_rope"),
            ("pack_swa_fp8_at", "sm70_dsv41::pack_swa_fp8_at"),
            ("pack_index_k", "sm70_dsv41::pack_index_k"),
            ("pack_index_k_rope", "sm70_dsv41::pack_index_k_rope"),
            ("pack_index_k_at", "sm70_dsv41::pack_index_k_at"),
            ("index_logits", "sm70_dsv41::index_logits"),
            ("index_logits_decode_dense", "sm70_dsv41::index_logits_decode_dense"),
            ("index_logits_decode_gather", "sm70_dsv41::index_logits_decode_gather"),
            ("sparse_decode", "sm70_dsv41::sparse_decode"),
            ("sparse_decode_indexed", "sm70_dsv41::sparse_decode_indexed"),
            ("sparse_prefill", "sm70_dsv41::sparse_prefill"),
        ],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
    )


def _contig(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _as_i64(positions: torch.Tensor) -> torch.Tensor:
    return positions if positions.dtype == torch.int64 else positions.to(torch.int64)


_FREQS_TABLE: dict[tuple, torch.Tensor] = {}


def freqs_cis_interleaved_table(freqs: torch.Tensor) -> torch.Tensor:
    """Cached fp32 [max_pos, rope_dim] interleaved real/imag of a complex table."""
    if not torch.is_complex(freqs):
        return _contig(freqs)
    key = (int(freqs.data_ptr()), tuple(freqs.shape), str(freqs.device))
    cached = _FREQS_TABLE.get(key)
    if cached is None:
        table = torch.view_as_real(freqs).reshape(freqs.shape[0], -1)
        cached = table if table.is_contiguous() else table.contiguous()
        _FREQS_TABLE[key] = cached
    return cached


def _freqs_interleaved(freqs: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(freqs):
        t = torch.view_as_real(freqs).reshape(freqs.shape[0], -1)
        return t if t.is_contiguous() else t.contiguous()
    return _contig(freqs)


def pack_kv_fp4(
    x: torch.Tensor,
    freqs: Optional[torch.Tensor] = None,
    rope_dim: int = ROPE_DIM,
) -> torch.Tensor:
    """fp16 [N, 512] -> uint8 [N, 288]. Optional RoPE is fp32 then round to fp16."""
    if x.shape[0] == 0:
        return torch.empty((0, KV_ROW_BYTES), dtype=torch.uint8, device=x.device)
    out = torch.empty((x.shape[0], KV_ROW_BYTES), dtype=torch.uint8, device=x.device)
    x = _contig(x)
    if freqs is None:
        _module().pack_kv_fp4(out, x)
    else:
        _module().pack_kv_fp4_rope(out, x, _freqs_interleaved(freqs), int(rope_dim))
    return out


def pack_index_k(
    x: torch.Tensor,
    freqs: Optional[torch.Tensor] = None,
    rope_dim: int = ROPE_DIM,
) -> torch.Tensor:
    """fp16 [N, 128] -> uint8 [N, 68]."""
    if x.shape[0] == 0:
        return torch.empty((0, INDEX_ROW_BYTES), dtype=torch.uint8, device=x.device)
    out = torch.empty((x.shape[0], INDEX_ROW_BYTES), dtype=torch.uint8, device=x.device)
    x = _contig(x)
    if freqs is None:
        _module().pack_index_k(out, x)
    else:
        _module().pack_index_k_rope(out, x, _freqs_interleaved(freqs), int(rope_dim))
    return out


def pack_swa_fp8(
    x: torch.Tensor,
    freqs: Optional[torch.Tensor] = None,
    rope_dim: int = ROPE_DIM,
) -> torch.Tensor:
    """fp16 [N, 512] -> uint8 [N, 528] (E4M3 + UE8M0 exponents)."""
    if x.shape[0] == 0:
        return torch.empty((0, SWA_ROW_BYTES), dtype=torch.uint8, device=x.device)
    out = torch.empty((x.shape[0], SWA_ROW_BYTES), dtype=torch.uint8, device=x.device)
    x = _contig(x)
    if freqs is None:
        _module().pack_swa_fp8(out, x)
    else:
        _module().pack_swa_fp8_rope(out, x, _freqs_interleaved(freqs), int(rope_dim))
    return out


def _pack_at(
    fn,
    dst: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor,
    freqs_table: Optional[torch.Tensor],
    rope_dim: int,
    row_div: int,
    freq_mul: int,
    dst_mod: int,
) -> None:
    if x.shape[0] == 0:
        return
    fn(
        dst,
        _contig(x),
        freqs_table,
        int(rope_dim),
        _as_i64(positions),
        int(row_div),
        int(freq_mul),
        int(dst_mod),
    )


def pack_kv_fp4_at(
    dst: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    freqs_table: Optional[torch.Tensor] = None,
    rope_dim: int = ROPE_DIM,
    row_div: int = 1,
    freq_mul: int = 1,
    dst_mod: int = 0,
) -> None:
    """Pack into ``dst[(pos // row_div) % dst_mod]``. ``dst_mod==0`` skips modulo."""
    _pack_at(
        _module().pack_kv_fp4_at,
        dst,
        x,
        positions,
        freqs_table,
        rope_dim,
        row_div,
        freq_mul,
        dst_mod,
    )


def pack_index_k_at(
    dst: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    freqs_table: Optional[torch.Tensor] = None,
    rope_dim: int = ROPE_DIM,
    row_div: int = 1,
    freq_mul: int = 1,
    dst_mod: int = 0,
) -> None:
    _pack_at(
        _module().pack_index_k_at,
        dst,
        x,
        positions,
        freqs_table,
        rope_dim,
        row_div,
        freq_mul,
        dst_mod,
    )


def pack_swa_fp8_at(
    dst: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    freqs_table: Optional[torch.Tensor] = None,
    rope_dim: int = ROPE_DIM,
    row_div: int = 1,
    freq_mul: int = 1,
    dst_mod: int = 0,
) -> None:
    _pack_at(
        _module().pack_swa_fp8_at,
        dst,
        x,
        positions,
        freqs_table,
        rope_dim,
        row_div,
        freq_mul,
        dst_mod,
    )


def index_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_rows: torch.Tensor,
    query_pos: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """q [T, H, 128] fp16, weights [T, H] fp16, rows [N, 68] -> logits [T, N] fp32."""
    t, n = q.shape[0], index_rows.shape[0]
    out = torch.empty((t, n), dtype=torch.float32, device=q.device)
    if t == 0 or n == 0:
        return out
    _module().index_logits(
        out,
        _contig(q),
        _contig(weights),
        _contig(index_rows),
        _contig(query_pos.to(torch.int32)),
        int(ratio),
    )
    return out


def index_logits_decode_dense(
    logits: torch.Tensor,
    q: torch.Tensor,
    weights: torch.Tensor,
    index_rows: torch.Tensor,
    query_pos: torch.Tensor,
    ratio: int,
    pad: int = 1,
) -> None:
    """Graph-safe T=1 indexer over a static-capacity buffer.

    ``logits`` [1, n_out] fp32 receives score(row j) for j < vis and -inf for
    vis <= j < roundup(vis, pad); vis = (query_pos[0]+1)//ratio is read on the
    device. Entries at or beyond roundup(vis, pad) are left untouched.
    """
    _module().index_logits_decode_dense(
        logits, q, weights, index_rows, query_pos, int(ratio), int(pad)
    )


def index_logits_decode_gather(
    logits: torch.Tensor,
    q: torch.Tensor,
    weights: torch.Tensor,
    index_rows: torch.Tensor,
    query_pos: torch.Tensor,
    key_ids: torch.Tensor,
    ratio: int,
) -> None:
    """Graph-safe T=1 indexer for an explicit key list ``key_ids`` [1, n_out] int32.

    ``logits[0, j]`` = score(row key_ids[j]), or -inf when key_ids[j] < 0 or
    key_ids[j] >= vis.
    """
    _module().index_logits_decode_gather(
        logits, q, weights, index_rows, query_pos, key_ids, int(ratio)
    )


def sparse_decode(
    q: torch.Tensor,
    swa_rows: torch.Tensor,
    kv_rows: torch.Tensor,
    swa_valid: torch.Tensor,
    kv_valid: torch.Tensor,
    sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """One-query sparse attn. q [H, 512] fp16; rows are packed 528/288-byte layouts."""
    if swa_rows.numel() == 0:
        swa_rows = q.new_empty((0, SWA_ROW_BYTES), dtype=torch.uint8)
        swa_valid = q.new_empty((0,), dtype=torch.uint8)
    if kv_rows.numel() == 0:
        kv_rows = q.new_empty((0, KV_ROW_BYTES), dtype=torch.uint8)
        kv_valid = q.new_empty((0,), dtype=torch.uint8)
    out = torch.empty_like(q)
    _module().sparse_decode(
        out,
        _contig(q),
        _contig(swa_rows),
        _contig(kv_rows),
        _contig(swa_valid.to(torch.uint8)),
        _contig(kv_valid.to(torch.uint8)),
        _contig(sink.float() if sink.dtype != torch.float32 else sink),
        float(softmax_scale),
    )
    return out


def sparse_prefill(
    q: torch.Tensor,
    swa_rows: torch.Tensor,
    kv_rows: torch.Tensor,
    swa_valid: torch.Tensor,
    kv_valid: torch.Tensor,
    sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """T-wide packed sparse. q [T, H, 512] fp16; rows [T, W, 528] / [T, K, 288]."""
    t = int(q.shape[0])
    if t == 0:
        return q
    # TensorMatcher rejects [T, 0, B] (nonzero stride on an empty dim).
    # One all-invalid dummy column is the same as zero keys.
    if swa_rows.numel() == 0 or int(swa_rows.shape[1]) == 0:
        swa_rows = q.new_zeros((t, 1, SWA_ROW_BYTES), dtype=torch.uint8)
        swa_valid = q.new_zeros((t, 1), dtype=torch.uint8)
    if kv_rows.numel() == 0 or int(kv_rows.shape[1]) == 0:
        kv_rows = q.new_zeros((t, 1, KV_ROW_BYTES), dtype=torch.uint8)
        kv_valid = q.new_zeros((t, 1), dtype=torch.uint8)
    h = int(q.shape[1])
    pad = (16 - (h % 16)) % 16
    q_in = _contig(q)
    sink32 = sink.float() if sink.dtype != torch.float32 else sink
    if pad:
        q_in = torch.nn.functional.pad(q_in, (0, 0, 0, pad))
        sink32 = torch.nn.functional.pad(_contig(sink32), (0, pad))
    out = torch.empty_like(q_in)
    _module().sparse_prefill(
        out,
        q_in,
        _contig(swa_rows),
        _contig(kv_rows),
        _contig(swa_valid.to(torch.uint8)),
        _contig(kv_valid.to(torch.uint8)),
        _contig(sink32),
        float(softmax_scale),
    )
    return out[:, :h] if pad else out


def sparse_decode_indexed(
    q: torch.Tensor,
    swa_rows: torch.Tensor,
    kv_table: torch.Tensor,
    kv_idx: torch.Tensor,
    positions: torch.Tensor,
    sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """One-query sparse attn over a SWA ring and an indexed KV table.

    q [H, 512] fp16; ``kv_idx`` int32 [K], negative entries skipped.
    Ring validity is ``remainder(pos - slot, W) <= pos``.
    """
    if kv_table.numel() == 0:
        kv_table = q.new_empty((0, KV_ROW_BYTES), dtype=torch.uint8)
        kv_idx = q.new_empty((0,), dtype=torch.int32)
    out = torch.empty_like(q)
    sink32 = sink if sink.dtype == torch.float32 else sink.float()
    _module().sparse_decode_indexed(
        out,
        _contig(q),
        swa_rows,
        kv_table,
        _contig(kv_idx.to(torch.int32)),
        _as_i64(positions).view(-1)[:1],
        _contig(sink32),
        float(softmax_scale),
    )
    return out
