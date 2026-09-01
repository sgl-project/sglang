"""FP8 SoA dispatch for the unified_kv path (aiter kernels).

Twin of :mod:`runtime` (the bf16 Triton path). Same unified_kv row addressing
and the same ragged ``kv_indices``/``kv_indptr`` streams built by
``build_decode_streams`` / ``build_prefill_indices``; the only differences:

  * the unified store is a two-buffer SoA (fp8 nope_scale [rows,512] +
    bf16 rope [rows,64]) instead of one bf16 [rows,512] buffer;
  * decode dispatches aiter ``mla_decode_fwd_v4_nm`` (#3112) and prefill
    dispatches aiter ``pa_sparse_prefill_fp8_opus`` (#3751), both of which read
    the SoA buffers directly with NO in-kernel dequant.

DSV4 MQA head_dim = 448 nope + 64 rope = 512; v_head_dim = 512. aiter's fixed
"V4-Pro" nope_scale block is 512 B = [448 fp8 | 14 e8m0 (dup) | 50 pad].
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.atom_soa_pack import pack_to_atom_soa
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)

HEAD_DIM = 512      # 448 nope + 64 rope
NOPE_BLOCK = 512    # packed: 448 fp8 + 14 e8m0 + 50 pad
ROPE_DIM = 64
V_HEAD_DIM = 512


# ---------------------------------------------------------------------------
# Quantization helpers (bf16 [.,512] -> SoA fp8 nope_scale [.,512] + rope [.,64])
# ---------------------------------------------------------------------------
def quant_to_soa(k_bf16: torch.Tensor):
    """[N, 512] bf16 -> (nope_scale [N,512] uint8, rope [N,64] bf16)."""
    assert k_bf16.shape[-1] == HEAD_DIM
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(k_bf16.contiguous())
    return pack_to_atom_soa(pack)  # ([N,512]u8, [N,64]bf16)


def quant_q_to_soa(q: torch.Tensor):
    """[T, H, 512] bf16 query -> (q_nope [T,H,512] fp8, q_rope [T,H,64] bf16)."""
    t, h, d = q.shape
    assert d == HEAD_DIM
    ns, rp = quant_to_soa(q.reshape(t * h, d))
    q_nope = ns.view(torch.float8_e4m3fn).view(t, h, NOPE_BLOCK)
    q_rope = rp.view(t, h, ROPE_DIM)
    return q_nope, q_rope


# ---------------------------------------------------------------------------
# Unified store (flat row scatter; unified rows are page_size-1 addressed)
# ---------------------------------------------------------------------------
def _scatter_soa(nope_buf, rope_buf, loc, ns_dense, rp_dense):
    loc = loc.long()
    nope_buf.view(torch.uint8)[loc] = ns_dense
    rope_buf[loc] = rp_dense


def store_swa_into_unified_fp8(
    *,
    kv: torch.Tensor,             # [T, 512] bf16
    state_slot: torch.Tensor,     # [T] int
    positions: torch.Tensor,      # [T] int
    nope_buf: torch.Tensor,       # [rows, 512] fp8/uint8
    rope_buf: torch.Tensor,       # [rows, 64]  bf16
    win: int,
    ring_stride: int,
    final_pos: Optional[torch.Tensor] = None,
) -> None:
    """Quantize this fwd's SWA K to SoA fp8 and scatter into the ring rows."""
    n_rows = kv.shape[0]
    if n_rows == 0:
        return
    pos = positions.to(torch.int64)
    keep = slice(None)
    if final_pos is not None:
        fp = final_pos.to(torch.int64)
        mask = pos > (fp - win)
        if not bool(mask.all()):
            keep = mask
            kv = kv[keep]
            pos = pos[keep]
            state_slot = state_slot[keep]
    loc = state_slot.to(torch.int64) * ring_stride + (pos % ring_stride)
    ns_dense, rp_dense = quant_to_soa(kv)
    _scatter_soa(nope_buf, rope_buf, loc, ns_dense, rp_dense)


def store_compress_into_unified_fp8(
    *,
    kv_compressed: torch.Tensor,  # [M, 512] bf16 (post norm+rope)
    out_loc: torch.Tensor,        # [M] int -- absolute unified row ids
    nope_buf: torch.Tensor,
    rope_buf: torch.Tensor,
) -> None:
    """Quantize compressed K rows to SoA fp8 and scatter into compressed rows."""
    if kv_compressed.shape[0] == 0:
        return
    ns_dense, rp_dense = quant_to_soa(kv_compressed.bfloat16())
    _scatter_soa(nope_buf, rope_buf, out_loc, ns_dense, rp_dense)


# ---------------------------------------------------------------------------
# Decode dispatch: mla_decode_fwd_v4_nm
# ---------------------------------------------------------------------------
def decode(
    *,
    q: torch.Tensor,              # [T, H, 512] bf16 query (448 nope + 64 rope)
    nope_buf: torch.Tensor,       # [rows, 512] fp8 unified nope_scale
    rope_buf: torch.Tensor,       # [rows, 64]  bf16 unified rope
    kv_indices: torch.Tensor,     # [total_kv] int32 -- ragged unified row ids
    kv_indptr: torch.Tensor,      # [T+1] int32
    attn_sink: torch.Tensor,      # [H] fp32
    softmax_scale: float,
) -> torch.Tensor:
    from aiter.mla import mla_decode_fwd_v4_nm

    t, h, _ = q.shape
    q_nope, q_rope = quant_q_to_soa(q)                       # [T,H,512]fp8, [T,H,64]bf16
    rows = nope_buf.shape[0]
    kv_buffer = nope_buf.view(torch.float8_e4m3fn).view(rows, 1, 1, NOPE_BLOCK)
    kvrope = rope_buf.view(rows, 1, 1, ROPE_DIM)

    qo_indptr = torch.arange(t + 1, device=q.device, dtype=torch.int32)  # 1 query/seq
    kv_page_indices = kv_indices.to(torch.int32)
    kv_indptr = kv_indptr.to(torch.int32)

    output = torch.empty((t, h, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    sink = attn_sink
    if sink.shape[0] != h:
        sink = sink[:h]
    sink = sink.to(torch.float32).contiguous()

    mla_decode_fwd_v4_nm(
        q_nope,
        q_rope,
        kv_buffer,
        kvrope,
        output,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        1,  # max_seqlen_q (decode: 1 query per seq)
        sink=sink,
    )
    return output


# ---------------------------------------------------------------------------
# Prefill dispatch: pa_sparse_prefill_fp8_opus
# ---------------------------------------------------------------------------
def prefill(
    *,
    q: torch.Tensor,                  # [T, H, 512] bf16
    nope_buf: torch.Tensor,           # [rows, 512] fp8 unified nope_scale (prefix)
    rope_buf: torch.Tensor,           # [rows, 64]  bf16 unified rope (prefix)
    kv_indices_prefix: torch.Tensor,  # [total_prefix] int32
    kv_indptr_prefix: torch.Tensor,   # [T+1] int32
    kv_extend: torch.Tensor,          # [total_tokens, 512] bf16 (this fwd's K)
    kv_indices_extend: torch.Tensor,  # [total_extend] int32
    kv_indptr_extend: torch.Tensor,   # [T+1] int32
    attn_sink: torch.Tensor,          # [H] fp32
    softmax_scale: float,
) -> torch.Tensor:
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus

    t, h, _ = q.shape
    q_nope, q_rope = quant_q_to_soa(q.contiguous())

    rows = nope_buf.shape[0]
    uk_nope = nope_buf.view(torch.float8_e4m3fn).view(rows, NOPE_BLOCK)
    uk_rope = rope_buf.view(rows, ROPE_DIM)

    # Extend K (this fwd's fresh K) quantized to the same SoA layout.
    ext_ns, ext_rp = quant_to_soa(kv_extend.contiguous())
    ext_nope = ext_ns.view(torch.float8_e4m3fn).view(-1, NOPE_BLOCK)
    ext_rope = ext_rp.view(-1, ROPE_DIM)

    sink = attn_sink
    if sink.shape[0] != h:
        sink = sink[:h]
    sink = sink.to(torch.float32).contiguous()

    return pa_sparse_prefill_fp8_opus(
        q_nope,
        q_rope,
        uk_nope,
        uk_rope,
        kv_indices_prefix.to(torch.int32),
        kv_indptr_prefix.to(torch.int32),
        ext_nope,
        ext_rope,
        kv_indices_extend.to(torch.int32),
        kv_indptr_extend.to(torch.int32),
        sink,
        float(softmax_scale),
    )
