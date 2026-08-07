"""Adapter: SGLang unified_kv decode -> aiter ``mla_decode_fwd_v4_nm`` (fp8 ASM).

Bridges SGLang's bf16 unified_kv decode contract onto aiter's fp8 v4 nm MLA
decode kernel. Gated by ``SGLANG_USE_AITER_MLA_DECODE`` in the runtime decode
entry; falls back to the triton path if the kernel/import is unavailable.

Kernel contract (verified against SGLang bf16 baseline, cos>0.999):
  * q          : [T, H, 512] bf16 -> packed to (q_fp8 [T,H,512], q_rope [T,H,64])
  * kv_buffer  : [pages, 1, 1, 512] fp8  (nope448 + 14 dup e8m0 scales + 50 pad)
  * kvrope     : [pages, 1, 1, 64]  bf16
  * qo_indptr  : arange(T+1)          (qseqlen1: one q token == one sequence)
  * kv_indptr  : per-token prefix sum (== SGLang kv_indptr)
  * kv_page_indices : SGLang kv_indices
  * sink       : [H] fp32, REQUIRED
  * result read from ``output`` (single-pass writes bf16 straight into it)

fp8 2-buffer packing reuses SGLang's own triton quant kernel
(``quant_to_nope_fp8_rope_bf16_pack_triton``) which emits
(nope_fp8[448], rope_bf16[64], scale_ue8m0[7]); we assemble the kernel's
512-byte reader layout by duplicating each of the 7 e8m0 scale bytes.
"""
from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)

_D = 512
_D_NOPE = 448
_D_ROPE = 64
_NUM_TILES = 7  # 448 / 64
_SCALE_BYTES = _NUM_TILES * 2  # 14 (each tile scale written twice)

_HAS_AITER: Optional[bool] = None
_mla_decode_fwd_v4_nm = None


def _load_aiter():
    global _HAS_AITER, _mla_decode_fwd_v4_nm
    if _HAS_AITER is not None:
        return _HAS_AITER
    try:
        from aiter.mla import mla_decode_fwd_v4_nm  # noqa: F401

        _mla_decode_fwd_v4_nm = mla_decode_fwd_v4_nm
        _HAS_AITER = True
    except Exception:
        _HAS_AITER = False
    return _HAS_AITER


def pack_native_bf16_to_2buff(x_bf16: torch.Tensor):
    """[..., 512] bf16 -> (nope_scale_buff [..., 512] fp8, rope_buff [..., 64] bf16).

    Byte layout of nope_scale_buff (matches the v4 nm asm reader):
      [ nope (448 fp8) | scale (14 e8m0 = s0,s0,..,s6,s6) | pad (50) ] = 512 B
    """
    leading = x_bf16.shape[:-1]
    n = 1
    for s in leading:
        n *= s
    flat = x_bf16.reshape(n, _D).contiguous()

    pack = quant_to_nope_fp8_rope_bf16_pack_triton(flat)
    nope_fp8 = pack.k_nope_fp8            # [n, 448] fp8
    rope_bf16 = pack.k_rope_bf16          # [n, 64]  bf16
    scale_u8 = pack.scale_k_nope_ue8m0    # [n, 7]   uint8 (e8m0 exponent bytes)

    buff = torch.zeros(n, _D, dtype=nope_fp8.dtype, device=x_bf16.device)
    buff[:, :_D_NOPE] = nope_fp8
    # duplicate-write the 7 e8m0 scale bytes as (s0,s0,s1,s1,...) into [448:462]
    scale_bytes = buff[:, _D_NOPE : _D_NOPE + _SCALE_BYTES].view(torch.uint8)
    scale_bytes[:, 0::2] = scale_u8
    scale_bytes[:, 1::2] = scale_u8

    buff = buff.reshape(*leading, _D)
    rope_bf16 = rope_bf16.reshape(*leading, _D_ROPE)
    return buff, rope_bf16


def is_available() -> bool:
    return _load_aiter()


def aiter_v4nm_paged_decode(
    q: torch.Tensor,            # [T, H, 512] bf16 (local heads)
    unified_kv: torch.Tensor,   # [pages, 512] bf16   OR pre-packed fp8 (see kv_fp8)
    kv_indices: torch.Tensor,   # [total_indices] int32 (per-token flat)
    kv_indptr: torch.Tensor,    # [T+1] int32 (per-token prefix sum)
    attn_sink: torch.Tensor,    # [H] fp32
    softmax_scale: float,       # ignored by kernel (hardcodes 1/sqrt(512))
    *,
    kv_fp8: Optional[torch.Tensor] = None,   # [pages,512] fp8 packed (store-time)
    kv_rope: Optional[torch.Tensor] = None,  # [pages,64]  bf16 (store-time)
) -> torch.Tensor:
    """Run aiter v4 nm decode; returns out [T, H, 512] bf16.

    KV source: if ``kv_fp8``/``kv_rope`` are provided (store-time fp8 pool), they
    are used directly; otherwise ``unified_kv`` (bf16) is packed on the fly
    (correctness/dispatch proof — adds a per-step quant of the whole pool rows
    referenced, so NOT a clean perf number).
    """
    assert _load_aiter(), "aiter mla_decode_fwd_v4_nm unavailable"
    T, H, D = q.shape
    assert D == _D
    dev = q.device

    q_fp8, q_rope = pack_native_bf16_to_2buff(q)  # [T,H,512]fp8 / [T,H,64]

    if kv_fp8 is None or kv_rope is None:
        # On-the-fly repack of ONLY the gathered pages (bounded per step, unlike
        # packing the whole pool). Gather the referenced bf16 rows, pack them to
        # a compact fp8 2-buffer, and reindex kv_page_indices to arange. The
        # kernel does identical work (same total KV entries) so mla_a8w8 self-
        # time is representative; the pack is a separate (ignorable) op in the
        # trace. Correct because it reads the real bf16 KV. NOT cuda-graph safe
        # (variable gather size) -> run with --disable-cuda-graph.
        # kv_indices is a RAGGED-PACKED buffer allocated at worst-case width;
        # only kv_indices[:kv_indptr[-1]] are valid, the tail is uninitialized
        # garbage. Gather ONLY the valid prefix (gathering the garbage tail
        # faults the GPU with an OOB read).
        total_valid = int(kv_indptr[-1].item())
        gathered = unified_kv.index_select(
            0, kv_indices[:total_valid].long()
        )  # [total_valid, 512]
        kv_fp8, kv_rope = pack_native_bf16_to_2buff(gathered)
        kv_indices = torch.arange(total_valid, device=dev, dtype=torch.int32)

    P = kv_fp8.shape[0]
    kv_buffer = kv_fp8.view(P, 1, 1, _D)
    kvrope = kv_rope.view(P, 1, 1, _D_ROPE).contiguous()

    if attn_sink.dtype != torch.float32:
        attn_sink = attn_sink.float()
    attn_sink = attn_sink.contiguous()

    qo_indptr = torch.arange(0, T + 1, dtype=torch.int32, device=dev)
    output = torch.empty(T, H, _D, dtype=torch.bfloat16, device=dev)

    _mla_decode_fwd_v4_nm(
        q=q_fp8,
        qrope=q_rope.contiguous(),
        kv_buffer=kv_buffer,
        kvrope=kvrope,
        output=output,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr.to(torch.int32),
        kv_page_indices=kv_indices.to(torch.int32),
        max_seqlen_q=1,
        sink=attn_sink,
        num_kv_splits=1,
    )
    return output
