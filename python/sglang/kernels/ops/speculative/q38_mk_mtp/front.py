"""M1-only QK Gemma/RoPE + Deferred-index KV preparation prototype.

The original QKV projection and native paged attention remain separate. Each
pack CTA independently transforms the current K, so it never consumes another
CTA's newly written cache entry. Static selection, dynamic tail, LOC0, counts
and BF16 cache boundaries follow qsa.fused_kv. No indexer is recomputed.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.qsa.fused_kv import _mtp_lookup_positions
from sglang.srt.layers.attention.qsa.mtp_lookup import DeferredMtpSparseIndices


@dataclass(frozen=True)
class MtpFrontPlan:
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    locations: torch.Tensor
    req_to_token: torch.Tensor
    row_requests: torch.Tensor
    indices: DeferredMtpSparseIndices
    lengths: torch.Tensor
    counts: torch.Tensor
    packed_k: torch.Tensor
    packed_v: torch.Tensor

    @property
    def prepared_kv(self):
        return self.counts, self.packed_k, self.packed_v

    def prepare(self, q_gate, k, v, q_gamma, k_gamma, cos_sin, positions,
                axis_map, *, eps):
        return prepare_front(
            q_gate, k, v, q_gamma, k_gamma, cos_sin, positions, axis_map,
            self.key_cache, self.value_cache, self.locations, self.req_to_token,
            self.row_requests, self.indices, self.lengths, self.counts,
            self.packed_k, self.packed_v, eps=eps,
        )


@triton.jit
def _norm_rope(X, Weight, CosSin, Positions, Axes, COS_STRIDE: tl.constexpr,
               POS_STRIDE: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr,
               EPS: tl.constexpr, MROPE: tl.constexpr):
    d = tl.arange(0, D)
    value = tl.load(X + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, 0) / D + EPS)
    normalized = (value * inv * (Weight + 1.0)).to(tl.bfloat16).to(tl.float32)
    # Reuse the parent's independently BF16-rounded normalization operands.
    half: tl.constexpr = ROT // 2
    r = d % half
    x1 = tl.gather(normalized, r, 0)
    x2 = tl.gather(normalized, r + half, 0)
    if MROPE:
        axis = tl.load(Axes + r)
        position = tl.load(Positions + axis * POS_STRIDE)
    else:
        position = tl.load(Positions)
    cosine = tl.load(CosSin + position.to(tl.int64) * COS_STRIDE + r).to(tl.float32)
    sine = tl.load(CosSin + position.to(tl.int64) * COS_STRIDE + half + r).to(tl.float32)
    first = x1 * cosine - x2 * sine
    second = x2 * cosine + x1 * sine
    return tl.where(d < half, first, tl.where(d < ROT, second, normalized)).to(tl.bfloat16)


@triton.jit
def _q38_mk_mtp_front(
    QG, K, V, QGamma, KGamma, CosSin, Positions, Axes,
    KC, VC, Loc, Map, RowRequest, Length, Idx, LookupReq, CapturedLen, LookupPos,
    QOut, GateOut, PackedK, PackedV, Counts,
    QH: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr,
    EPS: tl.constexpr, COS_STRIDE: tl.constexpr, POS_STRIDE: tl.constexpr,
    MAP_STRIDE: tl.constexpr, T: tl.constexpr, S: tl.constexpr,
    TAIL: tl.constexpr, B: tl.constexpr, BT: tl.constexpr, MROPE: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, D)
    gamma = tl.load(tl.where(pid < QH, QGamma + d, KGamma + d)).to(tl.float32)
    # Dynamic QKV, positions and lookup metadata are read after the producer.
    tl.extra.cuda.gdc_wait()
    if pid < QH:
        q = _norm_rope(QG + pid * 2 * D, gamma, CosSin, Positions, Axes,
                       COS_STRIDE, POS_STRIDE, D, ROT, EPS, MROPE)
        tl.store(QOut + pid * D + d, q)
        gate = tl.load(QG + pid * 2 * D + D + d)
        tl.store(GateOut + pid * D + d, gate)
    else:
        tiles: tl.constexpr = triton.cdiv(S, B) + 2
        kv_pid = pid - QH
        head, tile = kv_pid // tiles, kv_pid % tiles
        slot = tl.load(Loc).to(tl.int64)
        if tile < triton.cdiv(S, B):
            c = tile * B + tl.arange(0, B)
            pos = _mtp_lookup_positions(Idx, LookupReq, CapturedLen, LookupPos,
                                        0, c, T, TAIL)
            length = tl.load(Length)
            active = slot != 0
            valid = (pos >= 0) & (pos < length) & (c < T) & active
            request = tl.load(RowRequest).to(tl.int64)
            physical = tl.load(Map + request * MAP_STRIDE + pos, valid, 0).to(tl.int64)
            fresh = valid & (physical == slot)
            offset = physical[:, None] * KVH * D + head * D + d[None, :]
            # Mask the current location out of old-cache loads, preserving the
            # original direct-NK/NV path even while a sibling CTA writes it back.
            old_k = tl.load(KC + offset, valid[:, None] & ~fresh[:, None], 0.0)
            old_v = tl.load(VC + offset, valid[:, None] & ~fresh[:, None], 0.0)
            new_k = _norm_rope(K + head * D, gamma, CosSin, Positions, Axes,
                               COS_STRIDE, POS_STRIDE, D, ROT, EPS, MROPE)
            new_v = tl.load(V + head * D + d)
            k = tl.where(fresh[:, None], new_k[None, :], old_k)
            v = tl.where(fresh[:, None], new_v[None, :], old_v)
            out = c[:, None].to(tl.int64) * KVH * D + head * D + d[None, :]
            tl.store(PackedK + out, k, (c < S)[:, None])
            tl.store(PackedV + out, v, (c < S)[:, None])
        elif tile == triton.cdiv(S, B):
            new_k = _norm_rope(K + head * D, gamma, CosSin, Positions, Axes,
                               COS_STRIDE, POS_STRIDE, D, ROT, EPS, MROPE)
            new_v = tl.load(V + head * D + d)
            tl.store(KC + slot * KVH * D + head * D + d, new_k, slot != 0)
            tl.store(VC + slot * KVH * D + head * D + d, new_v, slot != 0)
        elif head == 0:
            c = tl.arange(0, BT)
            pos = _mtp_lookup_positions(Idx, LookupReq, CapturedLen, LookupPos,
                                        0, c, T, TAIL)
            length = tl.load(Length)
            count = tl.sum(((pos >= 0) & (pos < length)).to(tl.int32))
            tl.store(Counts, count)
    tl.extra.cuda.gdc_launch_dependents()


def prepare_front(q_gate, k, v, q_gamma, k_gamma, cos_sin, positions, axis_map,
                  k_cache, v_cache, loc, req_to_token, row_requests, indices,
                  sequence_lengths, counts, packed_k, packed_v, *, eps,
                  q_heads=6, kv_heads=1, rotary_dim=64):
    """Low-level BF16 M1 component; no serving fallback or loader integration."""
    assert isinstance(indices, DeferredMtpSparseIndices)
    assert q_gate.shape == (1, q_heads * 512)
    assert k.shape == v.shape == (1, kv_heads * 256)
    assert k_cache.dtype == v_cache.dtype == q_gate.dtype == torch.bfloat16
    assert k_cache.shape == v_cache.shape and k_cache.shape[1:] == (kv_heads, 256)
    assert indices.shape[0] == 1 and positions.ndim in (1, 2)
    assert (axis_map is not None) == (positions.ndim == 2)
    q = torch.empty((1, q_heads, 256), dtype=q_gate.dtype, device=q_gate.device)
    gate = torch.empty_like(q)
    stride, topk = packed_k.shape[0], indices.shape[1]
    block = 16
    _q38_mk_mtp_front[(q_heads + kv_heads * (triton.cdiv(stride, block) + 2),)](
        q_gate, k, v, q_gamma, k_gamma, cos_sin, positions, axis_map,
        k_cache, v_cache, loc, req_to_token, row_requests, sequence_lengths,
        indices.indices, indices.requests, indices.captured_len, indices.positions,
        q, gate, packed_k, packed_v, counts,
        QH=q_heads, KVH=kv_heads, D=256, ROT=rotary_dim, EPS=eps,
        COS_STRIDE=cos_sin.stride(0), POS_STRIDE=positions.stride(0),
        MAP_STRIDE=req_to_token.stride(0), T=topk, S=stride,
        TAIL=indices.tail_width, B=block, BT=triton.next_power_of_2(topk),
        MROPE=axis_map is not None, num_warps=4, launch_pdl=True,
    )
    return q, gate
