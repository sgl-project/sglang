# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""DFlash drafter: the MLA query block of one draft layer at wide row counts in three launches (DraftMlaQueryWide).

DraftMlaQuery (dflash_draft_mlaq.py) runs one 8-CTA cluster per (head, 32-row tile); above 96 rows (draft batches of 13-16 requests x
block 8) its fourth tile needs a second cluster wave and the served five-launch chain runs instead. DraftMlaQueryWide returns the
same outputs as DraftMlaQueryLayer.run:

  q [rows, heads, 576] BF16 (W_UK-absorbed nope 512 | GPT-J-rotated rope 64, unscaled), the FP8 KV row at loc[row] (normed kv_a 512 |
  rotated k_pe 64; plain e4m3 of the BF16 values, k_scale 1), k [rows, 1, 576] BF16 (the same row) and v [rows, 1, 512] (its latent part)

from three stream-ordered launches:

  K1 (rows programs): q_a / kv_a RMSNorm (div.rn mean, rsqrt; BF16 after x rstd gamma), k_pe RoPE, the k / v / FP8 KV-row stores;
  K2 torch.mm(xq, W_qb^T) -> qb [rows, heads * 192] BF16 (cuBLAS, FP32 accumulation);
  K3 (32-row blocks x heads x 4 latent blocks of 128): q[r, h, :512] = bf16(qb nope [r, h] @ W_UK[h]) (FP32 tl.dot); latent block 0
     also writes the rotated rope q[r, h, 512:].

RoPE is GPT-J (interleaved pairs) in FP32 with the served contraction (x cos - y sin = fma(x, cos, -(y sin)), y cos + x sin =
fma(y, cos, x sin)), as DraftMlaQuery's rotate_pair. Rounding points are the served chain's (BF16 after each norm, after Q_B, after
W_UK, after RoPE); FP32 summation orders differ (ulp-level, KL class).

PDL: when Triton has griddepcontrol (triton.language.extra.cuda.gdc_wait), K1 and K3 are launched programmatically dependent on
their stream predecessor (the qkv_a projection, the Q_B GEMM): each loads only static operands (gammas, W_UK) before
griddepcontrol.wait and reads the producer's output / writes anything only after it (the PDL form SGLang / FlashInfer kernels use).
Measured on a 1-GPU chain of 16 draft layers: faster than the served chain at
104-128 rows; without PDL the same kernels are no faster than the served chain.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_DIV_RN = hasattr(tl.math, "div_rn")
try:
    from triton.language.extra.cuda import gdc_wait as _gdc_wait  # noqa: F401

    HAVE_GDC = True
except Exception:  # noqa: BLE001
    HAVE_GDC = False

BLOCK_ROWS = 32


@triton.jit
def _draft_mlaqw_norm_kv(P, P_STRIDE, WQA, WKVA, XQ, KOUT, VOUT, CACHE, C_RS, SLOTS, POS, CS, EPS,
                         USE_DIV_RN: tl.constexpr, PDL: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, 2048)
    m = o < 1536
    ko = tl.arange(0, 512)
    g = tl.load(WQA + o, m, other=0).to(tl.float32)
    wk = tl.load(WKVA + ko).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(P + row * P_STRIDE + o, m, other=0).to(tl.float32)
    if USE_DIV_RN:
        mean = tl.math.div_rn(tl.sum(x * x, 0), 1536.0)
    else:
        mean = tl.sum(x * x, 0) / 1536.0
    rr = tl.rsqrt(mean + EPS)
    tl.store(XQ + row * 1536 + o, ((x * rr) * g).to(tl.bfloat16), m)
    kv = tl.load(P + row * P_STRIDE + 1536 + ko).to(tl.float32)
    if USE_DIV_RN:
        mk = tl.math.div_rn(tl.sum(kv * kv, 0), 512.0)
    else:
        mk = tl.sum(kv * kv, 0) / 512.0
    rk = tl.rsqrt(mk + EPS)
    kvn = ((kv * rk) * wk).to(tl.bfloat16)
    tl.store(KOUT + row * 576 + ko, kvn)
    tl.store(VOUT + row * 512 + ko, kvn)
    i = tl.arange(0, 32)
    pos = tl.load(POS + row).to(tl.int64)
    c = tl.load(CS + pos * 64 + i)
    s = tl.load(CS + pos * 64 + 32 + i)
    xe = tl.load(P + row * P_STRIDE + 2048 + 2 * i).to(tl.float32)
    xo = tl.load(P + row * P_STRIDE + 2049 + 2 * i).to(tl.float32)
    re = tl.fma(xe, c, -(xo * s)).to(tl.bfloat16)
    ro = tl.fma(xo, c, xe * s).to(tl.bfloat16)
    tl.store(KOUT + row * 576 + 512 + 2 * i, re)
    tl.store(KOUT + row * 576 + 513 + 2 * i, ro)
    slot = tl.load(SLOTS + row).to(tl.int64)
    if slot >= 0:
        base = slot * C_RS
        tl.store(CACHE + base + ko, kvn.to(tl.float32))
        tl.store(CACHE + base + 512 + 2 * i, re.to(tl.float32))
        tl.store(CACHE + base + 513 + 2 * i, ro.to(tl.float32))


@triton.jit
def _draft_mlaqw_absorb_rope(Q, Q_R, WUK, W_SH, W_SK, W_SN, OUT, POS, CS, ROWS,
                             HEADS: tl.constexpr, BM: tl.constexpr, PDL: tl.constexpr):
    pm = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    pn = tl.program_id(2)
    rm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    mrow = rm < ROWS
    kk = tl.arange(0, 128)
    rn = pn * 128 + tl.arange(0, 128)
    hw = h
    b = tl.load(WUK + hw * W_SH + kk[:, None] * W_SK + rn[None, :] * W_SN)
    if PDL:
        tl.extra.cuda.gdc_wait()
    a = tl.load(Q + rm[:, None] * Q_R + h * 192 + kk[None, :], mrow[:, None], other=0)
    acc = tl.dot(a, b)
    tl.store(OUT + (rm[:, None] * HEADS + h) * 576 + rn[None, :], acc.to(tl.bfloat16), mrow[:, None])
    if pn == 0:
        i = tl.arange(0, 32)
        pos = tl.load(POS + rm, mrow, other=0).to(tl.int64)
        c = tl.load(CS + pos[:, None] * 64 + i[None, :], mrow[:, None], other=0)
        s = tl.load(CS + pos[:, None] * 64 + 32 + i[None, :], mrow[:, None], other=0)
        qr = Q + rm[:, None] * Q_R + h * 192 + 128
        xe = tl.load(qr + 2 * i[None, :], mrow[:, None], other=0).to(tl.float32)
        xo = tl.load(qr + 2 * i[None, :] + 1, mrow[:, None], other=0).to(tl.float32)
        ob = OUT + (rm[:, None] * HEADS + h) * 576 + 512
        tl.store(ob + 2 * i[None, :], tl.fma(xe, c, -(xo * s)).to(tl.bfloat16), mrow[:, None])
        tl.store(ob + 2 * i[None, :] + 1, tl.fma(xo, c, xe * s).to(tl.bfloat16), mrow[:, None])


class DraftMlaQueryWide:
    """DraftMlaQueryLayer's contract for one DFlash MLA draft layer (``attn``) over its FP8 pool rows ``kv_rows`` [slots, 1, 576] uint8:
    ``run(qkv, positions, loc)`` -> (q, k [T, 1, 576], v [T, 1, 512]) after storing the FP8 KV rows at ``loc``, or None (shape it does not
    serve, or more than ``max_rows`` rows: the caller keeps the served chain)."""

    def __init__(self, attn, kv_rows, max_rows=128, pdl=None):
        self.heads = attn.num_local_heads
        self.w_qa = attn.q_a_layernorm.weight
        self.w_kva = attn.kv_a_layernorm.weight
        self.w_qb = attn.q_b_proj.weight
        self.w_uk = attn.w_kc.contiguous()  # [heads, 128, 512] row-major (w_kc itself is stored [heads, 512, 128])
        self.eps = float(attn.q_a_layernorm.variance_epsilon)
        self.cos_sin = attn.rotary_emb.cos_sin_cache
        self.kv_rows = kv_rows
        self.max_rows = int(max_rows)
        self.pdl = HAVE_GDC if pdl is None else bool(pdl)
        if self.pdl and not HAVE_GDC:
            raise ValueError("DraftMlaQueryWide: pdl=True needs Triton's griddepcontrol (triton.language.extra.cuda.gdc_wait)")
        h = self.heads
        if (self.w_qa.shape != (1536,) or self.w_kva.shape != (512,) or self.w_qb.shape != (h * 192, 1536) or self.w_uk.shape != (h, 128, 512)
                or any(t.dtype != torch.bfloat16 for t in (self.w_qa, self.w_kva, self.w_qb, self.w_uk)) or not self.w_qb.is_contiguous()):
            raise ValueError("DraftMlaQueryWide: K3 MLA draft shapes (q_a 1536, kv_a 512, Q_B heads x 192 rows, W_UK [heads, 128, 512]; BF16)")
        if self.cos_sin.dtype != torch.float32 or self.cos_sin.dim() != 2 or self.cos_sin.shape[1] != 64:
            raise ValueError("DraftMlaQueryWide: FP32 [max_pos, 64] cos / sin cache")
        if kv_rows.dtype != torch.uint8 or kv_rows.shape[-1] != 576 or kv_rows.stride(-1) != 1:
            raise ValueError("DraftMlaQueryWide: [slots, 1, 576] uint8 pool rows")
        self.calls = 0

    def form(self) -> str:
        return f"DraftMlaQueryWide heads {self.heads} rows <= {self.max_rows} block {BLOCK_ROWS} pdl {self.pdl} div_rn {_DIV_RN}"

    def run(self, qkv, positions, loc):
        rows, h, dev, bf = qkv.shape[0], self.heads, qkv.device, torch.bfloat16
        if not (0 < rows <= self.max_rows) or qkv.shape[1] < 2112 or qkv.stride(1) != 1 or positions.shape != (rows,) or loc.shape != (rows,):
            return None
        xq = torch.empty(rows, 1536, dtype=bf, device=dev)
        qb = torch.empty(rows, h * 192, dtype=bf, device=dev)
        q = torch.empty(rows, h, 576, dtype=bf, device=dev)
        k = torch.empty(rows, 576, dtype=bf, device=dev)
        v = torch.empty(rows, 512, dtype=bf, device=dev)
        c8 = self.kv_rows.view(torch.float8_e4m3fn)
        kw = {"launch_pdl": True} if self.pdl else {}
        _draft_mlaqw_norm_kv[(rows,)](qkv, qkv.stride(0), self.w_qa, self.w_kva, xq, k, v, c8, c8.stride(0), loc, positions, self.cos_sin,
                                      self.eps, _DIV_RN, self.pdl, num_warps=8, **kw)
        torch.mm(xq, self.w_qb.t(), out=qb)
        w = self.w_uk
        _draft_mlaqw_absorb_rope[(triton.cdiv(rows, BLOCK_ROWS), h, 4)](qb, qb.stride(0), w, w.stride(0), w.stride(1), w.stride(2), q, positions,
                                                                        self.cos_sin, rows, h, BLOCK_ROWS, self.pdl, num_warps=4, **kw)
        self.calls += 1
        return q, k.unsqueeze(1), v.unsqueeze(1)


class DraftMlaQueryRouted:
    """SGLANG_OPT_DFLASH_DRAFT_MLAQ_WIDE_MAX_ROWS: rows <= ``cap`` -> ``small`` (DraftMlaQueryLayer), cap < rows <= wide.max_rows -> ``wide``
    (DraftMlaQueryWide), more -> None (the served chain)."""

    def __init__(self, small, wide, cap):
        self.small, self.wide, self.cap = small, wide, int(cap)

    def run(self, qkv, positions, loc):
        rows = qkv.shape[0]
        if rows <= self.cap:
            return self.small.run(qkv, positions, loc)
        return self.wide.run(qkv, positions, loc)
